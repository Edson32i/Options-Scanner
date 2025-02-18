import os
import asyncio
import aiohttp
import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from math import log, sqrt
from scipy.stats import norm
from cachetools import TTLCache
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client  # for text alerts
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import yfinance as yf  # For earnings data
import altair as alt

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# Environment Variables / Credentials
# -----------------------------------------------------------------------------
TASTYTRADE_API_URL = "https://api.tastytrade.com"
USERNAME = os.getenv("TASTYTRADE_USERNAME")
PASSWORD = os.getenv("TASTYTRADE_PASSWORD")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER   = os.getenv("TWILIO_TO_NUMBER")

# -----------------------------------------------------------------------------
# Global Caches
# -----------------------------------------------------------------------------
options_chain_cache = TTLCache(maxsize=1000, ttl=300)
bid_ask_cache = TTLCache(maxsize=5000, ttl=300)

# -----------------------------------------------------------------------------
# Liquidity and Strategy Thresholds
# -----------------------------------------------------------------------------
MIN_UNDERLYING_VOLUME = 1000000   # at least 1 million daily volume
MIN_STRIKES_COUNT = 12            # at least 12 strikes available
MAX_SPREAD_PCT = 0.05             # maximum 5% bid/ask spread relative to mid price
STOP_LOSS_MULTIPLIER = 2.0        # 200% of credit received for effective max loss

# -----------------------------------------------------------------------------
# Black-Scholes Delta Calculation (Iron Condor Net Delta)
# -----------------------------------------------------------------------------
def black_scholes_delta(s, k, T, sigma, r, option_type):
    """
    Compute the Black-Scholes delta for a single option.
    s: underlying price
    k: strike
    T: time to expiration (years)
    sigma: volatility
    r: risk-free rate
    option_type: "call" or "put"
    """
    d1 = (log(s/k) + (r + sigma**2/2) * T) / (sigma * sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def calculate_net_delta_iron_condor(s, lower_strike, upper_strike, T, sigma, r):
    """
    Net delta for a short put at lower_strike + short call at upper_strike.
    """
    delta_put = black_scholes_delta(s, lower_strike, T, sigma, r, "put")
    delta_call = black_scholes_delta(s, upper_strike, T, sigma, r, "call")
    return - (delta_put + delta_call)

# -----------------------------------------------------------------------------
# Improved POP Estimation (Analytic + Monte Carlo)
# -----------------------------------------------------------------------------
def analytic_lognormal_pop(S, L, U, T, sigma, r):
    pop_L = norm.cdf((np.log(L/S) - (r - 0.5*sigma**2)*T) / (sigma*sqrt(T)))
    pop_U = norm.cdf((np.log(U/S) - (r - 0.5*sigma**2)*T) / (sigma*sqrt(T)))
    return (pop_U - pop_L) * 100

def monte_carlo_pop(S, L, U, T, sigma, r, N_sim=100000):
    Z = np.random.standard_normal(N_sim)
    S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*sqrt(T)*Z)
    pop_est = np.mean((S_T >= L) & (S_T <= U)) * 100
    return pop_est

def improved_pop_estimate(S, L, U, T, sigma, r=0.01, N_sim=100000, vol_skew_func=None):
    """
    Combine analytic lognormal, Monte Carlo, and optional skew-based approach.
    """
    analytic_pop = analytic_lognormal_pop(S, L, U, T, sigma, r)
    mc_pop = monte_carlo_pop(S, L, U, T, sigma, r, N_sim)
    if vol_skew_func is not None:
        sigma_lower = vol_skew_func(L)
        sigma_upper = vol_skew_func(U)
        sigma_eff = (sigma_lower + sigma_upper) / 2
        skew_pop = analytic_lognormal_pop(S, L, U, T, sigma_eff, r)
    else:
        skew_pop = analytic_pop
    combined_pop = (analytic_pop + mc_pop + skew_pop) / 3
    return combined_pop

# -----------------------------------------------------------------------------
# Earnings Data via yfinance
# -----------------------------------------------------------------------------
def get_next_earnings(ticker):
    """
    Return the next earnings date for the ticker using yfinance, or None if not found.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if not cal.empty and 'Earnings Date' in cal.index:
            return cal.loc['Earnings Date'][0]
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching earnings for {ticker}: {e}")
        return None

# -----------------------------------------------------------------------------
# Payoff Calculation: "Pure" at-expiration (ignoring stop-loss)
# -----------------------------------------------------------------------------
def payoff_iron_condor(prices, lower_put, upper_call, net_credit):
    payoffs = []
    for p in prices:
        if p < lower_put:
            payoffs.append(net_credit - (lower_put - p))
        elif p > upper_call:
            payoffs.append(net_credit - (p - upper_call))
        else:
            payoffs.append(net_credit)
    return payoffs

def payoff_vertical_bull_put(prices, sold_put_strike, protective_put_strike, net_credit):
    payoffs = []
    spread_width = sold_put_strike - protective_put_strike
    for p in prices:
        if p < protective_put_strike:
            payoffs.append(net_credit - spread_width)
        elif protective_put_strike <= p < sold_put_strike:
            payoffs.append(net_credit - (sold_put_strike - p))
        else:
            payoffs.append(net_credit)
    return payoffs

def payoff_bear_call(prices, sold_call_strike, protective_call_strike, net_credit):
    payoffs = []
    spread_width = protective_call_strike - sold_call_strike
    for p in prices:
        if p > protective_call_strike:
            payoffs.append(net_credit - spread_width)
        elif sold_call_strike < p <= protective_call_strike:
            payoffs.append(net_credit - (p - sold_call_strike))
        else:
            payoffs.append(net_credit)
    return payoffs

def generate_payoff_chart(row):
    """
    Build a payoff chart for the trade in 'row' using Altair.
    """
    current_price = row["Stock Price"]
    strategy = row["Strategy"]
    net_credit = row["Credit Received"]

    # Price range ±30% around current_price
    prices = np.linspace(0.7 * current_price, 1.3 * current_price, 101)

    if strategy == "Iron Condor":
        lower_put = row["Lower Put"]
        upper_call = row["Upper Call"]
        payoffs = payoff_iron_condor(prices, lower_put, upper_call, net_credit)
    elif strategy == "Vertical Spread":
        sold_put_strike = row["Sold Put Strike"]
        protective_put_strike = row["Protective Put Strike"]
        payoffs = payoff_vertical_bull_put(prices, sold_put_strike, protective_put_strike, net_credit)
    elif strategy == "Bear Call Spread":
        sold_call_strike = row["Sold Call Strike"]
        protective_call_strike = row["Protective Call Strike"]
        payoffs = payoff_bear_call(prices, sold_call_strike, protective_call_strike, net_credit)
    else:
        return None

    df_payoff = pd.DataFrame({"Price": prices, "Payoff": payoffs})
    chart = alt.Chart(df_payoff).mark_line().encode(
        x=alt.X("Price", title="Underlying Price at Expiration"),
        y=alt.Y("Payoff", title="P/L at Expiration"),
        tooltip=["Price", "Payoff"]
    ).interactive()
    return chart

# -----------------------------------------------------------------------------
# Synchronous Tastytrade Login
# -----------------------------------------------------------------------------
def tastytrade_login():
    import requests
    url = f"{TASTYTRADE_API_URL}/sessions"
    payload = {"login": USERNAME, "password": PASSWORD}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("Login response:", data)  # Debug print
        # Try to get the token from "token", "access_token", or "session-token"
        token = data["data"].get("token") or data["data"].get("access_token") or data["data"].get("session-token")
        if token:
            logging.info("Login successful.")
            return token
        else:
            logging.error("Token not found in response: %s", data)
            return None
    except Exception as e:
        logging.error("Login error: %s", e)
        return None


# -----------------------------------------------------------------------------
# Asynchronous API Helpers
# -----------------------------------------------------------------------------
async def async_get_available_tickers(token, session):
    url = f"{TASTYTRADE_API_URL}/markets/options/available-tickers"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers, timeout=10) as response:
        data = await response.json(content_type=None)
        tickers = [item.get("symbol") for item in data.get("data", []) if "symbol" in item]
        return tickers


async def async_get_options_chain(symbol, token, session):
    if symbol in options_chain_cache:
        return symbol, options_chain_cache[symbol]
    url = f"{TASTYTRADE_API_URL}/markets/options/chains/{symbol}"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers, timeout=10) as response:
        data = await response.json()
        if "data" in data:
            options_chain_cache[symbol] = data
            return symbol, data
        else:
            raise Exception(f"No 'data' for {symbol}")

async def async_get_bid_ask_spread(symbol, strike, option_type, token, session):
    key = f"{symbol}-{strike}-{option_type}"
    if key in bid_ask_cache:
        return bid_ask_cache[key]
    url = f"{TASTYTRADE_API_URL}/markets/options/{symbol}/strikes/{strike}/{option_type}"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers, timeout=10) as response:
        data = await response.json()
        data = data.get("data", {})
        try:
            bid = float(data.get("bid", 0))
            ask = float(data.get("ask", 0))
        except Exception as e:
            logging.error("Error parsing bid/ask for %s: %s", key, e)
            return None
        if bid and ask:
            mid = round((bid + ask) / 2, 2)
            spread = round(ask - bid, 2)
            bid_ask_cache[key] = (mid, spread)
            return (mid, spread)
        else:
            logging.warning("Incomplete bid/ask for %s", key)
            return None

# -----------------------------------------------------------------------------
# Basic Calculation Helpers
# -----------------------------------------------------------------------------
def calculate_expected_move(price, iv, days):
    return price * iv * np.sqrt(days / 252)

def calculate_ev(credit_received, effective_loss, pop_value):
    return round((credit_received * (pop_value / 100)) - (effective_loss * ((100 - pop_value) / 100)), 2)

def calculate_risk_reward(credit_received, effective_loss):
    return round(credit_received / effective_loss, 2) if effective_loss else 0

def calculate_margin_requirement(effective_loss):
    return effective_loss

# -----------------------------------------------------------------------------
# Twilio Text Alerts
# -----------------------------------------------------------------------------
def send_text_alert(message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        logging.info("Text alert sent successfully.")
    except Exception as e:
        logging.error("Failed to send text alert: %s", e)

# -----------------------------------------------------------------------------
# Strategy-Specific Processing (with concurrency limiting)
# -----------------------------------------------------------------------------
async def async_process_ticker_iron(symbol, token, days_out, pop_threshold, min_credit_ratio, session, earnings_filter="No Filter"):
    """
    Iron Condor: includes liquidity checks, improved POP, net delta, earnings filter.
    """
    try:
        _, data = await async_get_options_chain(symbol, token, session)
    except Exception as e:
        logging.error(f"Error fetching options chain for {symbol}: {e}")
        return None, None
    try:
        price = float(data["data"].get("underlying-price", 0))
        iv = float(data["data"].get("implied-volatility", 0))
        underlying_volume = float(data["data"].get("underlying-volume", 0))
        if price == 0 or iv == 0:
            return None, None
    except Exception as e:
        logging.error(f"Error extracting underlying data for {symbol}: {e}")
        return None, None

    # Liquidity checks
    if underlying_volume < MIN_UNDERLYING_VOLUME:
        return None, {"Ticker": symbol, "Rejection Reason": "Underlying volume below 1 million"}
    strikes_list = data["data"].get("strikes", [])
    if len(strikes_list) < MIN_STRIKES_COUNT:
        return None, {"Ticker": symbol, "Rejection Reason": "Less than 12 strikes available"}

    # Basic iron condor logic
    expected_move = calculate_expected_move(price, iv, days_out)
    lower_strike = round(price - expected_move, 2)
    upper_strike = round(price + expected_move, 2)

    try:
        bid_ask_put = await async_get_bid_ask_spread(symbol, lower_strike, "put", token, session)
        bid_ask_call = await async_get_bid_ask_spread(symbol, upper_strike, "call", token, session)
    except Exception as e:
        logging.error(f"Error fetching bid/ask for {symbol}: {e}")
        return None, None
    if bid_ask_put is None or bid_ask_call is None:
        return None, None

    put_mid, put_spread = bid_ask_put
    call_mid, call_spread = bid_ask_call
    if put_mid == 0 or call_mid == 0:
        return None, {"Ticker": symbol, "Rejection Reason": "Invalid option mid prices"}
    if (put_spread/put_mid) > MAX_SPREAD_PCT or (call_spread/call_mid) > MAX_SPREAD_PCT:
        return None, {"Ticker": symbol, "Rejection Reason": "Wide bid/ask spreads"}

    credit_received = put_mid + call_mid
    theoretical_max_loss = abs(upper_strike - lower_strike)
    effective_loss = min(theoretical_max_loss, STOP_LOSS_MULTIPLIER * credit_received)

    # Improved POP if the stop-loss cap applies
    pop = None
    # The "theoretical" normal-based pop (for reference):
    # This is a simpler approach ignoring advanced skew:
    adjusted_scale = expected_move * 1.2
    prob_lower = norm.cdf(lower_strike, loc=price, scale=adjusted_scale)
    prob_upper = norm.cdf(upper_strike, loc=price, scale=adjusted_scale)
    theoretical_pop = round((prob_upper - prob_lower) * 100, 2)

    if STOP_LOSS_MULTIPLIER * credit_received < theoretical_max_loss:
        lower_effective = lower_strike + STOP_LOSS_MULTIPLIER * credit_received
        upper_effective = upper_strike - STOP_LOSS_MULTIPLIER * credit_received
        T_years = days_out / 365.0
        r = 0.01
        pop = improved_pop_estimate(price, lower_effective, upper_effective, T_years, iv, r)[0]
    else:
        pop = theoretical_pop

    effective_ev = calculate_ev(credit_received, effective_loss, pop)
    effective_risk_reward = calculate_risk_reward(credit_received, effective_loss)
    margin_req = calculate_margin_requirement(effective_loss)

    # Net delta
    T_years = days_out / 365.0
    r = 0.01
    net_delta = calculate_net_delta_iron_condor(price, lower_strike, upper_strike, T_years, iv, r)

    # Earnings filter
    earnings = await asyncio.to_thread(get_next_earnings, symbol)
    expiration_date = (datetime.now() + timedelta(days=days_out)).date()
    earnings_date = None
    if earnings is not None:
        try:
            earnings_date = earnings.date()
        except Exception:
            earnings_date = None
    if earnings_filter == "Before Expiration":
        # Reject if earnings_date >= expiration_date
        if earnings_date is None or earnings_date >= expiration_date:
            pop = 0
    elif earnings_filter == "After Expiration":
        # Reject if earnings_date <= expiration_date
        if earnings_date is None or earnings_date <= expiration_date:
            pop = 0

    reasons = []
    if pop < pop_threshold:
        reasons.append("Effective POP too low")
    if effective_ev <= 0:
        reasons.append("Effective EV <= 0")
    if effective_risk_reward < min_credit_ratio:
        reasons.append("Effective risk/reward below threshold")

    result = {
        "Ticker": symbol,
        "Strategy": "Iron Condor",
        "Stock Price": price,
        "IV": iv,
        "Lower Put": lower_strike,
        "Upper Call": upper_strike,
        "Days Out": days_out,
        "Theoretical POP": theoretical_pop,
        "Effective POP": pop,
        "Effective EV": effective_ev,
        "Credit Received": credit_received,
        "Theoretical Max Loss": theoretical_max_loss,
        "Effective Max Loss": effective_loss,
        "Effective Risk/Reward": effective_risk_reward,
        "Margin Requirement": margin_req,
        "Net Delta": net_delta,
        "Next Earnings": earnings,
        "Expiration Date": expiration_date
    }
    if reasons:
        result["Rejection Reason"] = ", ".join(reasons)
        return None, result
    return result, None

async def async_process_ticker_vertical(symbol, token, days_out, pop_threshold, min_credit_ratio, session,
                                        spread_width_factor=0.2, earnings_filter="No Filter"):
    """
    Vertical Spread (Bull Put) with improved POP, liquidity checks, earnings filter.
    """
    try:
        _, data = await async_get_options_chain(symbol, token, session)
    except Exception as e:
        logging.error(f"Error fetching options chain for {symbol}: {e}")
        return None, None
    try:
        price = float(data["data"].get("underlying-price", 0))
        iv = float(data["data"].get("implied-volatility", 0))
        underlying_volume = float(data["data"].get("underlying-volume", 0))
        if price == 0 or iv == 0:
            return None, None
    except Exception as e:
        logging.error(f"Error extracting underlying data for {symbol}: {e}")
        return None, None

    if underlying_volume < MIN_UNDERLYING_VOLUME:
        return None, {"Ticker": symbol, "Rejection Reason": "Underlying volume below 1 million"}
    strikes_list = data["data"].get("strikes", [])
    if len(strikes_list) < MIN_STRIKES_COUNT:
        return None, {"Ticker": symbol, "Rejection Reason": "Less than 12 strikes available"}

    expected_move = calculate_expected_move(price, iv, days_out)
    sold_put_strike = round(price - expected_move, 2)
    protective_put_strike = round(sold_put_strike - (spread_width_factor * expected_move), 2)

    try:
        sold_put_data = await async_get_bid_ask_spread(symbol, sold_put_strike, "put", token, session)
        protective_put_data = await async_get_bid_ask_spread(symbol, protective_put_strike, "put", token, session)
    except Exception as e:
        logging.error(f"Error fetching bid/ask for vertical spread {symbol}: {e}")
        return None, None
    if sold_put_data is None or protective_put_data is None:
        return None, None

    sold_put_mid, sold_put_spread = sold_put_data
    protective_put_mid, protective_put_spread = protective_put_data
    if sold_put_mid == 0 or protective_put_mid == 0:
        return None, {"Ticker": symbol, "Rejection Reason": "Invalid option mid prices"}
    if (sold_put_spread/sold_put_mid) > MAX_SPREAD_PCT or (protective_put_spread/protective_put_mid) > MAX_SPREAD_PCT:
        return None, {"Ticker": symbol, "Rejection Reason": "Wide bid/ask spreads for vertical spread"}

    credit_received = sold_put_mid - protective_put_mid
    theoretical_max_loss = (sold_put_strike - protective_put_strike) - credit_received
    effective_loss = min(theoretical_max_loss, STOP_LOSS_MULTIPLIER * credit_received)

    # Basic normal-based pop for reference
    # Then improved pop if the stop applies
    adjusted_scale = expected_move * 1.2
    simple_pop = round((1 - norm.cdf(sold_put_strike, loc=price, scale=adjusted_scale)) * 100, 2)

    T_years = days_out / 365.0
    r = 0.01
    if STOP_LOSS_MULTIPLIER * credit_received < theoretical_max_loss:
        effective_stop = sold_put_strike + STOP_LOSS_MULTIPLIER * credit_received
        # Use large upper bound to approximate infinity
        pop = improved_pop_estimate(price, effective_stop, price * 10, T_years, iv, r)[0]
    else:
        pop = simple_pop

    effective_ev = calculate_ev(credit_received, effective_loss, pop)
    effective_risk_reward = calculate_risk_reward(credit_received, effective_loss)
    margin_req = calculate_margin_requirement(effective_loss)

    earnings = await asyncio.to_thread(get_next_earnings, symbol)
    expiration_date = (datetime.now() + timedelta(days=days_out)).date()
    earnings_date = None
    if earnings is not None:
        try:
            earnings_date = earnings.date()
        except Exception:
            earnings_date = None
    if earnings_filter == "Before Expiration":
        if earnings_date is None or earnings_date >= expiration_date:
            pop = 0
    elif earnings_filter == "After Expiration":
        if earnings_date is None or earnings_date <= expiration_date:
            pop = 0

    reasons = []
    if pop < pop_threshold:
        reasons.append("Effective POP too low")
    if effective_ev <= 0:
        reasons.append("Effective EV <= 0")
    if effective_risk_reward < min_credit_ratio:
        reasons.append("Effective risk/reward below threshold")

    result = {
        "Ticker": symbol,
        "Strategy": "Vertical Spread",
        "Stock Price": price,
        "IV": iv,
        "Sold Put Strike": sold_put_strike,
        "Protective Put Strike": protective_put_strike,
        "Days Out": days_out,
        "Theoretical POP": simple_pop,
        "Effective POP": pop,
        "Effective EV": effective_ev,
        "Credit Received": credit_received,
        "Theoretical Max Loss": theoretical_max_loss,
        "Effective Max Loss": effective_loss,
        "Effective Risk/Reward": effective_risk_reward,
        "Margin Requirement": margin_req,
        "Breakeven": round(sold_put_strike + credit_received, 2),
        "Next Earnings": earnings,
        "Expiration Date": expiration_date
    }
    if reasons:
        result["Rejection Reason"] = ", ".join(reasons)
        return None, result
    return result, None

async def async_process_ticker_bear_call(symbol, token, days_out, pop_threshold, min_credit_ratio, session,
                                         spread_width_factor=0.2, earnings_filter="No Filter"):
    """
    Bear Call Spread with improved POP, liquidity checks, earnings filter.
    """
    try:
        _, data = await async_get_options_chain(symbol, token, session)
    except Exception as e:
        logging.error(f"Error fetching options chain for {symbol}: {e}")
        return None, None
    try:
        price = float(data["data"].get("underlying-price", 0))
        iv = float(data["data"].get("implied-volatility", 0))
        underlying_volume = float(data["data"].get("underlying-volume", 0))
        if price == 0 or iv == 0:
            return None, None
    except Exception as e:
        logging.error(f"Error extracting underlying data for {symbol}: {e}")
        return None, None

    if underlying_volume < MIN_UNDERLYING_VOLUME:
        return None, {"Ticker": symbol, "Rejection Reason": "Underlying volume below 1 million"}
    strikes_list = data["data"].get("strikes", [])
    if len(strikes_list) < MIN_STRIKES_COUNT:
        return None, {"Ticker": symbol, "Rejection Reason": "Less than 12 strikes available"}

    expected_move = calculate_expected_move(price, iv, days_out)
    sold_call_strike = round(price + expected_move, 2)
    protective_call_strike = round(sold_call_strike + (spread_width_factor * expected_move), 2)

    try:
        sold_call_data = await async_get_bid_ask_spread(symbol, sold_call_strike, "call", token, session)
        protective_call_data = await async_get_bid_ask_spread(symbol, protective_call_strike, "call", token, session)
    except Exception as e:
        logging.error(f"Error fetching bid/ask for bear call spread {symbol}: {e}")
        return None, None
    if sold_call_data is None or protective_call_data is None:
        return None, None

    sold_call_mid, sold_call_spread = sold_call_data
    protective_call_mid, protective_call_spread = protective_call_data
    if sold_call_mid == 0 or protective_call_mid == 0:
        return None, {"Ticker": symbol, "Rejection Reason": "Invalid option mid prices"}
    if (sold_call_spread/sold_call_mid) > MAX_SPREAD_PCT or (protective_call_spread/protective_call_mid) > MAX_SPREAD_PCT:
        return None, {"Ticker": symbol, "Rejection Reason": "Wide bid/ask spreads for bear call spread"}

    credit_received = sold_call_mid - protective_call_mid
    theoretical_max_loss = (protective_call_strike - sold_call_strike) - credit_received
    effective_loss = min(theoretical_max_loss, STOP_LOSS_MULTIPLIER * credit_received)

    T_years = days_out / 365.0
    r = 0.01
    # Basic normal-based pop for reference
    adjusted_scale = expected_move * 1.2
    simple_pop = round(norm.cdf(sold_call_strike, loc=price, scale=adjusted_scale) * 100, 2)

    if STOP_LOSS_MULTIPLIER * credit_received < theoretical_max_loss:
        effective_stop = sold_call_strike - STOP_LOSS_MULTIPLIER * credit_received
        # Use a very low lower bound to approximate zero
        pop = improved_pop_estimate(price, 0.01, effective_stop, T_years, iv, r)[0]
    else:
        pop = simple_pop

    effective_ev = calculate_ev(credit_received, effective_loss, pop)
    effective_risk_reward = calculate_risk_reward(credit_received, effective_loss)
    margin_req = calculate_margin_requirement(effective_loss)

    earnings = await asyncio.to_thread(get_next_earnings, symbol)
    expiration_date = (datetime.now() + timedelta(days=days_out)).date()
    earnings_date = None
    if earnings is not None:
        try:
            earnings_date = earnings.date()
        except Exception:
            earnings_date = None
    if earnings_filter == "Before Expiration":
        if earnings_date is None or earnings_date >= expiration_date:
            pop = 0
    elif earnings_filter == "After Expiration":
        if earnings_date is None or earnings_date <= expiration_date:
            pop = 0

    reasons = []
    if pop < pop_threshold:
        reasons.append("Effective POP too low")
    if effective_ev <= 0:
        reasons.append("Effective EV <= 0")
    if effective_risk_reward < min_credit_ratio:
        reasons.append("Effective risk/reward below threshold")

    result = {
        "Ticker": symbol,
        "Strategy": "Bear Call Spread",
        "Stock Price": price,
        "IV": iv,
        "Sold Call Strike": sold_call_strike,
        "Protective Call Strike": protective_call_strike,
        "Days Out": days_out,
        "Theoretical POP": simple_pop,
        "Effective POP": pop,
        "Effective EV": effective_ev,
        "Credit Received": credit_received,
        "Theoretical Max Loss": theoretical_max_loss,
        "Effective Max Loss": effective_loss,
        "Effective Risk/Reward": effective_risk_reward,
        "Margin Requirement": margin_req,
        "Breakeven": round(sold_call_strike - credit_received, 2),
        "Next Earnings": earnings,
        "Expiration Date": expiration_date
    }
    if reasons:
        result["Rejection Reason"] = ", ".join(reasons)
        return None, result
    return result, None

# -----------------------------------------------------------------------------
# Orchestrators for Each Strategy (Scan All Tickers)
# -----------------------------------------------------------------------------
async def async_find_iron_condors(token, days_out, pop_threshold, min_credit_ratio, earnings_filter):
    async with aiohttp.ClientSession() as session:
        tickers = await async_get_available_tickers(token, session)
        sem = asyncio.Semaphore(50)
        tasks = [
            limited_process_iron(symbol, token, days_out, pop_threshold, min_credit_ratio, session, sem, earnings_filter)
            for symbol in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    accepted, rejected = [], []
    for res in results:
        if isinstance(res, Exception):
            continue
        accepted_item, rejected_item = res
        if accepted_item:
            accepted.append(accepted_item)
        elif rejected_item:
            rejected.append(rejected_item)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)

async def async_find_vertical_spreads(token, days_out, pop_threshold, min_credit_ratio, spread_width_factor, earnings_filter):
    async with aiohttp.ClientSession() as session:
        tickers = await async_get_available_tickers(token, session)
        sem = asyncio.Semaphore(50)
        tasks = [
            limited_process_vertical(symbol, token, days_out, pop_threshold, min_credit_ratio, session, spread_width_factor, sem, earnings_filter)
            for symbol in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    accepted, rejected = [], []
    for res in results:
        if isinstance(res, Exception):
            continue
        accepted_item, rejected_item = res
        if accepted_item:
            accepted.append(accepted_item)
        elif rejected_item:
            rejected.append(rejected_item)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)

async def async_find_bear_call_spreads(token, days_out, pop_threshold, min_credit_ratio, spread_width_factor, earnings_filter):
    async with aiohttp.ClientSession() as session:
        tickers = await async_get_available_tickers(token, session)
        sem = asyncio.Semaphore(50)
        tasks = [
            limited_process_bear(symbol, token, days_out, pop_threshold, min_credit_ratio, session, spread_width_factor, sem, earnings_filter)
            for symbol in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    accepted, rejected = [], []
    for res in results:
        if isinstance(res, Exception):
            continue
        accepted_item, rejected_item = res
        if accepted_item:
            accepted.append(accepted_item)
        elif rejected_item:
            rejected.append(rejected_item)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)

# -----------------------------------------------------------------------------
# Scheduled Scanning (Saves to CSV, Sends Text)
# -----------------------------------------------------------------------------
def save_scheduled_results(timestamp, ic_count, vs_count, bc_count):
    df = pd.DataFrame({
        "Timestamp": [timestamp],
        "Iron Condor": [ic_count],
        "Vertical Spread": [vs_count],
        "Bear Call Spread": [bc_count]
    })
    file_path = "scheduled_results.csv"
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

def scheduled_scan():
    logging.info("Starting scheduled scan...")
    token = tastytrade_login()
    if not token:
        logging.error("Scheduled scan: authentication failed.")
        return
    try:
        sd = st.session_state.get("scheduled_days_out", 30)
        sp = st.session_state.get("scheduled_pop_threshold", 70)
        scr = st.session_state.get("scheduled_min_credit_ratio", 0.10)
        swf = st.session_state.get("scheduled_spread_width_factor", 0.2)
        efilt = st.session_state.get("scheduled_earnings_filter", "No Filter")

        accepted_ic, _ = asyncio.run(async_find_iron_condors(token, sd, sp, scr, efilt))
        accepted_vs, _ = asyncio.run(async_find_vertical_spreads(token, sd, sp, scr, swf, efilt))
        accepted_bc, _ = asyncio.run(async_find_bear_call_spreads(token, sd, sp, scr, swf, efilt))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M EST")
        msg = (f"Scheduled Scan at {timestamp}: "
               f"Iron Condor: {len(accepted_ic)} trades, "
               f"Vertical Spread: {len(accepted_vs)} trades, "
               f"Bear Call Spread: {len(accepted_bc)} trades.")
        logging.info(msg)
        send_text_alert(msg)
        save_scheduled_results(timestamp, len(accepted_ic), len(accepted_vs), len(accepted_bc))
    except Exception as e:
        logging.error("Scheduled scan error: %s", e)

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("Tastytrade Option Scanner with Liquidity, Improved POP, Earnings, & Payoff Charts")

# Let the user pick an earnings filter
earnings_filter_option = st.sidebar.selectbox(
    "Earnings Timing Filter",
    options=["No Filter", "Before Expiration", "After Expiration"],
    index=0
)

mode = st.sidebar.radio("Mode", options=["Scan All Tickers", "Scan Specific Ticker", "Backtest & Optimize", "Enable Scheduled Scanning"])

if mode == "Enable Scheduled Scanning":
    st.sidebar.subheader("Scheduled Scan Settings")
    scheduled_days_out = st.sidebar.number_input("Scheduled Days Out", value=30, min_value=1)
    scheduled_pop_threshold = st.sidebar.number_input("Scheduled POP Threshold (%)", value=70, min_value=1, max_value=100)
    scheduled_min_credit_ratio = st.sidebar.number_input("Scheduled Minimum Credit Ratio", value=0.10, min_value=0.0, step=0.01)
    scheduled_spread_width_factor = st.sidebar.number_input("Scheduled Spread Width Factor", value=0.2, min_value=0.0, step=0.05)
    st.session_state["scheduled_days_out"] = scheduled_days_out
    st.session_state["scheduled_pop_threshold"] = scheduled_pop_threshold
    st.session_state["scheduled_min_credit_ratio"] = scheduled_min_credit_ratio
    st.session_state["scheduled_spread_width_factor"] = scheduled_spread_width_factor
    st.session_state["scheduled_earnings_filter"] = earnings_filter_option

    st.write("Scheduled scanning is enabled. The scanner will run 3 times daily at 10:30, 12:30, and 14:30 EST.")
    eastern = pytz.timezone("US/Eastern")
    scheduler = BackgroundScheduler(timezone=eastern)
    trigger = CronTrigger(hour="10,12,14", minute="30", timezone=eastern)
    scheduler.add_job(scheduled_scan, trigger)
    scheduler.start()
    st.write("Scheduler started.")

elif mode == "Scan Specific Ticker":
    ticker_input = st.sidebar.text_input("Enter Ticker (e.g. TLT)", value="TLT")
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies",
        options=["Iron Condor", "Vertical Spread", "Bear Call Spread"],
        default=["Iron Condor", "Bear Call Spread"]
    )
    days_range_str = st.sidebar.text_input("Days Out Range (comma-separated, e.g. 10,20,30,40)", value="10,20,30")
    try:
        days_range = [int(x.strip()) for x in days_range_str.split(",")]
    except Exception:
        days_range = [30]

    if st.sidebar.button("Run Scanner"):
        token = tastytrade_login()
        if token:
            st.write(f"Scanning best trades for {ticker_input} with filter: {earnings_filter_option}")
            for strat in selected_strategies:
                st.write(f"### Best {strat} Trades for {ticker_input}")
                # Build a simple function to find best trades for a single ticker...
                best_results = []
                for d_out in days_range:
                    if strat == "Iron Condor":
                        # We'll just reuse the single async_process function for a single ticker
                        accepted_item, rejected_item = asyncio.run(
                            async_process_ticker_iron(ticker_input, token, d_out, 0, 0, aiohttp.ClientSession(), earnings_filter_option)
                        )
                        if accepted_item:
                            best_results.append(accepted_item)
                    elif strat == "Vertical Spread":
                        accepted_item, rejected_item = asyncio.run(
                            async_process_ticker_vertical(ticker_input, token, d_out, 0, 0, aiohttp.ClientSession(),
                                                          0.2, earnings_filter_option)
                        )
                        if accepted_item:
                            best_results.append(accepted_item)
                    else:
                        accepted_item, rejected_item = asyncio.run(
                            async_process_ticker_bear_call(ticker_input, token, d_out, 0, 0, aiohttp.ClientSession(),
                                                           0.2, earnings_filter_option)
                        )
                        if accepted_item:
                            best_results.append(accepted_item)

                if not best_results:
                    st.warning(f"No acceptable {strat} trades found for {ticker_input}.")
                    continue

                # Convert to DataFrame
                best_df = pd.DataFrame(best_results)
                # For demonstration, let's just show them all
                for idx, row in best_df.iterrows():
                    with st.expander(f"{row['Ticker']} - {row['Strategy']} (DaysOut={row['Days Out']})"):
                        st.write(row)
                        chart = generate_payoff_chart(row)
                        if chart:
                            st.altair_chart(chart, use_container_width=True)

            send_text_alert(f"Scan complete for {ticker_input}. Check dashboard for details.")
        else:
            st.error("Authentication failed.")

elif mode == "Scan All Tickers":
    days_out = st.sidebar.number_input("Days Out", value=30, min_value=1)
    strategy = st.sidebar.selectbox("Select Strategy", ["Iron Condor", "Vertical Spread", "Bear Call Spread"])
    pop_threshold = st.sidebar.number_input("POP Threshold (%)", value=70, min_value=1, max_value=100)
    min_credit_ratio = st.sidebar.number_input("Minimum Credit Ratio", value=0.10, min_value=0.0, step=0.01)
    if strategy in ["Vertical Spread", "Bear Call Spread"]:
        spread_width_factor = st.sidebar.number_input("Spread Width Factor", value=0.2, min_value=0.0, step=0.05)
    else:
        spread_width_factor = None

    if st.sidebar.button("Run Scanner"):
        token = tastytrade_login()
        if token:
            st.write(f"Scanning all tickers for {strategy} with earnings filter: {earnings_filter_option}")
            if strategy == "Iron Condor":
                accepted_df, rejected_df = asyncio.run(
                    async_find_iron_condors(token, days_out, pop_threshold, min_credit_ratio, earnings_filter_option)
                )
            elif strategy == "Vertical Spread":
                accepted_df, rejected_df = asyncio.run(
                    async_find_vertical_spreads(token, days_out, pop_threshold, min_credit_ratio,
                                                spread_width_factor, earnings_filter_option)
                )
            else:
                accepted_df, rejected_df = asyncio.run(
                    async_find_bear_call_spreads(token, days_out, pop_threshold, min_credit_ratio,
                                                 spread_width_factor, earnings_filter_option)
                )

            st.write("Scanning completed.")
            if not accepted_df.empty:
                st.success(f"Found {len(accepted_df)} potential {strategy} trades.")
                # Instead of st.dataframe, we show each in an expander with a payoff chart
                for idx, trade_row in accepted_df.iterrows():
                    with st.expander(f"{trade_row['Ticker']} - {trade_row['Strategy']} (Index={idx})"):
                        st.write(trade_row)
                        # Generate payoff chart
                        chart = generate_payoff_chart(trade_row)
                        if chart:
                            st.altair_chart(chart, use_container_width=True)
                send_text_alert(f"Scan complete: {len(accepted_df)} potential {strategy} trades found.")
            else:
                st.warning("No promising trades were found with the current parameters.")
            if not rejected_df.empty:
                st.write("Some trades were rejected:")
                st.dataframe(rejected_df)
        else:
            st.error("Authentication failed.")

else:  # mode == "Backtest & Optimize" (Placeholder)
    st.write("Backtest & Optimize mode is not fully implemented in this snippet.")
