# src/derivatives/option_pricing_utils.py

import logging
import numpy as np
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def implied_volatility_call(price: float, S: float, K: float, T: float, r: float) -> float:
    """
    Calculates the implied volatility of a European call option using the Black-Scholes model
    and the market price via the brentq numerical method.

    Args:
        price (float): Market price of the call option.
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate (annualized).

    Returns:
        float: The implied volatility, or NaN if no solution is found.
    """
    from src.derivatives.black_scholes import black_scholes_call

    def price_diff(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price

    low_vol = 0.001
    high_vol = 5.0 # A sufficiently high volatility

    try:
        implied_vol = brentq(price_diff, low_vol, high_vol)
        return implied_vol
    except ValueError:
        logging.warning(f"Implied volatility for call option not found within the search range.")
        return np.nan
    except Exception as e:
        logging.error(f"An error occurred while calculating implied volatility for call: {e}")
        return np.nan

# You can add a similar function for put options if needed.

if __name__ == "__main__":
    # Example usage
    market_call_price = 10.50
    S = 100.0
    K = 105.0
    T = 1.0
    r = 0.05

    implied_vol = implied_volatility_call(market_call_price, S, K, T, r)
    if not np.isnan(implied_vol):
        print(f"Implied Volatility (Call): {implied_vol:.4f}")