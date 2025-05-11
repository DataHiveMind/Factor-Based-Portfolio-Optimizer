# src/derivatives/black_scholes.py

import numpy as np
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the price of a European call option using the Black-Scholes model.

    Args:
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).

    Returns:
        float: The price of the European call option.
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    except ZeroDivisionError:
        logging.error("Error in Black-Scholes call: Division by zero (likely zero volatility or time to expiration).")
        return np.nan
    except ValueError as e:
        logging.error(f"Error in Black-Scholes call: Invalid input value: {e}")
        return np.nan

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the price of a European put option using the Black-Scholes model.

    Args:
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).

    Returns:
        float: The price of the European put option.
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    except ZeroDivisionError:
        logging.error("Error in Black-Scholes put: Division by zero (likely zero volatility or time to expiration).")
        return np.nan
    except ValueError as e:
        logging.error(f"Error in Black-Scholes put: Invalid input value: {e}")
        return np.nan

if __name__ == "__main__":
    # Example usage
    S = 100.0  # Current stock price
    K = 105.0  # Strike price
    T = 1.0    # Time to expiration (1 year)
    r = 0.05   # Risk-free rate
    sigma = 0.2 # Volatility

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)

    print(f"Black-Scholes Call Price: {call_price:.4f}")
    print(f"Black-Scholes Put Price: {put_price:.4f}")