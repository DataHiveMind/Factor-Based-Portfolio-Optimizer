# src/derivatives/derivative_risk.py

import numpy as np
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the Delta of a European call option using the Black-Scholes model.
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
    except ZeroDivisionError:
        logging.error("Error in call delta: Division by zero.")
        return np.nan
    except ValueError as e:
        logging.error(f"Error in call delta: Invalid input value: {e}")
        return np.nan

def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the Gamma of a European option (same for call and put) using the Black-Scholes model.
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except ZeroDivisionError:
        logging.error("Error in gamma: Division by zero.")
        return np.nan
    except ValueError as e:
        logging.error(f"Error in gamma: Invalid input value: {e}")
        return np.nan

# Add functions for Vega, Theta, Rho as needed.

if __name__ == "__main__":
    # Example usage
    S = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    call_delta = delta_call(S, K, T, r, sigma)
    option_gamma = gamma(S, K, T, r, sigma)

    print(f"Call Option Delta: {call_delta:.4f}")
    print(f"Option Gamma: {option_gamma:.4f}")