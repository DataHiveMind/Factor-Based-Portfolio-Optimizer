# src/stochastic_models/asset_simulator.py

import numpy as np
import pandas as pd
import logging
from typing import Union, Dict, Tuple
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StochasticModelSimulator:
    """
    A base class for stochastic model simulators, providing common functionalities.
    """
    def __init__(self, params: Dict[str, float]):
        """
        Initializes the simulator with model parameters.

        Args:
            params (Dict[str, float]): A dictionary of model parameters.
        """
        self.params = params

    def _generate_wiener_process(self, num_steps: int, dt: float) -> np.ndarray:
        """
        Generates a Wiener process (Brownian motion).
        """
        return np.random.normal(0, np.sqrt(dt), num_steps)

    def calibrate_parameters(self, historical_data: pd.Series, method: str = 'MLE', **kwargs) -> Dict[str, float]:
        """
        Calibrates the model parameters to historical data. This is a placeholder
        and needs to be implemented in subclasses.

        Args:
            historical_data (pd.Series): Historical asset prices or returns.
            method (str): The calibration method (e.g., 'MLE', 'GMM').
            **kwargs: Additional arguments for the calibration method.

        Returns:
            Dict[str, float]: A dictionary of calibrated model parameters.
        """
        logging.warning(f"Calibration method '{method}' not implemented for {self.__class__.__name__}.")
        return self.params

    def simulate_paths(self, num_paths: int, num_steps: int, time_horizon: float, start_value: float) -> pd.DataFrame:
        """
        Simulates multiple paths of the asset price or return process. This needs
        to be implemented in subclasses.

        Args:
            num_paths (int): The number of simulation paths to generate.
            num_steps (int): The number of time steps in each simulation.
            time_horizon (float): The total time horizon for the simulation (in years).
            start_value (float): The initial value of the asset (price or return level).

        Returns:
            pd.DataFrame: A DataFrame where each column represents a simulation path
                          and the index represents time.
        """
        raise NotImplementedError("Simulate paths method must be implemented in the subclass.")

class GeometricBrownianMotionSimulator(StochasticModelSimulator):
    """
    Simulates asset prices using the Geometric Brownian Motion (GBM) model.
    """
    def __init__(self, mu: float, sigma: float, start_price: float):
        """
        Initializes the GBM simulator.

        Args:
            mu (float): The expected return (drift).
            sigma (float): The volatility.
            start_price (float): The initial price.
        """
        super().__init__(params={'mu': mu, 'sigma': sigma})
        self.start_price = start_price
        logging.info(f"GBM Simulator initialized with mu={mu}, sigma={sigma}, start_price={start_price}.")

    def simulate_paths(self, num_paths: int, num_steps: int, time_horizon: float, start_price: float = None) -> pd.DataFrame:
        """
        Simulates multiple paths of asset prices using GBM.

        Args:
            num_paths (int): Number of simulation paths.
            num_steps (int): Number of time steps.
            time_horizon (float): Total time horizon (in years).
            start_price (float, optional): Initial price. Defaults to the simulator's initial price.

        Returns:
            pd.DataFrame: DataFrame of simulated price paths.
        """
        start_price = start_price if start_price is not None else self.start_price
        dt = time_horizon / num_steps
        sqrt_dt = np.sqrt(dt)
        prices = np.zeros((num_steps + 1, num_paths))
        prices[0, :] = start_price
        mu = self.params['mu']
        sigma = self.params['sigma']

        for i in range(1, num_steps + 1):
            dW = np.random.normal(0, 1, num_paths) * sqrt_dt
            prices[i, :] = prices[i - 1, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

        time_index = pd.to_datetime(pd.date_range(start='now', periods=num_steps + 1, freq=f'{int(dt * 365 * 24 * 60 * 60)}S'))
        return pd.DataFrame(prices, index=time_index)

    def calibrate_parameters(self, historical_prices: pd.Series, method: str = 'MLE', **kwargs) -> Dict[str, float]:
        """
        Calibrates the mu and sigma parameters of GBM using Maximum Likelihood Estimation.

        Args:
            historical_prices (pd.Series): Historical asset prices.
            method (str): Calibration method (only MLE is implemented for now).
            **kwargs: Additional arguments.

        Returns:
            Dict[str, float]: Calibrated parameters {'mu': ..., 'sigma': ...}.
        """
        if method == 'MLE':
            returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
            mu_hat = np.mean(returns) * (252 if kwargs.get('annualize', True) else 1) # Assuming daily data
            sigma_hat = np.std(returns) * np.sqrt(252) if kwargs.get('annualize', True) else np.std(returns)
            self.params['mu'] = mu_hat
            self.params['sigma'] = sigma_hat
            logging.info(f"GBM parameters calibrated using MLE: mu={mu_hat:.4f}, sigma={sigma_hat:.4f}.")
            return self.params
        else:
            return super().calibrate_parameters(historical_prices, method, **kwargs)

class HestonModelSimulator(StochasticModelSimulator):
    """
    Simulates asset prices using the Heston model (stochastic volatility).

    dV_t = kappa * (theta - V_t) * dt + xi * sqrt(V_t) * dW_v_t
    dS_t = mu * S_t * dt + sqrt(V_t) * S_t * dW_s_t
    corr(dW_s_t, dW_v_t) = rho
    """
    def __init__(self, mu: float, kappa: float, theta: float, xi: float, rho: float, start_price: float, start_variance: float):
        """
        Initializes the Heston model simulator.

        Args:
            mu (float): Drift of the asset price.
            kappa (float): Mean reversion speed of the variance.
            theta (float): Long-term mean of the variance.
            xi (float): Volatility of the variance.
            rho (float): Correlation between asset returns and variance changes.
            start_price (float): Initial price of the asset.
            start_variance (float): Initial variance.
        """
        super().__init__(params={'mu': mu, 'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho})
        self.start_price = start_price
        self.start_variance = start_variance
        logging.info(f"Heston Simulator initialized with mu={mu}, kappa={kappa}, theta={theta}, xi={xi}, rho={rho}, start_price={start_price}, start_variance={start_variance}.")

    def simulate_paths(self, num_paths: int, num_steps: int, time_horizon: float,
                       start_price: float = None, start_variance: float = None) -> pd.DataFrame:
        """
        Simulates multiple paths of asset prices using the Heston model.

        Args:
            num_paths (int): Number of simulation paths.
            num_steps (int): Number of time steps.
            time_horizon (float): Total time horizon (in years).
            start_price (float, optional): Initial price. Defaults to the simulator's initial price.
            start_variance (float, optional): Initial variance. Defaults to the simulator's initial variance.

        Returns:
            pd.DataFrame: DataFrame of simulated price paths.
        """
        start_price = start_price if start_price is not None else self.start_price
        start_variance = start_variance if start_variance is not None else self.start_variance
        dt = time_horizon / num_steps
        prices = np.zeros((num_steps + 1, num_paths))
        variances = np.zeros((num_steps + 1, num_paths))
        prices[0, :] = start_price
        variances[0, :] = start_variance
        mu = self.params['mu']
        kappa = self.params['kappa']
        theta = self.params['theta']
        xi = self.params['xi']
        rho = self.params['rho']

        for i in range(1, num_steps + 1):
            dW_s = np.random.normal(0, 1, num_paths) * np.sqrt(dt)
            dW_v = (rho * dW_s + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, num_paths) * np.sqrt(dt))
            variances[i, :] = np.maximum(0, variances[i - 1, :] + kappa * (theta - variances[i - 1, :]) * dt + xi * np.sqrt(variances[i - 1, :]) * dW_v)
            prices[i, :] = prices[i - 1, :] * np.exp((mu - 0.5 * variances[i - 1, :]) * dt + np.sqrt(variances[i - 1, :]) * dW_s)

        time_index = pd.to_datetime(pd.date_range(start='now', periods=num_steps + 1, freq=f'{int(dt * 365 * 24 * 60 * 60)}S'))
        return pd.DataFrame(prices, index=time_index)

    # Implement calibration for Heston model (more complex, might involve optimization)
    def calibrate_parameters(self, historical_prices: pd.Series, method: str = 'Optimization', **kwargs) -> Dict[str, float]:
        logging.warning(f"Calibration for Heston model using '{method}' is not fully implemented. Returning initial parameters.")
        return self.params

# Add other stochastic models here (e.g., Jump Diffusion) following the same pattern

if __name__ == "__main__":
    # Example Usage (GBM)
    gbm_mu = 0.10
    gbm_sigma = 0.20
    gbm_start_price = 100.0
    gbm_num_steps = 252
    gbm_time_horizon = 1.0

    gbm_simulator = GeometricBrownianMotionSimulator(gbm_mu, gbm_sigma, gbm_start_price)
    gbm_simulated_prices = gbm_simulator.simulate_paths(num_paths=5, num_steps=gbm_num_steps, time_horizon=gbm_time_horizon)
    print("\nSimulated Prices (GBM):\n", gbm_simulated_prices.head())

    # Example Calibration (GBM) - requires historical data
    historical_data = pd.Series(np.random.rand(100) * 100 + 50, index=pd.to_datetime(pd.date_range('2024-01-01', periods=100, freq='B')))
    calibrated_params_gbm = gbm_simulator.calibrate_parameters(historical_data)
    print("\nCalibrated GBM Parameters:\n", calibrated_params_gbm)

    # Example Usage (Heston)
    heston_mu = 0.05
    heston_kappa = 2.0
    heston_theta = 0.04
    heston_xi = 0.1
    heston_rho = -0.7
    heston_start_price = 100.0
    heston_start_variance = 0.04
    heston_num_steps = 252
    heston_time_horizon = 1.0

    heston_simulator = HestonModelSimulator(heston_mu, heston_kappa, heston_theta, heston_xi, heston_rho, heston_start_price, heston_start_variance)
    heston_simulated_prices = heston_simulator.simulate_paths(num_paths=5, num_steps=heston_num_steps, time_horizon=heston_time_horizon)
    print("\nSimulated Prices (Heston):\n", heston_simulated_prices.head())