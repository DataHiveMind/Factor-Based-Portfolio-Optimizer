# src/trading/portfolio_manager.py

import pandas as pd
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortfolioManager:
    """
    Manages the current portfolio holdings, cash balance, and generates trade orders
    based on target allocations, considering transaction costs and risk tolerance.
    """
    def __init__(self, initial_capital: float = 100000.0, transaction_cost_model=None, risk_tolerance: float = 0.02):
        """
        Initializes the portfolio manager.

        Args:
            initial_capital (float): The initial cash balance.
            transaction_cost_model (callable, optional): A function that takes an order and returns the estimated transaction cost. Defaults to None (no costs).
            risk_tolerance (float): A measure of the portfolio's risk tolerance, used in position sizing (e.g., max % of capital per trade).
        """
        self.cash = initial_capital
        self.holdings: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
        self.transaction_cost_model = transaction_cost_model if transaction_cost_model else lambda order: 0
        self.risk_tolerance = risk_tolerance
        self.logger.info(f"Portfolio Manager initialized with ${initial_capital:.2f} capital, risk tolerance {risk_tolerance:.2f}.")

    def update_holdings(self, symbol: str, quantity: int):
        """
        Updates the portfolio holdings after a trade execution.

        Args:
            symbol (str): The trading symbol.
            quantity (int): The change in the number of shares.
        """
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
        self.logger.info(f"Updated holdings: {self.holdings}")

    def update_cash(self, amount: float):
        """
        Updates the cash balance after a trade.

        Args:
            amount (float): The change in cash balance (negative for buy, positive for sell, including transaction costs).
        """
        self.cash += amount
        self.logger.info(f"Updated cash balance: ${self.cash:.2f}")

    def _calculate_transaction_cost(self, order: Dict[str, Any]) -> float:
        """
        Calculates the estimated transaction cost for an order.

        Args:
            order (Dict[str, Any]): The trade order details.

        Returns:
            float: The estimated transaction cost.
        """
        return self.transaction_cost_model(order)

    def generate_trade_orders(self, target_weights: pd.Series, current_prices: pd.Series) -> list:
        """
        Generates trade orders to move the current portfolio towards the target weights,
        considering risk tolerance for position sizing.

        Args:
            target_weights (pd.Series): The desired portfolio weights (symbols as index).
            current_prices (pd.Series): The current market prices of the assets (symbols as index).

        Returns:
            list: A list of trade orders (dictionaries) to be executed.
        """
        orders = []
        total_portfolio_value = self.cash + sum(self.holdings.get(symbol, 0) * current_prices.get(symbol, 0) for symbol in self.holdings)
        self.logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")

        for symbol, weight in target_weights.items():
            if symbol not in current_prices:
                self.logger.warning(f"Current price not available for {symbol}, skipping order generation.")
                continue

            target_position_value = total_portfolio_value * weight
            current_position_value = self.holdings.get(symbol, 0) * current_prices[symbol]
            required_value = target_position_value - current_position_value
            price = current_prices[symbol]

            if abs(required_value) > self.risk_tolerance * total_portfolio_value: # Simple risk-based threshold
                trade_quantity = int(required_value / price)
                if trade_quantity != 0:
                    order = {'symbol': symbol, 'quantity': trade_quantity, 'order_type': "MARKET"} # Basic market order
                    orders.append(order)
                    self.logger.info(f"Generated order: {trade_quantity} shares of {symbol} (MARKET).")

        return orders

    def get_portfolio_value(self, current_prices: pd.Series) -> float:
        """
        Calculates the current total value of the portfolio.

        Args:
            current_prices (pd.Series): The current market prices of the assets.

        Returns:
            float: The total portfolio value.
        """
        asset_value = sum(self.holdings.get(symbol, 0) * current_prices.get(symbol, 0) for symbol in self.holdings)
        return self.cash + asset_value

    def get_holdings(self) -> Dict[str, int]:
        """
        Returns the current portfolio holdings.
        """
        return self.holdings

    def get_cash_balance(self) -> float:
        """
        Returns the current cash balance.
        """
        return self.cash

# Example Transaction Cost Model
def simple_transaction_cost_model(order: Dict[str, Any]) -> float:
    """
    A simple example of a transaction cost model (e.g., commission per share).
    """
    if order.get('quantity', 0) != 0:
        return abs(order['quantity']) * 0.01  # $0.01 per share commission
    return 0

if __name__ == "__main__":
    # Example Usage
    transaction_cost_model = simple_transaction_cost_model
    portfolio_manager = PortfolioManager(initial_capital=100000, transaction_cost_model=transaction_cost_model, risk_tolerance=0.01)
    target_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
    current_prices = pd.Series({'AAPL': 170.0, 'MSFT': 300.0})

    orders = portfolio_manager.generate_trade_orders(target_weights, current_prices)
    print("\nGenerated Trade Orders:\n", orders)

    # Simulate execution and update portfolio
    executor = SimulatedExecutionHandler()
    for order in orders:
        execution_result = executor.execute_order(order)
        if execution_result['status'] == 'FILLED':
            symbol = execution_result['symbol']
            fill_quantity = execution_result['fill_quantity']
            execution_price = execution_result['execution_price']
            transaction_cost = portfolio_manager._calculate_transaction_cost(order)
            portfolio_manager.update_holdings(symbol, fill_quantity)
            portfolio_manager.update_cash(-fill_quantity * execution_price - transaction_cost)

    print("\nCurrent Holdings:\n", portfolio_manager.get_holdings())
    print("\nCurrent Cash Balance:\n", portfolio_manager.get_cash_balance())
    print("\nCurrent Portfolio Value:\n", portfolio_manager.get_portfolio_value(current_prices))