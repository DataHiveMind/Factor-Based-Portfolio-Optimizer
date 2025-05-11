# src/trading/order_management.py

import logging
from typing import Dict, List, Optional
import uuid
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderManager:
    """
    Manages the lifecycle of trade orders, including tracking status, fills,
    and supporting more complex order types.
    """
    def __init__(self):
        """
        Initializes the order manager.
        """
        self.orders: Dict[str, dict] = {} # order_id: order_details
        logging.info("Order Manager initialized.")

    def create_order(self, symbol: str, quantity: int, order_type: str, price: float = None, stop_price: float = None, take_profit_price: float = None) -> str:
        """
        Creates a new trade order with support for various order types.

        Args:
            symbol (str): The trading symbol.
            quantity (int): The number of shares.
            order_type (str): The type of order ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'TAKE_PROFIT').
            price (float, optional): The limit price for LIMIT and STOP_LIMIT orders.
            stop_price (float, optional): The stop price for STOP and STOP_LIMIT orders.
            take_profit_price (float, optional): The take-profit price.

        Returns:
            str: The unique ID of the created order.
        """
        order_id = str(uuid.uuid4())
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'stop_price': stop_price,
            'take_profit_price': take_profit_price,
            'status': OrderStatus.PENDING.value,
            'filled_quantity': 0,
            'execution_price': None
        }
        self.orders[order_id] = order
        logging.info(f"Order created with ID {order_id}: {order}")
        return order_id

    def update_order_status(self, order_id: str, status: str, execution_price: float = None, filled_quantity: int = None):
        """
        Updates the status and execution details of an order.

        Args:
            order_id (str): The ID of the order.
            status (str): The new status (from OrderStatus enum).
            execution_price (float, optional): The price at which the order was executed.
            filled_quantity (int, optional): The quantity that has been filled.
        """
        if order_id in self.orders:
            self.orders[order_id]['status'] = status
            if execution_price is not None:
                self.orders[order_id]['execution_price'] = execution_price
            if filled_quantity is not None:
                self.orders[order_id]['filled_quantity'] = filled_quantity
            logging.info(f"Order {order_id} updated: status='{status}'")
        else:
            logging.warning(f"Order with ID {order_id} not found")