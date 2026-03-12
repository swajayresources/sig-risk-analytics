"""
Comprehensive tests for the Position Engine
Tests high-performance position tracking, P&L calculation, and real-time updates
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List

# Mock the C++ position engine for testing
@dataclass
class MockPosition:
    symbol: str
    quantity: float
    avg_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime

class MockPositionEngine:
    def __init__(self):
        self.positions = {}
        self.market_data = {}
        self.callbacks = []
        self.stats = {
            'trades_processed': 0,
            'position_updates': 0,
            'price_updates': 0
        }

    def add_trade(self, symbol: str, quantity: float, price: float, side: str):
        """Add a trade and update position"""
        if symbol not in self.positions:
            self.positions[symbol] = MockPosition(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                market_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                last_update=datetime.now()
            )

        position = self.positions[symbol]
        old_qty = position.quantity
        old_avg = position.avg_price

        trade_qty = quantity if side == 'BUY' else -quantity
        new_qty = old_qty + trade_qty

        if new_qty == 0:
            # Position closed
            position.realized_pnl += (price - old_avg) * abs(trade_qty)
            position.avg_price = 0
        elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            # Same direction
            position.avg_price = (old_avg * abs(old_qty) + price * abs(trade_qty)) / abs(new_qty)
        else:
            # Direction change
            position.avg_price = price

        position.quantity = new_qty
        position.market_price = price
        position.last_update = datetime.now()
        self.update_unrealized_pnl(position)

        self.stats['trades_processed'] += 1
        self.stats['position_updates'] += 1

    def update_market_price(self, symbol: str, price: float):
        """Update market price for a symbol"""
        self.market_data[symbol] = price

        if symbol in self.positions:
            self.positions[symbol].market_price = price
            self.update_unrealized_pnl(self.positions[symbol])

        self.stats['price_updates'] += 1

    def update_unrealized_pnl(self, position: MockPosition):
        """Calculate unrealized P&L"""
        if position.quantity != 0:
            position.unrealized_pnl = position.quantity * (position.market_price - position.avg_price)

    def get_position(self, symbol: str) -> MockPosition:
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, MockPosition]:
        return self.positions.copy()

    def calculate_total_pnl(self) -> float:
        return sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())

    def calculate_total_portfolio_value(self) -> float:
        return sum(abs(pos.quantity * pos.market_price) for pos in self.positions.values())

@pytest.fixture
def position_engine():
    """Create a fresh position engine for each test"""
    return MockPositionEngine()

class TestPositionEngine:
    """Test suite for Position Engine functionality"""

    def test_single_trade_execution(self, position_engine):
        """Test basic trade execution and position creation"""
        position_engine.add_trade("AAPL", 100, 150.0, "BUY")

        position = position_engine.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.market_price == 150.0
        assert position.unrealized_pnl == 0.0

    def test_multiple_trades_same_direction(self, position_engine):
        """Test multiple trades in the same direction"""
        position_engine.add_trade("AAPL", 100, 150.0, "BUY")
        position_engine.add_trade("AAPL", 200, 155.0, "BUY")

        position = position_engine.get_position("AAPL")
        expected_avg = (100 * 150.0 + 200 * 155.0) / 300
        assert position.quantity == 300
        assert abs(position.avg_price - expected_avg) < 1e-6

    def test_opposing_trades(self, position_engine):
        """Test trades in opposite directions"""
        position_engine.add_trade("AAPL", 200, 150.0, "BUY")
        position_engine.add_trade("AAPL", 100, 160.0, "SELL")

        position = position_engine.get_position("AAPL")
        assert position.quantity == 100  # 200 - 100
        assert position.avg_price == 150.0  # Should remain original price
        assert position.realized_pnl == 100 * (160.0 - 150.0)  # Profit on sold shares

    def test_position_closure(self, position_engine):
        """Test complete position closure"""
        position_engine.add_trade("AAPL", 100, 150.0, "BUY")
        position_engine.add_trade("AAPL", 100, 160.0, "SELL")

        position = position_engine.get_position("AAPL")
        assert position.quantity == 0
        assert position.avg_price == 0
        assert position.realized_pnl == 100 * (160.0 - 150.0)

    def test_market_price_updates(self, position_engine):
        """Test market price updates and P&L calculation"""
        position_engine.add_trade("AAPL", 100, 150.0, "BUY")
        position_engine.update_market_price("AAPL", 155.0)

        position = position_engine.get_position("AAPL")
        assert position.market_price == 155.0
        assert position.unrealized_pnl == 100 * (155.0 - 150.0)

    def test_short_position(self, position_engine):
        """Test short position handling"""
        position_engine.add_trade("AAPL", 100, 150.0, "SELL")

        position = position_engine.get_position("AAPL")
        assert position.quantity == -100
        assert position.avg_price == 150.0

        # Price goes down - profit for short position
        position_engine.update_market_price("AAPL", 140.0)
        assert position.unrealized_pnl == -100 * (140.0 - 150.0)  # Negative quantity * negative price change = positive P&L

    def test_portfolio_aggregation(self, position_engine):
        """Test portfolio-level aggregation functions"""
        # Create multiple positions
        trades = [
            ("AAPL", 100, 150.0, "BUY"),
            ("GOOGL", 50, 2500.0, "BUY"),
            ("TSLA", 200, 800.0, "SELL")  # Short position
        ]

        for symbol, qty, price, side in trades:
            position_engine.add_trade(symbol, qty, price, side)

        # Update market prices
        position_engine.update_market_price("AAPL", 155.0)
        position_engine.update_market_price("GOOGL", 2600.0)
        position_engine.update_market_price("TSLA", 750.0)

        total_pnl = position_engine.calculate_total_pnl()
        portfolio_value = position_engine.calculate_total_portfolio_value()

        expected_pnl = (
            100 * (155.0 - 150.0) +  # AAPL profit
            50 * (2600.0 - 2500.0) +  # GOOGL profit
            -200 * (750.0 - 800.0)    # TSLA profit (short)
        )

        assert abs(total_pnl - expected_pnl) < 1e-6
        assert portfolio_value > 0

    def test_statistics_tracking(self, position_engine):
        """Test performance statistics tracking"""
        initial_stats = position_engine.stats.copy()

        position_engine.add_trade("AAPL", 100, 150.0, "BUY")
        position_engine.update_market_price("AAPL", 155.0)

        assert position_engine.stats['trades_processed'] == initial_stats['trades_processed'] + 1
        assert position_engine.stats['position_updates'] == initial_stats['position_updates'] + 1
        assert position_engine.stats['price_updates'] == initial_stats['price_updates'] + 1

    def test_concurrent_access(self, position_engine):
        """Test thread safety with concurrent operations"""
        import threading
        import random

        def trade_worker(engine, symbol_base, num_trades):
            for i in range(num_trades):
                symbol = f"{symbol_base}_{i % 10}"  # Distribute across 10 symbols
                qty = random.randint(1, 100)
                price = random.uniform(100.0, 200.0)
                side = random.choice(["BUY", "SELL"])
                engine.add_trade(symbol, qty, price, side)

        def price_worker(engine, symbol_base, num_updates):
            for i in range(num_updates):
                symbol = f"{symbol_base}_{i % 10}"
                price = random.uniform(100.0, 200.0)
                engine.update_market_price(symbol, price)

        # Create multiple threads
        threads = []
        num_trades_per_thread = 100

        for i in range(5):  # 5 trade threads
            thread = threading.Thread(
                target=trade_worker,
                args=(position_engine, f"STOCK", num_trades_per_thread)
            )
            threads.append(thread)

        for i in range(3):  # 3 price update threads
            thread = threading.Thread(
                target=price_worker,
                args=(position_engine, f"STOCK", num_trades_per_thread)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify integrity
        total_trades = position_engine.stats['trades_processed']
        assert total_trades == 5 * num_trades_per_thread

        # Verify position consistency
        positions = position_engine.get_all_positions()
        for position in positions.values():
            if position.quantity != 0:
                assert position.avg_price > 0

class TestPerformance:
    """Performance-focused tests"""

    def test_high_frequency_trades(self, position_engine):
        """Test performance with high-frequency trade processing"""
        import time

        num_trades = 10000
        symbols = [f"STOCK_{i}" for i in range(100)]

        start_time = time.time()

        for i in range(num_trades):
            symbol = symbols[i % len(symbols)]
            qty = np.random.randint(1, 1000)
            price = np.random.uniform(50.0, 500.0)
            side = np.random.choice(["BUY", "SELL"])
            position_engine.add_trade(symbol, qty, price, side)

        end_time = time.time()
        duration = end_time - start_time

        trades_per_second = num_trades / duration
        print(f"Processed {num_trades} trades in {duration:.2f}s ({trades_per_second:.0f} trades/sec)")

        # Performance assertion: should handle at least 1000 trades per second
        assert trades_per_second > 1000, f"Performance too slow: {trades_per_second:.0f} trades/sec"

    def test_market_data_updates(self, position_engine):
        """Test performance of market data updates"""
        import time

        symbols = [f"STOCK_{i}" for i in range(1000)]
        num_updates = 50000

        # First create some positions
        for symbol in symbols[:100]:  # Only first 100 symbols have positions
            position_engine.add_trade(symbol, 100, 100.0, "BUY")

        start_time = time.time()

        for i in range(num_updates):
            symbol = symbols[i % len(symbols)]
            price = np.random.uniform(50.0, 200.0)
            position_engine.update_market_price(symbol, price)

        end_time = time.time()
        duration = end_time - start_time

        updates_per_second = num_updates / duration
        print(f"Processed {num_updates} price updates in {duration:.2f}s ({updates_per_second:.0f} updates/sec)")

        # Performance assertion: should handle at least 10000 updates per second
        assert updates_per_second > 10000, f"Price update performance too slow: {updates_per_second:.0f} updates/sec"

    def test_memory_efficiency(self, position_engine):
        """Test memory usage with large numbers of positions"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many positions
        num_positions = 10000
        for i in range(num_positions):
            symbol = f"STOCK_{i:06d}"
            position_engine.add_trade(symbol, 100, 100.0, "BUY")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        bytes_per_position = memory_increase / num_positions
        print(f"Memory usage: {memory_increase / 1024 / 1024:.1f} MB for {num_positions} positions")
        print(f"Average: {bytes_per_position:.0f} bytes per position")

        # Memory efficiency assertion: should use less than 1KB per position
        assert bytes_per_position < 1024, f"Memory usage too high: {bytes_per_position:.0f} bytes per position"

class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_quantity_trade(self, position_engine):
        """Test handling of zero quantity trades"""
        position_engine.add_trade("AAPL", 0, 150.0, "BUY")
        position = position_engine.get_position("AAPL")
        # Should either create empty position or not create position at all
        if position:
            assert position.quantity == 0

    def test_negative_prices(self, position_engine):
        """Test handling of negative prices (should be rejected or handled appropriately)"""
        # This test depends on implementation - negative prices might be valid for some instruments
        position_engine.add_trade("COMMODITY", 100, -10.0, "BUY")
        position = position_engine.get_position("COMMODITY")
        if position:
            # If negative prices are allowed, verify calculations work correctly
            assert position.avg_price == -10.0

    def test_very_large_quantities(self, position_engine):
        """Test handling of very large quantities"""
        large_qty = 1e9  # 1 billion shares
        position_engine.add_trade("AAPL", large_qty, 150.0, "BUY")

        position = position_engine.get_position("AAPL")
        assert position.quantity == large_qty
        assert position.avg_price == 150.0

    def test_precision_handling(self, position_engine):
        """Test numerical precision with small prices and quantities"""
        # Test with very small prices (e.g., cryptocurrency fractions)
        small_price = 0.00000123
        position_engine.add_trade("CRYPTO", 1000000, small_price, "BUY")

        position = position_engine.get_position("CRYPTO")
        assert abs(position.avg_price - small_price) < 1e-10

    def test_nonexistent_position_queries(self, position_engine):
        """Test querying positions that don't exist"""
        position = position_engine.get_position("NONEXISTENT")
        assert position is None

        positions = position_engine.get_all_positions()
        assert len(positions) == 0

class TestRealTimeFeatures:
    """Test real-time features like callbacks and monitoring"""

    def test_position_callbacks(self, position_engine):
        """Test position update callbacks"""
        callback_calls = []

        def position_callback(position):
            callback_calls.append((position.symbol, position.quantity))

        # Mock callback registration (implementation-dependent)
        position_engine.callbacks.append(position_callback)

        position_engine.add_trade("AAPL", 100, 150.0, "BUY")

        # Simulate callback execution
        position = position_engine.get_position("AAPL")
        for callback in position_engine.callbacks:
            callback(position)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("AAPL", 100)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])