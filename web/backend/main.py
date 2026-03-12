"""
FastAPI Backend for Real-Time Risk Analytics Dashboard
High-performance API with WebSocket support for real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# Internal imports (these would be properly structured in a real implementation)
# from risk_engine import PositionEngine, RiskCalculator, PortfolioGreeksCalculator
# from optimization import PortfolioOptimizationEngine
# from stress_testing import StressTestingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Data Models
class PositionData(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime

class RiskMetrics(BaseModel):
    portfolio_value: float
    var_1d: float
    var_10d: float
    expected_shortfall: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    last_calculated: datetime

class GreeksData(BaseModel):
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    last_calculated: datetime

class StressTestResult(BaseModel):
    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_percent: float
    var_breach: bool
    scenario_probability: Optional[float] = None

class AlertData(BaseModel):
    id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

class TradeRequest(BaseModel):
    symbol: str
    quantity: float
    price: float
    side: str  # "BUY" or "SELL"
    order_type: str = "MARKET"

class OptimizationRequest(BaseModel):
    objective: str
    constraints: Dict[str, Any]
    expected_returns: Dict[str, float]
    risk_aversion: float = 1.0

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict, topic: str = None):
        disconnected = []
        for connection in self.active_connections:
            try:
                # Check subscription filter
                if topic and topic not in self.subscriptions.get(connection, []):
                    continue
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# Global instances
manager = ConnectionManager()

# Simulated data stores (in production, these would be proper databases/caches)
class DataStore:
    def __init__(self):
        self.positions = {}
        self.risk_metrics = {}
        self.greeks = {}
        self.alerts = []
        self.market_data = {}
        self.stress_results = {}

    def update_position(self, symbol: str, position_data: PositionData):
        self.positions[symbol] = position_data

    def get_positions(self) -> Dict[str, PositionData]:
        return self.positions

    def update_risk_metrics(self, metrics: RiskMetrics):
        self.risk_metrics = metrics

    def update_greeks(self, greeks: GreeksData):
        self.greeks = greeks

    def add_alert(self, alert: AlertData):
        self.alerts.append(alert)
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

data_store = DataStore()

# Real-time data simulation
class RealTimeDataSimulator:
    def __init__(self):
        self.running = False
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]

    async def start_simulation(self):
        self.running = True
        while self.running:
            await self.generate_market_data()
            await self.update_positions()
            await self.update_risk_metrics()
            await self.check_alerts()
            await asyncio.sleep(1)  # Update every second

    def stop_simulation(self):
        self.running = False

    async def generate_market_data(self):
        """Generate simulated market data"""
        for symbol in self.symbols:
            # Simulate price movement
            current_price = data_store.market_data.get(symbol, 100.0)
            price_change = np.random.normal(0, 0.02) * current_price
            new_price = max(current_price + price_change, 1.0)

            data_store.market_data[symbol] = new_price

            # Broadcast market data update
            await manager.broadcast({
                "type": "market_data",
                "symbol": symbol,
                "price": new_price,
                "timestamp": datetime.now()
            }, "market_data")

    async def update_positions(self):
        """Update position data based on market movements"""
        for symbol in self.symbols:
            if symbol in data_store.market_data:
                current_price = data_store.market_data[symbol]

                # Simulate position if doesn't exist
                if symbol not in data_store.positions:
                    position = PositionData(
                        symbol=symbol,
                        quantity=np.random.randint(-1000, 1000),
                        avg_price=current_price * np.random.uniform(0.95, 1.05),
                        market_price=current_price,
                        unrealized_pnl=0.0,
                        realized_pnl=np.random.uniform(-1000, 1000),
                        last_update=datetime.now()
                    )
                else:
                    position = data_store.positions[symbol]
                    position.market_price = current_price
                    position.last_update = datetime.now()

                # Calculate unrealized P&L
                position.unrealized_pnl = position.quantity * (position.market_price - position.avg_price)

                data_store.update_position(symbol, position)

                # Broadcast position update
                await manager.broadcast({
                    "type": "position_update",
                    "symbol": symbol,
                    "position": position.dict()
                }, "positions")

    async def update_risk_metrics(self):
        """Update portfolio risk metrics"""
        total_value = sum(
            abs(pos.quantity * pos.market_price) for pos in data_store.positions.values()
        )
        total_pnl = sum(
            pos.unrealized_pnl + pos.realized_pnl for pos in data_store.positions.values()
        )

        # Simulate risk metrics
        metrics = RiskMetrics(
            portfolio_value=total_value,
            var_1d=total_value * np.random.uniform(0.01, 0.05),
            var_10d=total_value * np.random.uniform(0.03, 0.15),
            expected_shortfall=total_value * np.random.uniform(0.02, 0.08),
            beta=np.random.uniform(0.8, 1.2),
            sharpe_ratio=np.random.uniform(-0.5, 2.0),
            max_drawdown=np.random.uniform(0.05, 0.25),
            last_calculated=datetime.now()
        )

        data_store.update_risk_metrics(metrics)

        # Simulate Greeks for options positions
        greeks = GreeksData(
            total_delta=np.random.uniform(-10000, 10000),
            total_gamma=np.random.uniform(-1000, 1000),
            total_theta=np.random.uniform(-500, 0),
            total_vega=np.random.uniform(-2000, 2000),
            total_rho=np.random.uniform(-1000, 1000),
            last_calculated=datetime.now()
        )

        data_store.update_greeks(greeks)

        # Broadcast risk updates
        await manager.broadcast({
            "type": "risk_metrics",
            "metrics": metrics.dict(),
            "greeks": greeks.dict()
        }, "risk_metrics")

    async def check_alerts(self):
        """Check for risk limit violations and generate alerts"""
        if hasattr(data_store, 'risk_metrics') and data_store.risk_metrics:
            metrics = data_store.risk_metrics

            # Check VaR limit (example: 5% of portfolio value)
            var_limit = metrics.portfolio_value * 0.05
            if metrics.var_1d > var_limit:
                alert = AlertData(
                    id=f"var_breach_{datetime.now().timestamp()}",
                    type="VAR_BREACH",
                    severity="HIGH",
                    message=f"1-day VaR ({metrics.var_1d:,.0f}) exceeds limit ({var_limit:,.0f})",
                    timestamp=datetime.now()
                )
                data_store.add_alert(alert)

                await manager.broadcast({
                    "type": "alert",
                    "alert": alert.dict()
                }, "alerts")

# Initialize simulator
simulator = RealTimeDataSimulator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting real-time data simulation...")
    asyncio.create_task(simulator.start_simulation())
    yield
    # Shutdown
    logger.info("Stopping real-time data simulation...")
    simulator.stop_simulation()

# FastAPI app
app = FastAPI(
    title="Quantitative Risk Analytics API",
    description="Real-time risk monitoring and portfolio analytics",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency (simplified)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT validation
    if credentials.credentials == "demo_token":
        return {"user_id": "demo_user", "role": "trader"}
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# API Endpoints

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/positions", response_model=Dict[str, PositionData])
async def get_positions(user=Depends(get_current_user)):
    """Get all current positions"""
    return data_store.get_positions()

@app.get("/api/risk-metrics", response_model=RiskMetrics)
async def get_risk_metrics(user=Depends(get_current_user)):
    """Get current portfolio risk metrics"""
    if not data_store.risk_metrics:
        raise HTTPException(status_code=404, detail="Risk metrics not available")
    return data_store.risk_metrics

@app.get("/api/greeks", response_model=GreeksData)
async def get_greeks(user=Depends(get_current_user)):
    """Get current portfolio Greeks"""
    if not data_store.greeks:
        raise HTTPException(status_code=404, detail="Greeks data not available")
    return data_store.greeks

@app.get("/api/alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=1000),
    severity: Optional[str] = Query(None),
    user=Depends(get_current_user)
):
    """Get recent alerts"""
    alerts = data_store.alerts[-limit:]
    if severity:
        alerts = [alert for alert in alerts if alert.severity == severity]
    return {"alerts": [alert.dict() for alert in alerts]}

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user=Depends(get_current_user)):
    """Acknowledge an alert"""
    for alert in data_store.alerts:
        if alert.id == alert_id:
            alert.acknowledged = True
            return {"success": True}
    raise HTTPException(status_code=404, detail="Alert not found")

@app.post("/api/trades")
async def submit_trade(trade: TradeRequest, user=Depends(get_current_user)):
    """Submit a new trade"""
    # In production, this would integrate with order management system
    logger.info(f"Trade submitted: {trade.dict()}")

    # Simulate trade execution
    executed_price = trade.price * np.random.uniform(0.999, 1.001)

    return {
        "trade_id": f"trade_{datetime.now().timestamp()}",
        "status": "EXECUTED",
        "executed_price": executed_price,
        "executed_quantity": trade.quantity,
        "timestamp": datetime.now()
    }

@app.post("/api/optimize")
async def optimize_portfolio(request: OptimizationRequest, user=Depends(get_current_user)):
    """Run portfolio optimization"""
    # Simplified optimization simulation
    symbols = list(request.expected_returns.keys())
    n_assets = len(symbols)

    # Generate random weights (in production, use actual optimization)
    weights = np.random.dirichlet(np.ones(n_assets))
    optimized_weights = {symbols[i]: float(weights[i]) for i in range(n_assets)}

    expected_return = sum(request.expected_returns[symbol] * weight
                         for symbol, weight in optimized_weights.items())

    return {
        "optimization_id": f"opt_{datetime.now().timestamp()}",
        "status": "COMPLETED",
        "optimized_weights": optimized_weights,
        "expected_return": expected_return,
        "expected_risk": np.random.uniform(0.1, 0.3),
        "sharpe_ratio": np.random.uniform(0.5, 2.0),
        "timestamp": datetime.now()
    }

@app.get("/api/stress-test")
async def run_stress_test(
    scenario: str = Query("covid_crash_2020"),
    user=Depends(get_current_user)
):
    """Run stress test scenario"""
    # Simulate stress test results
    portfolio_value = sum(
        abs(pos.quantity * pos.market_price) for pos in data_store.positions.values()
    )

    # Generate scenario-specific results
    stress_multipliers = {
        "covid_crash_2020": -0.34,
        "lehman_crisis_2008": -0.45,
        "black_monday_1987": -0.22,
        "flash_crash_2010": -0.09
    }

    multiplier = stress_multipliers.get(scenario, -0.15)
    stressed_pnl = portfolio_value * multiplier

    result = StressTestResult(
        scenario_name=scenario,
        portfolio_pnl=stressed_pnl,
        portfolio_pnl_percent=multiplier,
        var_breach=abs(stressed_pnl) > portfolio_value * 0.05,
        scenario_probability=0.001
    )

    return result.dict()

@app.get("/api/performance")
async def get_performance_metrics(
    period: str = Query("1d", regex="^(1d|1w|1m|3m|1y)$"),
    user=Depends(get_current_user)
):
    """Get portfolio performance metrics"""
    # Simulate performance data
    days_map = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "1y": 365}
    days = days_map[period]

    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    cumulative_returns = np.cumprod(1 + returns) - 1

    return {
        "period": period,
        "total_return": float(cumulative_returns[-1]),
        "annualized_return": float(np.mean(returns) * 252),
        "volatility": float(np.std(returns) * np.sqrt(252)),
        "sharpe_ratio": float(np.mean(returns) / np.std(returns) * np.sqrt(252)),
        "max_drawdown": float(np.min(cumulative_returns)),
        "var_95": float(np.percentile(returns, 5)),
        "returns_history": returns.tolist(),
        "timestamps": [(datetime.now() - timedelta(days=days-i)).isoformat()
                      for i in range(days)]
    }

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive subscription requests
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                topics = message.get("topics", [])
                manager.subscriptions[websocket] = topics
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "topics": topics
                }, websocket)

            elif message.get("type") == "unsubscribe":
                topics = message.get("topics", [])
                current_subs = manager.subscriptions.get(websocket, [])
                for topic in topics:
                    if topic in current_subs:
                        current_subs.remove(topic)
                await manager.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "topics": topics
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )