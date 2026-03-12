"""
FastAPI Backend for Quantitative Risk Analytics Engine
Provides real-time risk metrics API with WebSocket support
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Redis for caching and pub/sub
import redis.asyncio as redis
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our risk engine modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from risk_engine.advanced_risk_metrics import (
    HistoricalSimulationVaR, ParametricVaR, MonteCarloVaR,
    GARCHVolatilityForecaster, BlackScholesGreeks, PortfolioRiskMetrics
)
from risk_engine.monte_carlo_engine import (
    MonteCarloEngine, PathGenerationConfig, VarianceReductionConfig
)
from optimization.advanced_portfolio_optimization import (
    MeanVarianceOptimizer, RiskParityOptimizer, BlackLittermanOptimizer,
    HierarchicalRiskParityOptimizer, OptimizationConfig, OptimizationConstraints
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for shared state
risk_engines = {}
websocket_connections = set()
background_tasks_running = False
redis_client = None
executor = ThreadPoolExecutor(max_workers=8)

# Pydantic models for API
class Position(BaseModel):
    symbol: str
    quantity: float
    price: float
    market_value: float
    weight: Optional[float] = None

class Portfolio(BaseModel):
    positions: List[Position]
    total_value: float
    last_updated: datetime

class RiskMetricsRequest(BaseModel):
    portfolio: Portfolio
    confidence_level: float = Field(default=0.95, ge=0.01, le=0.99)
    lookback_days: int = Field(default=252, ge=30, le=2520)
    monte_carlo_scenarios: int = Field(default=10000, ge=1000, le=100000)

class OptimizationRequest(BaseModel):
    expected_returns: List[float]
    covariance_matrix: List[List[float]]
    method: str = Field(default="mean_variance", regex="^(mean_variance|risk_parity|black_litterman|hrp)$")
    constraints: Optional[Dict] = None
    config: Optional[Dict] = None

class StressTestRequest(BaseModel):
    portfolio: Portfolio
    scenario: str = Field(default="market_crash", regex="^(market_crash|interest_rate_shock|sector_rotation|custom)$")
    custom_shocks: Optional[Dict[str, float]] = None

class GreeksRequest(BaseModel):
    underlying_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    option_type: str = Field(regex="^(call|put)$")

# Initialize Redis connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, background_tasks_running

    # Startup
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    background_tasks_running = True

    # Initialize risk engines
    _initialize_risk_engines()

    # Start background tasks
    asyncio.create_task(market_data_simulator())
    asyncio.create_task(risk_calculation_scheduler())

    logger.info("Risk Analytics API started")

    yield

    # Shutdown
    background_tasks_running = False
    await redis_client.close()
    executor.shutdown(wait=True)
    logger.info("Risk Analytics API stopped")

# Create FastAPI app
app = FastAPI(
    title="Quantitative Risk Analytics API",
    description="Real-time risk metrics and portfolio optimization API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _initialize_risk_engines():
    """Initialize risk calculation engines"""
    global risk_engines

    risk_engines['hist_var'] = HistoricalSimulationVaR(lookback_days=252)
    risk_engines['param_var'] = ParametricVaR()
    risk_engines['mc_var'] = MonteCarloVaR(n_simulations=10000)
    risk_engines['garch'] = GARCHVolatilityForecaster()
    risk_engines['greeks'] = BlackScholesGreeks()
    risk_engines['portfolio_metrics'] = PortfolioRiskMetrics()

    # Monte Carlo engine
    path_config = PathGenerationConfig(
        n_paths=10000,
        n_steps=21,
        use_antithetic=True,
        use_moment_matching=True
    )
    variance_config = VarianceReductionConfig(
        use_control_variates=True,
        use_importance_sampling=False
    )
    risk_engines['monte_carlo'] = MonteCarloEngine(path_config, variance_config)

    # Optimization engines
    opt_config = OptimizationConfig(risk_aversion=3.0, max_iterations=1000, tolerance=1e-8)
    risk_engines['mv_optimizer'] = MeanVarianceOptimizer(opt_config)
    risk_engines['rp_optimizer'] = RiskParityOptimizer(opt_config)
    risk_engines['bl_optimizer'] = BlackLittermanOptimizer(opt_config)
    risk_engines['hrp_optimizer'] = HierarchicalRiskParityOptimizer(opt_config)

    logger.info("Risk engines initialized")

def _portfolio_to_arrays(portfolio: Portfolio) -> tuple:
    """Convert portfolio to numpy arrays for calculations"""
    symbols = [pos.symbol for pos in portfolio.positions]
    quantities = np.array([pos.quantity for pos in portfolio.positions])
    prices = np.array([pos.price for pos in portfolio.positions])
    market_values = np.array([pos.market_value for pos in portfolio.positions])
    weights = market_values / portfolio.total_value

    return symbols, quantities, prices, weights

async def _get_historical_returns(symbols: List[str], days: int = 252) -> np.ndarray:
    """Get historical returns for symbols (simulated for demo)"""
    # In production, this would fetch real market data
    np.random.seed(int(time.time()) % 1000)

    n_assets = len(symbols)
    # Generate correlated returns
    correlation_strength = 0.3
    base_vol = 0.02

    correlation_matrix = np.eye(n_assets) * (1 - correlation_strength) + np.full((n_assets, n_assets), correlation_strength)

    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix * base_vol**2,
        size=days
    )

    return returns

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Quantitative Risk Analytics API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "redis": redis_status,
        "engines_loaded": len(risk_engines),
        "active_connections": len(websocket_connections)
    }

@app.post("/api/risk-metrics")
async def calculate_risk_metrics(request: RiskMetricsRequest):
    """Calculate comprehensive risk metrics for portfolio"""
    try:
        symbols, quantities, prices, weights = _portfolio_to_arrays(request.portfolio)

        # Get historical returns
        returns = await _get_historical_returns(symbols, request.lookback_days)
        portfolio_returns = returns @ weights
        covariance_matrix = np.cov(returns.T)

        # Calculate risk metrics in parallel
        tasks = []

        # Historical VaR
        hist_var_engine = risk_engines['hist_var']
        hist_var_engine.lookback_days = min(request.lookback_days, len(returns))
        tasks.append(executor.submit(
            hist_var_engine.calculate_var,
            portfolio_returns,
            request.confidence_level
        ))

        # Parametric VaR
        param_var_engine = risk_engines['param_var']
        tasks.append(executor.submit(
            param_var_engine.calculate_var,
            weights,
            covariance_matrix,
            request.confidence_level
        ))

        # Monte Carlo VaR
        mc_var_engine = risk_engines['mc_var']
        mc_var_engine.n_simulations = request.monte_carlo_scenarios
        expected_returns = np.mean(returns, axis=0)
        tasks.append(executor.submit(
            mc_var_engine.calculate_var,
            weights,
            expected_returns,
            covariance_matrix,
            request.confidence_level
        ))

        # GARCH volatility forecast
        garch_engine = risk_engines['garch']
        tasks.append(executor.submit(
            garch_engine.forecast_volatility,
            portfolio_returns,
            horizon=5
        ))

        # Portfolio risk metrics
        portfolio_metrics_engine = risk_engines['portfolio_metrics']
        tasks.append(executor.submit(
            portfolio_metrics_engine.calculate_component_var,
            weights,
            covariance_matrix,
            request.confidence_level
        ))

        tasks.append(executor.submit(
            portfolio_metrics_engine.calculate_marginal_var,
            weights,
            covariance_matrix,
            request.confidence_level
        ))

        # Wait for all tasks to complete
        results = [task.result() for task in tasks]

        hist_var_result = results[0]
        param_var_result = results[1]
        mc_var_result = results[2]
        garch_result = results[3]
        component_var = results[4]
        marginal_var = results[5]

        # Calculate additional metrics
        portfolio_return = float(np.mean(portfolio_returns) * 252)  # Annualized
        portfolio_volatility = float(np.std(portfolio_returns) * np.sqrt(252))  # Annualized
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Max drawdown calculation
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        response = {
            "portfolio_metrics": {
                "total_value": request.portfolio.total_value,
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            },
            "var_metrics": {
                "historical_var": {
                    "var": float(hist_var_result['var']),
                    "expected_shortfall": float(hist_var_result['expected_shortfall']),
                    "confidence_level": request.confidence_level
                },
                "parametric_var": {
                    "var": float(param_var_result['var']),
                    "expected_shortfall": float(param_var_result.get('expected_shortfall', param_var_result['var'] * 1.3)),
                    "confidence_level": request.confidence_level
                },
                "monte_carlo_var": {
                    "var": float(mc_var_result['var']),
                    "expected_shortfall": float(mc_var_result['expected_shortfall']),
                    "confidence_level": request.confidence_level,
                    "scenarios": request.monte_carlo_scenarios
                }
            },
            "risk_decomposition": {
                "component_var": [float(x) for x in component_var],
                "marginal_var": [float(x) for x in marginal_var],
                "asset_symbols": symbols
            },
            "volatility_forecast": {
                "forecasted_volatility": [float(x) for x in garch_result['forecasted_volatility']],
                "forecast_horizon": len(garch_result['forecasted_volatility'])
            },
            "calculation_timestamp": datetime.utcnow().isoformat(),
            "calculation_parameters": {
                "confidence_level": request.confidence_level,
                "lookback_days": request.lookback_days,
                "monte_carlo_scenarios": request.monte_carlo_scenarios
            }
        }

        # Cache results in Redis
        await redis_client.setex(
            f"risk_metrics:{hash(str(request.portfolio.dict()))}",
            300,  # 5 minutes TTL
            json.dumps(response, default=str)
        )

        # Broadcast to WebSocket clients
        await _broadcast_to_websockets({
            "type": "risk_metrics_update",
            "data": response
        })

        return response

    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")

@app.post("/api/greeks")
async def calculate_greeks(request: GreeksRequest):
    """Calculate options Greeks"""
    try:
        greeks_engine = risk_engines['greeks']

        greeks = await asyncio.get_event_loop().run_in_executor(
            executor,
            greeks_engine.calculate_greeks,
            request.underlying_price,
            request.strike_price,
            request.time_to_expiry,
            request.risk_free_rate,
            request.volatility,
            request.dividend_yield,
            request.option_type
        )

        response = {
            "greeks": {
                "delta": float(greeks['delta']),
                "gamma": float(greeks['gamma']),
                "theta": float(greeks['theta']),
                "vega": float(greeks['vega']),
                "rho": float(greeks['rho'])
            },
            "option_price": float(greeks.get('option_price', 0)),
            "inputs": request.dict(),
            "calculation_timestamp": datetime.utcnow().isoformat()
        }

        return response

    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        raise HTTPException(status_code=500, detail=f"Greeks calculation failed: {str(e)}")

@app.post("/api/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    """Optimize portfolio allocation"""
    try:
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)

        # Select optimizer
        optimizer_map = {
            "mean_variance": risk_engines['mv_optimizer'],
            "risk_parity": risk_engines['rp_optimizer'],
            "black_litterman": risk_engines['bl_optimizer'],
            "hrp": risk_engines['hrp_optimizer']
        }

        optimizer = optimizer_map[request.method]

        # Set up constraints
        constraints = OptimizationConstraints()
        if request.constraints:
            if 'min_weights' in request.constraints:
                constraints.min_weights = np.array(request.constraints['min_weights'])
            if 'max_weights' in request.constraints:
                constraints.max_weights = np.array(request.constraints['max_weights'])
            if 'no_short_selling' in request.constraints:
                constraints.no_short_selling = request.constraints['no_short_selling']

        # Run optimization
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            optimizer.optimize,
            expected_returns,
            covariance_matrix,
            constraints
        )

        response = {
            "optimization_result": {
                "weights": [float(x) for x in result['weights']],
                "expected_return": float(result['expected_return']),
                "expected_risk": float(result['expected_risk']),
                "sharpe_ratio": float(result['sharpe_ratio']),
                "optimization_status": result['optimization_status']
            },
            "method": request.method,
            "calculation_timestamp": datetime.utcnow().isoformat()
        }

        if 'risk_contributions' in result:
            response['optimization_result']['risk_contributions'] = [float(x) for x in result['risk_contributions']]

        return response

    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/api/stress-test")
async def run_stress_test(request: StressTestRequest):
    """Run stress testing scenarios"""
    try:
        symbols, quantities, prices, weights = _portfolio_to_arrays(request.portfolio)

        # Get historical returns
        returns = await _get_historical_returns(symbols, 252)

        # Define stress scenarios
        stress_scenarios = {
            "market_crash": {"equity_shock": -0.30, "volatility_spike": 2.0},
            "interest_rate_shock": {"rate_increase": 0.02, "duration_effect": -0.10},
            "sector_rotation": {"tech_underperform": -0.15, "defensive_outperform": 0.10}
        }

        if request.scenario == "custom" and request.custom_shocks:
            scenario_params = request.custom_shocks
        else:
            scenario_params = stress_scenarios.get(request.scenario, stress_scenarios["market_crash"])

        # Apply stress scenario
        stressed_returns = returns.copy()

        if "equity_shock" in scenario_params:
            shock_magnitude = scenario_params["equity_shock"]
            stressed_returns[-1, :] += shock_magnitude

        if "volatility_spike" in scenario_params:
            vol_multiplier = scenario_params["volatility_spike"]
            stressed_returns *= vol_multiplier

        # Calculate stressed portfolio returns
        portfolio_returns = stressed_returns @ weights

        # Calculate stress VaR
        hist_var_engine = risk_engines['hist_var']
        hist_var_engine.lookback_days = min(60, len(portfolio_returns))

        stress_var = await asyncio.get_event_loop().run_in_executor(
            executor,
            hist_var_engine.calculate_var,
            portfolio_returns,
            0.95
        )

        # Calculate scenario return
        scenario_return = float(np.sum(stressed_returns[-1, :] * weights))
        scenario_impact = scenario_return * request.portfolio.total_value

        response = {
            "stress_test_result": {
                "scenario": request.scenario,
                "scenario_return": scenario_return,
                "scenario_impact_dollar": scenario_impact,
                "stress_var": float(stress_var['var']),
                "stress_expected_shortfall": float(stress_var['expected_shortfall']),
                "stress_var_dollar": float(stress_var['var'] * request.portfolio.total_value),
                "scenario_parameters": scenario_params
            },
            "calculation_timestamp": datetime.utcnow().isoformat()
        }

        return response

    except Exception as e:
        logger.error(f"Error in stress testing: {e}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, days: int = 30):
    """Get market data for a symbol"""
    try:
        # Simulate market data (in production, fetch from real data source)
        np.random.seed(hash(symbol) % 1000)

        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='D'
        )

        # Generate realistic price series
        initial_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = initial_price * np.cumprod(1 + returns)

        volumes = np.random.randint(100000, 1000000, days)

        market_data = []
        for i, date in enumerate(dates):
            market_data.append({
                "symbol": symbol,
                "date": date.isoformat(),
                "price": float(prices[i]),
                "volume": int(volumes[i]),
                "return": float(returns[i])
            })

        return {
            "symbol": symbol,
            "data": market_data,
            "current_price": float(prices[-1]),
            "daily_return": float(returns[-1]),
            "volatility": float(np.std(returns) * np.sqrt(252))  # Annualized
        }

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=f"Market data fetch failed: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.add(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(websocket_connections)}")

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
            elif message.get("type") == "subscribe":
                # Handle subscription to specific topics
                topics = message.get("topics", [])
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "topics": topics,
                    "timestamp": datetime.utcnow().isoformat()
                }))

    except WebSocketDisconnect:
        websocket_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(websocket_connections)}")

async def _broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if websocket_connections:
        message_str = json.dumps(message, default=str)
        disconnected = set()

        for websocket in websocket_connections:
            try:
                await websocket.send_text(message_str)
            except:
                disconnected.add(websocket)

        # Remove disconnected clients
        websocket_connections -= disconnected

# Background tasks
async def market_data_simulator():
    """Simulate real-time market data updates"""
    while background_tasks_running:
        try:
            # Generate random market update
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            symbol = np.random.choice(symbols)

            # Simulate price change
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price = 100 * (1 + price_change)  # Base price of 100

            market_update = {
                "type": "market_data_update",
                "data": {
                    "symbol": symbol,
                    "price": current_price,
                    "change": price_change,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            # Broadcast to WebSocket clients
            await _broadcast_to_websockets(market_update)

            # Cache in Redis
            await redis_client.setex(
                f"market_data:{symbol}",
                60,  # 1 minute TTL
                json.dumps(market_update["data"], default=str)
            )

            await asyncio.sleep(5)  # Update every 5 seconds

        except Exception as e:
            logger.error(f"Error in market data simulator: {e}")
            await asyncio.sleep(10)

async def risk_calculation_scheduler():
    """Scheduled risk calculations for active portfolios"""
    while background_tasks_running:
        try:
            # In production, this would fetch active portfolios from database
            # For demo, we'll just broadcast periodic risk updates

            risk_update = {
                "type": "risk_metrics_update",
                "data": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_status": "active",
                    "calculations_per_minute": np.random.randint(50, 200),
                    "average_latency_ms": np.random.uniform(1, 10)
                }
            }

            await _broadcast_to_websockets(risk_update)

            await asyncio.sleep(30)  # Update every 30 seconds

        except Exception as e:
            logger.error(f"Error in risk calculation scheduler: {e}")
            await asyncio.sleep(60)

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "risk_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )