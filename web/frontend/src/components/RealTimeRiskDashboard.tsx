/**
 * Real-Time Risk Analytics Dashboard
 * React component for displaying live risk metrics and portfolio analytics
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Grid,
  Typography,
  Paper,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

// Types
interface Position {
  symbol: string;
  quantity: number;
  price: number;
  market_value: number;
  weight?: number;
}

interface Portfolio {
  positions: Position[];
  total_value: number;
  last_updated: string;
}

interface RiskMetrics {
  portfolio_metrics: {
    total_value: number;
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
  };
  var_metrics: {
    historical_var: { var: number; expected_shortfall: number };
    parametric_var: { var: number; expected_shortfall: number };
    monte_carlo_var: { var: number; expected_shortfall: number };
  };
  risk_decomposition: {
    component_var: number[];
    marginal_var: number[];
    asset_symbols: string[];
  };
  volatility_forecast: {
    forecasted_volatility: number[];
    forecast_horizon: number;
  };
  calculation_timestamp: string;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  timestamp: string;
}

interface WebSocketMessage {
  type: string;
  data: any;
}

// Styled components
const MetricCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'box-shadow 0.3s ease-in-out',
  '&:hover': {
    boxShadow: theme.shadows[8],
  },
}));

const MetricValue = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 'bold',
  color: theme.palette.primary.main,
}));

const ChangeIndicator = styled(Box)<{ positive: boolean }>(({ theme, positive }) => ({
  display: 'flex',
  alignItems: 'center',
  color: positive ? theme.palette.success.main : theme.palette.error.main,
}));

const StatusIndicator = styled(Chip)<{ status: 'healthy' | 'warning' | 'critical' }>(({ theme, status }) => ({
  backgroundColor:
    status === 'healthy'
      ? theme.palette.success.main
      : status === 'warning'
      ? theme.palette.warning.main
      : theme.palette.error.main,
  color: theme.palette.common.white,
  fontWeight: 'bold',
}));

// API URLs
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

// Custom hooks
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');

      // Subscribe to all topics
      ws.send(JSON.stringify({
        type: 'subscribe',
        topics: ['positions', 'risk_metrics', 'market_data', 'alerts']
      }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    // Cleanup
    return () => {
      ws.close();
    };
  }, [url]);

  // Send ping every 30 seconds to keep connection alive
  useEffect(() => {
    if (!socket || !connected) return;

    const interval = setInterval(() => {
      socket.send(JSON.stringify({ type: 'ping' }));
    }, 30000);

    return () => clearInterval(interval);
  }, [socket, connected]);

  return { socket, connected, lastMessage };
};

const useRiskMetrics = () => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculateRiskMetrics = useCallback(async (portfolio: Portfolio) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/risk-metrics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio,
          confidence_level: 0.95,
          lookback_days: 252,
          monte_carlo_scenarios: 10000,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRiskMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to calculate risk metrics');
    } finally {
      setLoading(false);
    }
  }, []);

  return { riskMetrics, loading, error, calculateRiskMetrics };
};

// Sample portfolio data
const SAMPLE_PORTFOLIO: Portfolio = {
  positions: [
    { symbol: 'AAPL', quantity: 100, price: 150.25, market_value: 15025 },
    { symbol: 'GOOGL', quantity: 50, price: 2800.50, market_value: 140025 },
    { symbol: 'MSFT', quantity: 75, price: 330.75, market_value: 24806.25 },
    { symbol: 'AMZN', quantity: 40, price: 3200.00, market_value: 128000 },
    { symbol: 'TSLA', quantity: 30, price: 800.00, market_value: 24000 },
  ],
  total_value: 331856.25,
  last_updated: new Date().toISOString(),
};

// Calculate weights
SAMPLE_PORTFOLIO.positions = SAMPLE_PORTFOLIO.positions.map(pos => ({
  ...pos,
  weight: pos.market_value / SAMPLE_PORTFOLIO.total_value,
}));

const RealTimeRiskDashboard: React.FC = () => {
  const [portfolio] = useState<Portfolio>(SAMPLE_PORTFOLIO);
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'warning' | 'critical'>('healthy');

  const { connected, lastMessage } = useWebSocket(WS_URL);
  const { riskMetrics, loading, error, calculateRiskMetrics } = useRiskMetrics();

  // Initialize risk metrics calculation
  useEffect(() => {
    calculateRiskMetrics(portfolio);
  }, [calculateRiskMetrics, portfolio]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    switch (lastMessage.type) {
      case 'market_data_update':
        setMarketData(prev => ({
          ...prev,
          [lastMessage.data.symbol]: lastMessage.data,
        }));
        break;

      case 'risk_metrics_update':
        // Could update risk metrics in real-time
        console.log('Risk metrics update:', lastMessage.data);
        break;

      case 'alert':
        // Handle risk alerts
        console.log('Risk alert:', lastMessage.data);
        if (lastMessage.data.severity === 'high') {
          setSystemStatus('critical');
        } else if (lastMessage.data.severity === 'medium') {
          setSystemStatus('warning');
        }
        break;

      default:
        break;
    }
  }, [lastMessage]);

  // Chart configurations
  const portfolioCompositionData = useMemo(() => ({
    labels: portfolio.positions.map(pos => pos.symbol),
    datasets: [
      {
        data: portfolio.positions.map(pos => pos.weight! * 100),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
        ],
        borderWidth: 2,
        borderColor: '#fff',
      },
    ],
  }), [portfolio]);

  const varComparisonData = useMemo(() => {
    if (!riskMetrics) return null;

    return {
      labels: ['Historical VaR', 'Parametric VaR', 'Monte Carlo VaR'],
      datasets: [
        {
          label: 'VaR (% of Portfolio)',
          data: [
            riskMetrics.var_metrics.historical_var.var * 100,
            riskMetrics.var_metrics.parametric_var.var * 100,
            riskMetrics.var_metrics.monte_carlo_var.var * 100,
          ],
          backgroundColor: ['rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)', 'rgba(255, 206, 86, 0.6)'],
          borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)'],
          borderWidth: 1,
        },
        {
          label: 'Expected Shortfall (% of Portfolio)',
          data: [
            riskMetrics.var_metrics.historical_var.expected_shortfall * 100,
            riskMetrics.var_metrics.parametric_var.expected_shortfall * 100,
            riskMetrics.var_metrics.monte_carlo_var.expected_shortfall * 100,
          ],
          backgroundColor: ['rgba(255, 99, 132, 0.3)', 'rgba(54, 162, 235, 0.3)', 'rgba(255, 206, 86, 0.3)'],
          borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)'],
          borderWidth: 1,
        },
      ],
    };
  }, [riskMetrics]);

  const riskContributionData = useMemo(() => {
    if (!riskMetrics) return null;

    return {
      labels: riskMetrics.risk_decomposition.asset_symbols,
      datasets: [
        {
          label: 'Component VaR (%)',
          data: riskMetrics.risk_decomposition.component_var.map(x => x * 100),
          backgroundColor: 'rgba(75, 192, 192, 0.6)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1,
        },
      ],
    };
  }, [riskMetrics]);

  const volatilityForecastData = useMemo(() => {
    if (!riskMetrics) return null;

    return {
      labels: Array.from({ length: riskMetrics.volatility_forecast.forecast_horizon }, (_, i) => `Day ${i + 1}`),
      datasets: [
        {
          label: 'Forecasted Volatility (%)',
          data: riskMetrics.volatility_forecast.forecasted_volatility.map(x => x * 100),
          borderColor: 'rgba(153, 102, 255, 1)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.4,
          fill: true,
        },
      ],
    };
  }, [riskMetrics]);

  const formatCurrency = (value: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const formatPercentage = (value: number) =>
    new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value);

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">
          <Typography variant="h6">Error Loading Risk Dashboard</Typography>
          <Typography>{error}</Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h3" component="h1" gutterBottom>
          Real-Time Risk Analytics Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <StatusIndicator
            status={systemStatus}
            label={`System ${systemStatus.toUpperCase()}`}
            icon={systemStatus === 'healthy' ? <CheckCircleIcon /> : <WarningIcon />}
          />
          <Chip
            label={connected ? 'Live' : 'Disconnected'}
            color={connected ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>
      </Box>

      {loading && (
        <Box display="flex" justifyContent="center" mb={3}>
          <CircularProgress size={40} />
          <Typography variant="body1" ml={2}>
            Calculating risk metrics...
          </Typography>
        </Box>
      )}

      <Grid container spacing={3}>
        {/* Portfolio Overview */}
        <Grid item xs={12} md={6} lg={3}>
          <MetricCard>
            <CardHeader title="Portfolio Value" />
            <CardContent>
              <MetricValue>{formatCurrency(portfolio.total_value)}</MetricValue>
              <Typography variant="body2" color="textSecondary">
                {portfolio.positions.length} positions
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Expected Return */}
        <Grid item xs={12} md={6} lg={3}>
          <MetricCard>
            <CardHeader title="Expected Return" />
            <CardContent>
              {riskMetrics ? (
                <>
                  <MetricValue>{formatPercentage(riskMetrics.portfolio_metrics.expected_return)}</MetricValue>
                  <ChangeIndicator positive={riskMetrics.portfolio_metrics.expected_return > 0}>
                    {riskMetrics.portfolio_metrics.expected_return > 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    <Typography variant="body2" ml={1}>
                      Annualized
                    </Typography>
                  </ChangeIndicator>
                </>
              ) : (
                <LinearProgress />
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Portfolio Volatility */}
        <Grid item xs={12} md={6} lg={3}>
          <MetricCard>
            <CardHeader title="Portfolio Volatility" />
            <CardContent>
              {riskMetrics ? (
                <>
                  <MetricValue>{formatPercentage(riskMetrics.portfolio_metrics.volatility)}</MetricValue>
                  <Typography variant="body2" color="textSecondary">
                    Annualized
                  </Typography>
                </>
              ) : (
                <LinearProgress />
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Sharpe Ratio */}
        <Grid item xs={12} md={6} lg={3}>
          <MetricCard>
            <CardHeader title="Sharpe Ratio" />
            <CardContent>
              {riskMetrics ? (
                <>
                  <MetricValue>{riskMetrics.portfolio_metrics.sharpe_ratio.toFixed(2)}</MetricValue>
                  <Typography variant="body2" color="textSecondary">
                    Risk-adjusted return
                  </Typography>
                </>
              ) : (
                <LinearProgress />
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Portfolio Composition */}
        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardHeader title="Portfolio Composition" />
            <CardContent>
              <Box height={300}>
                <Pie
                  data={portfolioCompositionData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                      },
                      tooltip: {
                        callbacks: {
                          label: (context) => {
                            const label = context.label || '';
                            const value = context.parsed;
                            return `${label}: ${value.toFixed(1)}%`;
                          },
                        },
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>

        {/* VaR Comparison */}
        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardHeader title="Value-at-Risk Comparison (95% Confidence)" />
            <CardContent>
              {varComparisonData ? (
                <Box height={300}>
                  <Bar
                    data={varComparisonData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'top',
                        },
                        tooltip: {
                          callbacks: {
                            label: (context) => {
                              const value = context.parsed.y;
                              return `${context.dataset.label}: ${value.toFixed(2)}%`;
                            },
                          },
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          title: {
                            display: true,
                            text: 'Percentage of Portfolio',
                          },
                        },
                      },
                    }}
                  />
                </Box>
              ) : (
                <Box display="flex" justifyContent="center" alignItems="center" height={300}>
                  <CircularProgress />
                </Box>
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Risk Contribution */}
        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardHeader title="Risk Contribution by Asset" />
            <CardContent>
              {riskContributionData ? (
                <Box height={300}>
                  <Bar
                    data={riskContributionData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          title: {
                            display: true,
                            text: 'Component VaR (%)',
                          },
                        },
                      },
                    }}
                  />
                </Box>
              ) : (
                <Box display="flex" justifyContent="center" alignItems="center" height={300}>
                  <CircularProgress />
                </Box>
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Volatility Forecast */}
        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardHeader title="GARCH Volatility Forecast" />
            <CardContent>
              {volatilityForecastData ? (
                <Box height={300}>
                  <Line
                    data={volatilityForecastData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          title: {
                            display: true,
                            text: 'Volatility (%)',
                          },
                        },
                      },
                    }}
                  />
                </Box>
              ) : (
                <Box display="flex" justifyContent="center" alignItems="center" height={300}>
                  <CircularProgress />
                </Box>
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Detailed Risk Metrics Table */}
        <Grid item xs={12}>
          <MetricCard>
            <CardHeader title="Detailed Risk Metrics" />
            <CardContent>
              {riskMetrics ? (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Metric</strong></TableCell>
                        <TableCell align="right"><strong>Value</strong></TableCell>
                        <TableCell align="right"><strong>Dollar Amount</strong></TableCell>
                        <TableCell><strong>Description</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Historical VaR (95%)</TableCell>
                        <TableCell align="right">{formatPercentage(riskMetrics.var_metrics.historical_var.var)}</TableCell>
                        <TableCell align="right">{formatCurrency(riskMetrics.var_metrics.historical_var.var * portfolio.total_value)}</TableCell>
                        <TableCell>Maximum expected loss based on historical data</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Parametric VaR (95%)</TableCell>
                        <TableCell align="right">{formatPercentage(riskMetrics.var_metrics.parametric_var.var)}</TableCell>
                        <TableCell align="right">{formatCurrency(riskMetrics.var_metrics.parametric_var.var * portfolio.total_value)}</TableCell>
                        <TableCell>VaR assuming normal distribution</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Monte Carlo VaR (95%)</TableCell>
                        <TableCell align="right">{formatPercentage(riskMetrics.var_metrics.monte_carlo_var.var)}</TableCell>
                        <TableCell align="right">{formatCurrency(riskMetrics.var_metrics.monte_carlo_var.var * portfolio.total_value)}</TableCell>
                        <TableCell>VaR from Monte Carlo simulation</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Expected Shortfall</TableCell>
                        <TableCell align="right">{formatPercentage(riskMetrics.var_metrics.historical_var.expected_shortfall)}</TableCell>
                        <TableCell align="right">{formatCurrency(riskMetrics.var_metrics.historical_var.expected_shortfall * portfolio.total_value)}</TableCell>
                        <TableCell>Average loss beyond VaR</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Maximum Drawdown</TableCell>
                        <TableCell align="right">{formatPercentage(riskMetrics.portfolio_metrics.max_drawdown)}</TableCell>
                        <TableCell align="right">{formatCurrency(riskMetrics.portfolio_metrics.max_drawdown * portfolio.total_value)}</TableCell>
                        <TableCell>Largest peak-to-trough decline</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Box display="flex" justifyContent="center" p={4}>
                  <CircularProgress />
                </Box>
              )}
            </CardContent>
          </MetricCard>
        </Grid>

        {/* Position Details */}
        <Grid item xs={12}>
          <MetricCard>
            <CardHeader title="Position Details" />
            <CardContent>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Symbol</strong></TableCell>
                      <TableCell align="right"><strong>Quantity</strong></TableCell>
                      <TableCell align="right"><strong>Price</strong></TableCell>
                      <TableCell align="right"><strong>Market Value</strong></TableCell>
                      <TableCell align="right"><strong>Weight</strong></TableCell>
                      <TableCell align="right"><strong>Live Price</strong></TableCell>
                      <TableCell align="right"><strong>Change</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio.positions.map((position) => {
                      const liveData = marketData[position.symbol];
                      return (
                        <TableRow key={position.symbol}>
                          <TableCell>{position.symbol}</TableCell>
                          <TableCell align="right">{position.quantity.toLocaleString()}</TableCell>
                          <TableCell align="right">{formatCurrency(position.price)}</TableCell>
                          <TableCell align="right">{formatCurrency(position.market_value)}</TableCell>
                          <TableCell align="right">{formatPercentage(position.weight!)}</TableCell>
                          <TableCell align="right">
                            {liveData ? formatCurrency(liveData.price) : '-'}
                          </TableCell>
                          <TableCell align="right">
                            {liveData && (
                              <ChangeIndicator positive={liveData.change > 0}>
                                {liveData.change > 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                                {formatPercentage(liveData.change)}
                              </ChangeIndicator>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RealTimeRiskDashboard;