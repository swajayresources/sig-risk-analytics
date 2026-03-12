import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  NotificationsActive,
  Timeline
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

// Type definitions
interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  market_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  last_update: string;
}

interface RiskMetrics {
  portfolio_value: number;
  var_1d: number;
  var_10d: number;
  expected_shortfall: number;
  beta: number;
  sharpe_ratio: number;
  max_drawdown: number;
  last_calculated: string;
}

interface Greeks {
  total_delta: number;
  total_gamma: number;
  total_theta: number;
  total_vega: number;
  total_rho: number;
  last_calculated: string;
}

interface AlertData {
  id: string;
  type: string;
  severity: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

interface MarketData {
  symbol: string;
  price: number;
  timestamp: string;
}

const RiskDashboard: React.FC = () => {
  // State management
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [greeks, setGreeks] = useState<Greeks | null>(null);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [marketData, setMarketData] = useState<Record<string, number>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [realTimeEnabled, setRealTimeEnabled] = useState(true);
  const [priceHistory, setPriceHistory] = useState<Record<string, Array<{time: string, price: number}>>>({});

  // WebSocket connection
  useEffect(() => {
    if (!realTimeEnabled) return;

    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      setIsConnected(true);
      // Subscribe to all data types
      ws.send(JSON.stringify({
        type: 'subscribe',
        topics: ['positions', 'risk_metrics', 'market_data', 'alerts']
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [realTimeEnabled]);

  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'position_update':
        setPositions(prev => ({
          ...prev,
          [message.symbol]: message.position
        }));
        break;

      case 'risk_metrics':
        setRiskMetrics(message.metrics);
        setGreeks(message.greeks);
        break;

      case 'market_data':
        setMarketData(prev => ({
          ...prev,
          [message.symbol]: message.price
        }));

        // Update price history for charts
        setPriceHistory(prev => {
          const symbolHistory = prev[message.symbol] || [];
          const newHistory = [...symbolHistory, {
            time: new Date(message.timestamp).toLocaleTimeString(),
            price: message.price
          }].slice(-50); // Keep last 50 points

          return {
            ...prev,
            [message.symbol]: newHistory
          };
        });
        break;

      case 'alert':
        setAlerts(prev => [message.alert, ...prev].slice(0, 100)); // Keep last 100 alerts
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  // Initial data load
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [positionsRes, riskRes, greeksRes, alertsRes] = await Promise.all([
          fetch('/api/positions', {
            headers: { 'Authorization': 'Bearer demo_token' }
          }),
          fetch('/api/risk-metrics', {
            headers: { 'Authorization': 'Bearer demo_token' }
          }),
          fetch('/api/greeks', {
            headers: { 'Authorization': 'Bearer demo_token' }
          }),
          fetch('/api/alerts?limit=20', {
            headers: { 'Authorization': 'Bearer demo_token' }
          })
        ]);

        if (positionsRes.ok) {
          const positionsData = await positionsRes.json();
          setPositions(positionsData);
        }

        if (riskRes.ok) {
          const riskData = await riskRes.json();
          setRiskMetrics(riskData);
        }

        if (greeksRes.ok) {
          const greeksData = await greeksRes.json();
          setGreeks(greeksData);
        }

        if (alertsRes.ok) {
          const alertsData = await alertsRes.json();
          setAlerts(alertsData.alerts);
        }
      } catch (error) {
        console.error('Error loading initial data:', error);
      }
    };

    loadInitialData();
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'HIGH': return 'error';
      case 'MEDIUM': return 'warning';
      case 'LOW': return 'info';
      default: return 'default';
    }
  };

  const getGreeksChartData = () => {
    if (!greeks) return [];
    return [
      { name: 'Delta', value: Math.abs(greeks.total_delta), color: '#8884d8' },
      { name: 'Gamma', value: Math.abs(greeks.total_gamma), color: '#82ca9d' },
      { name: 'Theta', value: Math.abs(greeks.total_theta), color: '#ffc658' },
      { name: 'Vega', value: Math.abs(greeks.total_vega), color: '#ff7300' },
      { name: 'Rho', value: Math.abs(greeks.total_rho), color: '#00ff00' }
    ];
  };

  const getPositionsPnLData = () => {
    return Object.entries(positions).map(([symbol, position]) => ({
      symbol,
      pnl: position.unrealized_pnl,
      pnlPercent: ((position.market_price - position.avg_price) / position.avg_price) * 100
    })).sort((a, b) => b.pnl - a.pnl);
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Risk Analytics Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={isConnected ? <CheckCircle /> : <Error />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
          />
          <FormControlLabel
            control={
              <Switch
                checked={realTimeEnabled}
                onChange={(e) => setRealTimeEnabled(e.target.checked)}
              />
            }
            label="Real-time Updates"
          />
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Portfolio Value
              </Typography>
              <Typography variant="h5">
                {riskMetrics ? formatCurrency(riskMetrics.portfolio_value) : '—'}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {riskMetrics && riskMetrics.portfolio_value > 0 ? (
                  <TrendingUp color="success" />
                ) : (
                  <TrendingDown color="error" />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                1-Day VaR
              </Typography>
              <Typography variant="h5">
                {riskMetrics ? formatCurrency(riskMetrics.var_1d) : '—'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {riskMetrics && riskMetrics.portfolio_value > 0
                  ? formatPercent(riskMetrics.var_1d / riskMetrics.portfolio_value)
                  : '—'
                } of portfolio
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Sharpe Ratio
              </Typography>
              <Typography variant="h5">
                {riskMetrics ? riskMetrics.sharpe_ratio.toFixed(2) : '—'}
              </Typography>
              <Box sx={{ mt: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={riskMetrics ? Math.min(Math.max((riskMetrics.sharpe_ratio + 1) * 25, 0), 100) : 0}
                  color={riskMetrics && riskMetrics.sharpe_ratio > 1 ? 'success' : 'warning'}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Portfolio Beta
              </Typography>
              <Typography variant="h5">
                {riskMetrics ? riskMetrics.beta.toFixed(2) : '—'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                vs Market
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Position P&L Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Position P&L
              </Typography>
              <Box sx={{ height: 300, width: '100%' }}>
                <ResponsiveContainer>
                  <BarChart data={getPositionsPnLData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="symbol" />
                    <YAxis />
                    <RechartsTooltip
                      formatter={(value: number) => [formatCurrency(value), 'P&L']}
                    />
                    <Bar
                      dataKey="pnl"
                      fill={(entry: any) => entry.pnl >= 0 ? '#4caf50' : '#f44336'}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Greeks Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Options Greeks
              </Typography>
              <Box sx={{ height: 300, width: '100%' }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={getGreeksChartData()}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value.toFixed(0)}`}
                    >
                      {getGreeksChartData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tables Row */}
      <Grid container spacing={3}>
        {/* Positions Table */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Positions
              </Typography>
              <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Avg Price</TableCell>
                      <TableCell align="right">Market Price</TableCell>
                      <TableCell align="right">Unrealized P&L</TableCell>
                      <TableCell align="right">% Return</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(positions).map(([symbol, position]) => {
                      const returnPct = ((position.market_price - position.avg_price) / position.avg_price) * 100;
                      return (
                        <TableRow key={symbol}>
                          <TableCell component="th" scope="row">
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {symbol}
                              {marketData[symbol] && (
                                <Chip
                                  size="small"
                                  label={formatCurrency(marketData[symbol])}
                                  color="primary"
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            {position.quantity.toLocaleString()}
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(position.avg_price)}
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(position.market_price)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main',
                              fontWeight: 'bold'
                            }}
                          >
                            {formatCurrency(position.unrealized_pnl)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: returnPct >= 0 ? 'success.main' : 'error.main'
                            }}
                          >
                            {formatPercent(returnPct / 100)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Alerts Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <NotificationsActive />
                <Typography variant="h6">
                  Risk Alerts
                </Typography>
                <Chip
                  size="small"
                  label={alerts.filter(a => !a.acknowledged).length}
                  color="error"
                />
              </Box>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {alerts.slice(0, 10).map((alert) => (
                  <Alert
                    key={alert.id}
                    severity={getSeverityColor(alert.severity) as any}
                    sx={{ mb: 1 }}
                    action={
                      !alert.acknowledged && (
                        <IconButton
                          size="small"
                          onClick={() => {
                            // Acknowledge alert logic
                            setAlerts(prev =>
                              prev.map(a =>
                                a.id === alert.id ? { ...a, acknowledged: true } : a
                              )
                            );
                          }}
                        >
                          <CheckCircle fontSize="small" />
                        </IconButton>
                      )
                    }
                  >
                    <Typography variant="body2">
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" display="block">
                      {new Date(alert.timestamp).toLocaleString()}
                    </Typography>
                  </Alert>
                ))}
                {alerts.length === 0 && (
                  <Typography variant="body2" color="textSecondary" textAlign="center">
                    No alerts
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Real-time Price Charts */}
      {Object.keys(priceHistory).length > 0 && (
        <Grid container spacing={3} sx={{ mt: 2 }}>
          {Object.entries(priceHistory).slice(0, 4).map(([symbol, history]) => (
            <Grid item xs={12} md={6} key={symbol}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {symbol} - Real-time Price
                  </Typography>
                  <Box sx={{ height: 200, width: '100%' }}>
                    <ResponsiveContainer>
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <RechartsTooltip
                          formatter={(value: number) => [formatCurrency(value), 'Price']}
                        />
                        <Line
                          type="monotone"
                          dataKey="price"
                          stroke="#8884d8"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default RiskDashboard;