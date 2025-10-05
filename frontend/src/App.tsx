import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  TextField,
  Button,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  LinearProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Rocket,
  Public,
  Science,
  Analytics,
  CheckCircle,
  Cancel,
  Help,
  CloudUpload,
  Download,
  TableChart
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import axios from 'axios';
import Papa from 'papaparse';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4',
    },
    secondary: {
      main: '#ff9800',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Orbitron", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

interface PredictionData {
  prediction: string;
  confidence: number;
  probabilities: {
    'FALSE POSITIVE': number;
    'CANDIDATE': number;
    'CONFIRMED': number;
  };
}

interface BatchResult {
  message: string;
  total_predictions: number;
  prediction_summary: {
    'FALSE POSITIVE': number;
    'CANDIDATE': number;
    'CONFIRMED': number;
  };
  pdf_download_url: string;
  results: any[];
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [formData, setFormData] = useState({
    koi_period: '',
    koi_duration: '',
    koi_depth: '',
    koi_prad: '',
    koi_teq: '',
    koi_insol: '',
    koi_model_snr: '',
    koi_fpflag_nt: '0',
    koi_fpflag_ss: '0',
    koi_fpflag_co: '0',
    koi_fpflag_ec: '0',
    koi_steff: '',
    koi_slogg: '',
    koi_srad: '',
    koi_kepmag: ''
  });

  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/api/predict', {
        ...formData,
        koi_fpflag_nt: parseInt(formData.koi_fpflag_nt),
        koi_fpflag_ss: parseInt(formData.koi_fpflag_ss),
        koi_fpflag_co: parseInt(formData.koi_fpflag_co),
        koi_fpflag_ec: parseInt(formData.koi_fpflag_ec),
      });
      
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleCsvUpload = async () => {
    if (!csvFile) {
      setError('Please select a CSV file first.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', csvFile);
      
      const response = await axios.post('http://localhost:8000/api/upload-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setBatchResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to process CSV file.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setCsvFile(file);
      setError(null);
    }
  };

  const downloadPdf = () => {
    if (batchResult?.pdf_download_url) {
      window.open(`http://localhost:8000${batchResult.pdf_download_url}`, '_blank');
    }
  };

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'CONFIRMED': return 'success';
      case 'CANDIDATE': return 'warning';
      case 'FALSE POSITIVE': return 'error';
      default: return 'default';
    }
  };

  const getPredictionIcon = (prediction: string) => {
    switch (prediction) {
      case 'CONFIRMED': return <CheckCircle />;
      case 'CANDIDATE': return <Help />;
      case 'FALSE POSITIVE': return <Cancel />;
      default: return <Science />;
    }
  };

  const chartData = prediction ? [
    { name: 'False Positive', value: prediction.probabilities['FALSE POSITIVE'] * 100, color: '#f44336' },
    { name: 'Candidate', value: prediction.probabilities['CANDIDATE'] * 100, color: '#ff9800' },
    { name: 'Confirmed', value: prediction.probabilities['CONFIRMED'] * 100, color: '#4caf50' }
  ] : [];

  const batchChartData = batchResult ? [
    { name: 'False Positive', value: batchResult.prediction_summary['FALSE POSITIVE'], color: '#f44336' },
    { name: 'Candidate', value: batchResult.prediction_summary['CANDIDATE'], color: '#ff9800' },
    { name: 'Confirmed', value: batchResult.prediction_summary['CONFIRMED'], color: '#4caf50' }
  ] : [];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
        width: '100%',
        overflow: 'hidden'
      }}>
        <Container 
          maxWidth={false} 
          sx={{ 
            py: 4, 
            px: { xs: 2, sm: 3, md: 4 },
            width: '100%',
            maxWidth: '100%'
          }}
        >
          <Box textAlign="center" mb={6}>
            <Rocket sx={{ fontSize: { xs: 40, sm: 50, md: 60 }, color: 'primary.main', mb: 2 }} />
            <Typography 
              variant="h2" 
              component="h1" 
              gutterBottom 
              sx={{ 
                background: 'linear-gradient(45deg, #00bcd4, #ff9800)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 'bold',
                fontSize: { xs: '2rem', sm: '2.5rem', md: '3rem' }
              }}
            >
              NASA Exoplanet Predictor
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ fontSize: { xs: '1rem', sm: '1.1rem' } }}>
              Advanced Machine Learning for Exoplanet Classification
            </Typography>
          </Box>

          <Paper elevation={8} sx={{ borderRadius: 3, overflow: 'hidden' }}>
            <Tabs 
              value={activeTab} 
              onChange={(e, newValue) => setActiveTab(newValue)}
              sx={{ 
                background: 'linear-gradient(45deg, #1a1a1a, #2a2a2a)',
                '& .MuiTab-root': { color: 'white' }
              }}
            >
              <Tab 
                icon={<Science />} 
                label="Single Prediction" 
                iconPosition="start"
                sx={{ minHeight: 60 }}
              />
              <Tab 
                icon={<CloudUpload />} 
                label="Batch CSV Upload" 
                iconPosition="start"
                sx={{ minHeight: 60 }}
              />
            </Tabs>

            {activeTab === 0 && (
              <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
                <Grid container spacing={4}>
                  <Grid item xs={12} lg={8}>
                    <Typography 
                      variant="h4" 
                      gutterBottom 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        fontSize: { xs: '1.5rem', sm: '2rem' }
                      }}
                    >
                      <Science color="primary" />
                      Astronomical Parameters
                    </Typography>
                    
                    <Grid container spacing={{ xs: 2, sm: 3 }} sx={{ mt: 2 }}>
                      {[
                        { key: 'koi_period', label: 'Orbital Period (days)', type: 'number' },
                        { key: 'koi_duration', label: 'Transit Duration (hours)', type: 'number' },
                        { key: 'koi_depth', label: 'Transit Depth (ppm)', type: 'number' },
                        { key: 'koi_prad', label: 'Planetary Radius (Earth radii)', type: 'number' },
                        { key: 'koi_teq', label: 'Equilibrium Temperature (K)', type: 'number' },
                        { key: 'koi_insol', label: 'Insolation Flux (Earth units)', type: 'number' },
                        { key: 'koi_model_snr', label: 'Signal-to-Noise Ratio', type: 'number' },
                        { key: 'koi_steff', label: 'Stellar Temperature (K)', type: 'number' },
                        { key: 'koi_slogg', label: 'Stellar Surface Gravity (log g)', type: 'number' },
                        { key: 'koi_srad', label: 'Stellar Radius (Solar radii)', type: 'number' },
                        { key: 'koi_kepmag', label: 'Kepler Magnitude', type: 'number' }
                      ].map((field) => (
                        <Grid item xs={12} sm={6} md={4} key={field.key}>
                          <TextField
                            fullWidth
                            label={field.label}
                            type={field.type}
                            value={formData[field.key as keyof typeof formData]}
                            onChange={(e) => handleInputChange(field.key, e.target.value)}
                            variant="outlined"
                            size="small"
                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
                          />
                        </Grid>
                      ))}
                      
                      <Grid item xs={12}>
                        <Typography variant="h6" gutterBottom sx={{ mt: 2, fontSize: { xs: '1.1rem', sm: '1.25rem' } }}>
                          Detection Quality Flags
                        </Typography>
                      </Grid>
                      
                      {[
                        { key: 'koi_fpflag_nt', label: 'Not Transit-like' },
                        { key: 'koi_fpflag_ss', label: 'Stellar Eclipse' },
                        { key: 'koi_fpflag_co', label: 'Centroid Offset' },
                        { key: 'koi_fpflag_ec', label: 'Ephemeris Contamination' }
                      ].map((flag) => (
                        <Grid item xs={12} sm={6} md={3} key={flag.key}>
                          <TextField
                            fullWidth
                            label={flag.label}
                            type="number"
                            value={formData[flag.key as keyof typeof formData]}
                            onChange={(e) => handleInputChange(flag.key, e.target.value)}
                            variant="outlined"
                            select
                            SelectProps={{ native: true }}
                            size="small"
                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
                          >
                            <option value="0">No Flag</option>
                            <option value="1">Flagged</option>
                          </TextField>
                        </Grid>
                      ))}
                    </Grid>

                    <Box sx={{ mt: 4, textAlign: 'center' }}>
                      <Button
                        variant="contained"
                        size="large"
                        onClick={handlePredict}
                        disabled={loading}
                        startIcon={loading ? <CircularProgress size={20} /> : <Analytics />}
                        sx={{ 
                          px: { xs: 3, sm: 4 }, 
                          py: 1.5, 
                          borderRadius: 3,
                          background: 'linear-gradient(45deg, #00bcd4, #ff9800)',
                          '&:hover': {
                            background: 'linear-gradient(45deg, #00acc1, #f57c00)',
                          },
                          fontSize: { xs: '0.9rem', sm: '1rem' }
                        }}
                      >
                        {loading ? 'Analyzing...' : 'Predict Exoplanet Classification'}
                      </Button>
                    </Box>
                  </Grid>

                  <Grid item xs={12} lg={4}>
                    <Typography 
                      variant="h4" 
                      gutterBottom 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        fontSize: { xs: '1.5rem', sm: '2rem' }
                      }}
                    >
                      <Public color="primary" />
                      Prediction Results
                    </Typography>

                    {prediction ? (
                      <Box>
                        <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #1a1a1a, #2a2a2a)' }}>
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                              {getPredictionIcon(prediction.prediction)}
                              <Typography variant="h5" sx={{ fontSize: { xs: '1.2rem', sm: '1.5rem' } }}>
                                {prediction.prediction}
                              </Typography>
                            </Box>
                            
                            <Box sx={{ mb: 2 }}>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                Confidence Level
                              </Typography>
                              <LinearProgress 
                                variant="determinate" 
                                value={prediction.confidence * 100} 
                                sx={{ height: 8, borderRadius: 4 }}
                              />
                              <Typography variant="body2" sx={{ mt: 1 }}>
                                {(prediction.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Box>

                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                              {Object.entries(prediction.probabilities).map(([key, value]) => (
                                <Chip
                                  key={key}
                                  label={`${key}: ${(value * 100).toFixed(1)}%`}
                                  color={getPredictionColor(key) as any}
                                  size="small"
                                  sx={{ fontSize: '0.75rem' }}
                                />
                              ))}
                            </Box>
                          </CardContent>
                        </Card>

                        <Box sx={{ height: { xs: 250, sm: 300 } }}>
                          <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1rem', sm: '1.1rem' } }}>
                            Classification Probabilities
                          </Typography>
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                              >
                                {chartData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <RechartsTooltip formatter={(value) => `${value}%`} />
                            </PieChart>
                          </ResponsiveContainer>
                        </Box>
                      </Box>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Science sx={{ fontSize: { xs: 40, sm: 60 }, color: 'text.secondary', mb: 2 }} />
                        <Typography variant="h6" color="text.secondary" sx={{ fontSize: { xs: '0.9rem', sm: '1rem' } }}>
                          Enter astronomical parameters and click "Predict" to get classification results
                        </Typography>
                      </Box>
                    )}
                  </Grid>
                </Grid>
              </Box>
            )}

            {activeTab === 1 && (
              <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
                <Typography 
                  variant="h4" 
                  gutterBottom 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 1,
                    fontSize: { xs: '1.5rem', sm: '2rem' },
                    mb: 3
                  }}
                >
                  <CloudUpload color="primary" />
                  Batch CSV Upload
                </Typography>

                <Grid container spacing={4}>
                  <Grid item xs={12} md={6}>
                    <Card sx={{ p: 3, background: 'linear-gradient(135deg, #1a1a1a, #2a2a2a)' }}>
                      <Typography variant="h6" gutterBottom>
                        Upload CSV File
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                        Upload a CSV file with exoplanet data for batch prediction. The CSV must contain these columns:
                      </Typography>
                      
                      <Box sx={{ mb: 3 }}>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                          koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
                        </Typography>
                      </Box>

                      <input
                        accept=".csv"
                        style={{ display: 'none' }}
                        id="csv-upload"
                        type="file"
                        onChange={handleFileChange}
                      />
                      <label htmlFor="csv-upload">
                        <Button
                          variant="outlined"
                          component="span"
                          startIcon={<CloudUpload />}
                          sx={{ mb: 2, width: '100%' }}
                        >
                          {csvFile ? csvFile.name : 'Choose CSV File'}
                        </Button>
                      </label>

                      <Button
                        variant="contained"
                        onClick={handleCsvUpload}
                        disabled={!csvFile || loading}
                        startIcon={loading ? <CircularProgress size={20} /> : <Analytics />}
                        sx={{ 
                          width: '100%',
                          background: 'linear-gradient(45deg, #00bcd4, #ff9800)',
                          '&:hover': {
                            background: 'linear-gradient(45deg, #00acc1, #f57c00)',
                          }
                        }}
                      >
                        {loading ? 'Processing...' : 'Process CSV'}
                      </Button>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    {batchResult && (
                      <Card sx={{ p: 3, background: 'linear-gradient(135deg, #1a1a1a, #2a2a2a)' }}>
                        <Typography variant="h6" gutterBottom>
                          Batch Results
                        </Typography>
                        
                        <Box sx={{ mb: 3 }}>
                          <Typography variant="body1" gutterBottom>
                            Total Predictions: {batchResult.total_predictions}
                          </Typography>
                          
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                            <Chip 
                              label={`FALSE POSITIVE: ${batchResult.prediction_summary['FALSE POSITIVE']}`} 
                              color="error" 
                              size="small" 
                            />
                            <Chip 
                              label={`CANDIDATE: ${batchResult.prediction_summary['CANDIDATE']}`} 
                              color="warning" 
                              size="small" 
                            />
                            <Chip 
                              label={`CONFIRMED: ${batchResult.prediction_summary['CONFIRMED']}`} 
                              color="success" 
                              size="small" 
                            />
                          </Box>

                          <Button
                            variant="contained"
                            startIcon={<Download />}
                            onClick={downloadPdf}
                            sx={{ 
                              background: 'linear-gradient(45deg, #4caf50, #2196f3)',
                              '&:hover': {
                                background: 'linear-gradient(45deg, #45a049, #1976d2)',
                              }
                            }}
                          >
                            Download PDF Report
                          </Button>
                        </Box>

                        <Box sx={{ height: 300 }}>
                          <Typography variant="h6" gutterBottom>
                            Prediction Distribution
                          </Typography>
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={batchChartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                              >
                                {batchChartData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <RechartsTooltip formatter={(value) => `${value} predictions`} />
                            </PieChart>
                          </ResponsiveContainer>
                        </Box>
                      </Card>
                    )}
                  </Grid>
                </Grid>
              </Box>
            )}

            {error && (
              <Box sx={{ p: 2 }}>
                <Alert severity="error">
                  {error}
                </Alert>
              </Box>
            )}
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App;
