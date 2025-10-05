import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const message = error.response.data?.error || 
                    error.response.data?.detail || 
                    `Server error: ${error.response.status}`;
      throw new Error(message);
    } else if (error.request) {
      throw new Error('Unable to connect to the server. Please check your connection and try again.');
    } else {
      throw new Error('An unexpected error occurred. Please try again.');
    }
  }
);

export const predictExoplanet = async (formData) => {
  try {
    const response = await api.post('/api/predict', formData);
    return response.data;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};

export default api;
