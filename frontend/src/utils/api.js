import axios from 'axios';

axios.defaults.withCredentials = true;
axios.interceptors.request.use(function (config) {
  // Don't set this header manually - let the browser handle it
  if (config.headers['Access-Control-Allow-Origin']) {
    delete config.headers['Access-Control-Allow-Origin'];
  }
  return config;
});

const PYTHON_API_URL = import.meta.env.VITE_PYTHON_API_URL;

export const pythonApi = axios.create({
  baseURL: PYTHON_API_URL,
  headers: {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*'
  },
  withCredentials: false
});

export const checkPythonServer = async () => {
  try {
    const response = await pythonApi.get('/status');
    return response.data.status === 'running';
  } catch (error) {
    console.error('Python server check failed:', error);
    return false;
  }
};