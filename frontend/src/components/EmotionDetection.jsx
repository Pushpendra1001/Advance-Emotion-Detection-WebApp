import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { 
  Box, 
  Container, 
  Typography, 
  Paper,
  CircularProgress,
  Button,
  Grid,
  Divider,
  List,
  ListItem,
  ListItemText,
  LinearProgress
} from '@mui/material'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import { PieChart, Pie, Tooltip, Legend, Cell, ResponsiveContainer } from 'recharts';
import axios from 'axios'

const modelPaths = {
  'doctor-patient': 'src/models/my_model.keras',
  'teacher-student': 'src/models/my_model.keras', 
  'general-analysis': 'src/models/my_model.keras'
};

const PYTHON_API_URL = import.meta.env.VITE_PYTHON_API_URL || 'http://localhost:5005';

async function loadModel(modelPath) {
  try {
    const response = await fetch('/load-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelPath }),
      credentials: 'include'
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      // Show error message to user
      console.error('Model loading error:', data.error);
      return { success: false, error: data.error };
    }
    
    console.log('Model loaded successfully:', data);
    return { success: true, data };
  } catch (error) {
    console.error('Error loading model:', error);
    return { success: false, error: error.message };
  }
}

async function checkModelStatus() {
  try {
    const response = await fetch('/model-status', {
      credentials: 'include'
    });
    
    if (!response.ok) {
      return { loaded: false };
    }
    
    const data = await response.json();
    return { loaded: data.status === 'loaded', info: data.model_info };
  } catch (error) {
    console.error('Error checking model status:', error);
    return { loaded: false, error: error.message };
  }
}

async function fetchAvailableModels() {
  try {
    const response = await fetch('/check-models', {
      credentials: 'include'
    });
    
    if (!response.ok) {
      return { success: false, models: [] };
    }
    
    const data = await response.json();
    return { 
      success: true, 
      models: data.available_models || [],
      cwd: data.cwd
    };
  } catch (error) {
    console.error('Error fetching models:', error);
    return { success: false, models: [], error: error.message };
  }
}

const SessionReport = ({ report }) => {
  if (!report) return null;

  // Calculate max confidence for color coding
  const maxConfidence = Math.max(...Object.values(report.emotionPercentages || {}));

  return (
    <Box>
      <List>
        <ListItem>
          <ListItemText 
            primary={
              <Typography variant="h6" color="primary">
                Session Summary
              </Typography>
            }
          />
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText 
            primary="Session Duration" 
            secondary={
              <Typography variant="body1" color="text.secondary">
                {report.duration.toFixed(1)} minutes
              </Typography>
            }
          />
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText 
            primary="Total Detections" 
            secondary={
              <Typography variant="body1" color="text.secondary">
                {report.totalDetections} emotions detected
              </Typography>
            }
          />
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText 
            primary="Dominant Emotion" 
            secondary={
              <Typography 
                variant="body1" 
                color="primary"
                sx={{ fontWeight: 'bold' }}
              >
                {report.dominantEmotion || 'None detected'}
              </Typography>
            }
          />
        </ListItem>
        <Divider />
        <ListItem>
          <ListItemText 
            primary="Emotion Breakdown"
            secondary={
              <Box sx={{ mt: 1 }}>
                {Object.entries(report.emotionPercentages || {}).map(([emotion, percentage]) => (
                  <Box 
                    key={emotion} 
                    sx={{ 
                      mb: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}
                  >
                    <Typography variant="body2" sx={{ width: '30%' }}>
                      {emotion}:
                    </Typography>
                    <Box sx={{ width: '60%', mr: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={percentage}
                        sx={{
                          height: 8,
                          borderRadius: 5,
                          backgroundColor: 'grey.200',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: percentage === maxConfidence ? 'primary.main' : 'secondary.main'
                          }
                        }}
                      />
                    </Box>
                    <Typography variant="body2" sx={{ width: '10%' }}>
                      {percentage.toFixed(1)}%
                    </Typography>
                  </Box>
                ))}
              </Box>
            }
          />
        </ListItem>
      </List>
      
      {/* Add real-time chart */}
      {report.emotionBreakdown && Object.keys(report.emotionBreakdown).length > 0 && (
        <Box sx={{ mt: 3, height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={Object.entries(report.emotionBreakdown).map(([emotion, count]) => ({
                  name: emotion,
                  value: count
                }))}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
              >
                {Object.entries(report.emotionBreakdown).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
};

export default function EmotionDetection() {
  const { modelId } = useParams()
  const navigate = useNavigate()
  const [isConnected, setIsConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [isTracking, setIsTracking] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [sessionReport, setSessionReport] = useState(null)

  useEffect(() => {
    const connectToBackend = async () => {
      try {
        console.log('Connecting to backend at:', PYTHON_API_URL);
        setLoading(true);
        setError(null);
        
        const response = await fetch(`${PYTHON_API_URL}/load-model`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
            // Don't add Access-Control-Allow-Origin header here - it's causing the CORS error
          },
          credentials: 'include', // Add this for cookies
          body: JSON.stringify({
            modelPath: modelPaths[modelId] || '/models/emotion_model.keras'
          })
        });
        
        console.log('Model loading response status:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to load model');
        }
        
        const data = await response.json();
        console.log('Model loading success:', data);
        
        setIsConnected(true);
        // Start the video feed automatically once connected
        await startVideoFeed();
      } catch (err) {
        console.error('Error connecting to Python backend:', err);
        setError(`Could not connect to server: ${err.message}`);
        setIsConnected(false);
      } finally {
        setLoading(false);
      }
    };

    connectToBackend();
  }, [modelId]);

  const startSession = async () => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/model-status`);
      if (!response.ok) {
        throw new Error('Failed to check model status');
      }
      const modelStatus = await response.json();
      
      if (!modelStatus.loaded) {
        alert('No model is loaded. Please load a model first.');
        return;
      }
      
      // Proceed with starting session...
      await startTracking();
    } catch (err) {
      console.error('Error checking model status:', err);
      setError(err.message);
    }
  };

  const startTracking = async () => {
    try {
      setError(null);
      
      // Check camera permissions first
      const hasPermission = await checkCameraPermission();
      if (!hasPermission) {
        return;
      }
      
      console.log('Starting tracking session...');
      
      const response = await fetch(`${PYTHON_API_URL}/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ modelType: modelId })
      });
      
      console.log('Start session response:', response.status);
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to start session');
      }
      
      const data = await response.json();
      console.log('Session started successfully:', data);
      
      setSessionId(data.sessionId);
      setIsTracking(true);
      setSessionReport(null);
      
      // Start video feed with new session ID
      await startVideoFeed();
    } catch (err) {
      console.error('Error starting tracking:', err);
      setError(err.message);
    }
  }

  const stopTracking = useCallback(async () => {
    try {
      if (!sessionId) return;
      
      // Stop the video feed
      const videoElement = document.getElementById('video-feed');
      if (videoElement) {
        videoElement.srcObject = null;
        videoElement.src = '';
        videoElement.style.display = 'none';
      }
      
      const response = await fetch(`${PYTHON_API_URL}/stop-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sessionId })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to stop session');
      }
      
      const data = await response.json();
      setSessionReport(data);
      setIsTracking(false);
      setSessionId(null);
    } catch (err) {
      console.error('Error stopping tracking:', err);
      setError(err.message);
    }
  }, [sessionId]);

  // Cleanup effect
  useEffect(() => {
    return () => {
      // Cleanup video elements when component unmounts
      const videoElement = document.getElementById('video-feed');
      const overlayElement = document.getElementById('emotion-overlay');
      
      if (videoElement && videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
      
      if (overlayElement) {
        overlayElement.remove();
      }
    };
  }, []);

  const startVideoFeed = async () => {
    try {
      const videoElement = document.getElementById('video-feed');
      if (!videoElement) {
        throw new Error('Video element not found');
      }

      // Create video URL with timestamp to prevent caching
      const videoUrl = new URL(`${PYTHON_API_URL}/video_feed`);
      videoUrl.searchParams.append('model_type', modelId);
      videoUrl.searchParams.append('t', Date.now());  // Prevent caching
      if (sessionId) {
        videoUrl.searchParams.append('session_id', sessionId);
      }

      // Set up video element
      videoElement.style.display = 'block';
      videoElement.src = videoUrl.toString();
      
      console.log('Starting video feed from:', videoUrl.toString());

    } catch (err) {
      console.error('Error starting video feed:', err);
      setError('Failed to start video feed: ' + err.message);
    }
  };

  // Add this function near the top of your component
  const checkCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      // Stop the stream immediately as we just needed to check permissions
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch (err) {
      console.error('Camera permission error:', err);
      setError('Camera access denied. Please enable camera permissions.');
      return false;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/model-selection')}
        sx={{ mb: 2 }}
      >
        Back to Model Selection
      </Button>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
              {modelId.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('/')} Emotion Detection
            </Typography>

            {error && (
              <Typography color="error" sx={{ mb: 2 }}>
                Error: {error}
              </Typography>
            )}

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
                <Typography sx={{ ml: 2 }}>Connecting to server...</Typography>
              </Box>
            ) : isConnected ? (
              <>
                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="contained"
                    color={isTracking ? "error" : "primary"}
                    startIcon={isTracking ? <StopIcon /> : <PlayArrowIcon />}
                    onClick={isTracking ? stopTracking : startSession}
                  >
                    {isTracking ? "Stop Tracking" : "Start Tracking"}
                  </Button>
                </Box>

                <Box
                  sx={{
                    width: '100%',
                    maxWidth: 800,
                    height: 600,
                    margin: '0 auto',
                    position: 'relative',
                    overflow: 'hidden',
                    borderRadius: 2,
                    boxShadow: 3,
                    bgcolor: 'black',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  <img
                    id="video-feed"
                    alt="Emotion Detection Feed"
                    style={{ 
                      width: '100%', 
                      height: '100%', 
                      objectFit: 'contain',
                      display: isTracking ? 'block' : 'none'
                    }}
                  />
                  {!isTracking && (
                    <Typography variant="h6" color="white" sx={{ textAlign: 'center' }}>
                      Click "Start Tracking" to begin emotion detection
                    </Typography>
                  )}
                </Box>
              </>
            ) : (
              <Box sx={{ textAlign: 'center', p: 4 }}>
                <Typography color="error">
                  Could not connect to the emotion detection service.
                </Typography>
                <Button 
                  variant="contained" 
                  onClick={() => window.location.reload()}
                  sx={{ mt: 2 }}
                >
                  Retry Connection
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Session Report
            </Typography>
            {isTracking ? (
              <Typography variant="body2" color="text.secondary">
                Session in progress...
              </Typography>
            ) : sessionReport ? (
              <SessionReport report={sessionReport} />
            ) : (
              <Typography variant="body2" color="text.secondary">
                No session report available. Start tracking to generate a report.
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}
