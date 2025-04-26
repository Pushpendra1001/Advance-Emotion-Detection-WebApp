import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

const PYTHON_API_URL = import.meta.env.VITE_PYTHON_API_URL || 'http://localhost:5005';

export default function EmotionDetection() {
  const location = useLocation();
  const navigate = useNavigate();
  const { patientName, modelType } = location.state || {};
  const videoRef = useRef(null);
  const [isTracking, setIsTracking] = useState(false);
  const [sessionReport, setSessionReport] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [emotionData, setEmotionData] = useState([]);

  // Add stopCamera function
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => {
    if (!patientName || !modelType) {
      navigate('/model-selection');
      return;
    }

    // Cleanup function
    return () => {
      stopCamera();
      if (isTracking) {
        stopTracking();
      }
    };
  }, [patientName, modelType, navigate, isTracking]);

  const startTracking = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setEmotionData([]);
      setStartTime(new Date());

      // Initialize session with backend
      const sessionResponse = await fetch(`${PYTHON_API_URL}/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          patientName,
          modelType
        })
      });

      if (!sessionResponse.ok) {
        throw new Error('Failed to start session');
      }

      const { sessionId } = await sessionResponse.json();

      // Create video feed URL
      const videoUrl = new URL(`${PYTHON_API_URL}/video_feed`);
      videoUrl.searchParams.append('session_id', sessionId);
      videoUrl.searchParams.append('t', Date.now());

      // Set up video feed
      const videoFeed = document.getElementById('video-feed');
      if (videoFeed) {
        videoFeed.style.display = 'block';
        videoFeed.src = videoUrl.toString();
      }

      setIsTracking(true);

    } catch (err) {
      setError('Failed to start tracking: ' + err.message);
      setIsTracking(false);
      const videoFeed = document.getElementById('video-feed');
      if (videoFeed) {
        videoFeed.src = '';
      }
    } finally {
      setIsLoading(false);
    }
  };

  const stopTracking = async () => {
    try {
      setIsLoading(true);
      setIsTracking(false);

      const response = await fetch(`${PYTHON_API_URL}/stop-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          patientName,
          modelType,
          startTime: startTime?.toISOString(),
          endTime: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error('Failed to stop tracking');
      }

      const data = await response.json();
      setSessionReport(data);

      // Clear video feed
      const videoFeed = document.getElementById('video-feed');
      if (videoFeed) {
        videoFeed.src = '';
        videoFeed.style.display = 'none';
      }

    } catch (err) {
      setError('Failed to stop tracking: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadSessionData = async () => {
    try {
      const csvContent = 'data:text/csv;charset=utf-8,' + 
        'Timestamp,Emotion,Confidence\n' +
        emotionData.map(e => `${e.timestamp},${e.emotion},${e.confidence}`).join('\n');
      
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', `session_${patientName}_${new Date().toISOString()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      setError('Failed to download session data: ' + err.message);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/model-selection')}
        sx={{ mb: 2 }}
      >
        Back to Model Selection
      </Button>

      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Camera Feed - {patientName}
            </Typography>
            <Box sx={{ position: 'relative' }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ 
                  width: '100%', 
                  borderRadius: '8px',
                  display: !isTracking ? 'block' : 'none' 
                }}
              />
              <img
                id="video-feed"
                alt="Emotion Detection Feed"
                style={{ 
                  width: '100%',
                  height: '100%',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  display: isTracking ? 'block' : 'none',
                  objectFit: 'contain'
                }}
              />
              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  color={isTracking ? "error" : "primary"}
                  onClick={isTracking ? stopTracking : startTracking}
                  fullWidth
                  disabled={isLoading}
                >
                  {isLoading ? <CircularProgress size={24} /> : 
                    isTracking ? "Stop Tracking" : "Start Tracking"}
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Session Report
            </Typography>
            {(isTracking || sessionReport) && (
              <List>
                <ListItem>
                  <ListItemText
                    primary="Session Duration"
                    secondary={`${((new Date() - startTime) / 1000 / 60).toFixed(2)} minutes`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Emotions Detected"
                    secondary={emotionData.length}
                  />
                </ListItem>
                {sessionReport && (
                  <ListItem>
                    <ListItemText
                      primary="Dominant Emotion"
                      secondary={sessionReport.dominantEmotion || 'N/A'}
                    />
                  </ListItem>
                )}
              </List>
            )}
            {(isTracking || sessionReport) && (
              <Button
                variant="contained"
                color="primary"
                onClick={downloadSessionData}
                sx={{ mt: 2 }}
                fullWidth
                disabled={emotionData.length === 0}
              >
                Download Session Data
              </Button>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}