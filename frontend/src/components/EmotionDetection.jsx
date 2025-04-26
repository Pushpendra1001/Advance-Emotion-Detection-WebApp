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
  const [videoUrl, setVideoUrl] = useState('');
  const [sessionId, setSessionId] = useState(null); 

  
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
      setSessionId(sessionId); 

      
      const url = new URL(`${PYTHON_API_URL}/video_feed`);
      url.searchParams.append('session_id', sessionId);
      url.searchParams.append('t', Date.now());
      url.searchParams.append('patientName', patientName); 

      setVideoUrl(url.toString());
      console.log("Video feed URL:", url.toString());

      setIsTracking(true);

    } catch (err) {
      console.error("Error starting tracking:", err);
      setError('Failed to start tracking: ' + err.message);
      setIsTracking(false);
      setVideoUrl('');
    } finally {
      setIsLoading(false);
    }
  };

  

  const stopTracking = async () => {
    try {
      setIsLoading(true);
      setIsTracking(false);
      setVideoUrl('');

      const response = await fetch(`${PYTHON_API_URL}/stop-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          patientName,
          modelType,
          sessionId,
          startTime: startTime?.toISOString(),
          endTime: new Date().toISOString()
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setSessionReport(data);
        setSessionId(null);
      } else {
        
        console.error("Server reported an error: ", data.error);
        setSessionReport({
          dominantEmotion: 'Unknown',
          duration: ((new Date() - startTime) / 1000 / 60).toFixed(2)
        });
      }
    } catch (err) {
      console.error("Error stopping tracking:", err);
      
      
      setSessionReport({
        dominantEmotion: 'Unknown',
        duration: ((new Date() - startTime) / 1000 / 60).toFixed(2)
      });
    } finally {
      setIsLoading(false);
      setIsTracking(false); 
      setVideoUrl(''); 
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

  
  const fetchEmotionData = async (sid) => {
    if (!sid) return; 
    
    try {
      const response = await fetch(`${PYTHON_API_URL}/emotion-data?session_id=${sid}`);
      if (response.ok) {
        const data = await response.json();
        if (data && data.emotions) {
          setEmotionData(data.emotions);
        }
      }
    } catch (err) {
      console.error('Error fetching emotion data:', err);
    }
  };

  
  useEffect(() => {
    let intervalId;
    
    if (isTracking && sessionId) {
      
      intervalId = setInterval(() => {
        fetchEmotionData(sessionId);
      }, 2000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTracking, sessionId]); 

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
            <Box sx={{ 
              position: 'relative', 
              minHeight: '400px',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              backgroundColor: '#f0f0f0',
              borderRadius: '8px',
              overflow: 'hidden'
            }}>
              {!isTracking && !videoUrl && (
                <Typography variant="body2" color="text.secondary">
                  Click "Start Tracking" to begin emotion detection
                </Typography>
              )}
              
              {isTracking && videoUrl && (
                <img
                  src={videoUrl}
                  alt="Emotion Detection Feed"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                  }}
                />
              )}
            </Box>
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
                    secondary={`${((new Date() - (startTime || new Date())) / 1000 / 60).toFixed(2)} minutes`}
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