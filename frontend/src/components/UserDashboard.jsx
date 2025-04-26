import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Container, Typography, Paper, Button, Grid, Box, Tabs, Tab, Alert } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import PhotoIcon from '@mui/icons-material/Photo';
import StopIcon from '@mui/icons-material/Stop';
import ImageAnalyzer from './ImageAnalyzer';

const PYTHON_API_URL = 'http://localhost:5005';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const UserDashboard = ({ showImageTab = false }) => {
  const location = useLocation();
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [error, setError] = useState(null);
  
  // Check if we should start on the image tab
  useEffect(() => {
    if (showImageTab || location.state?.showImageTab) {
      setTabValue(1); 
    }
  }, [location.state, showImageTab]);
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const startSession = async () => {
    try {
      // For general user, we don't need to ask for patient name
      const response = await fetch(`${PYTHON_API_URL}/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          // No patient name needed for general users
          modelType: 'general'
        }),
        credentials: 'include',
      });
      
      const data = await response.json();
      
      if (data.success) {
        setSessionId(data.sessionId);
        setSessionActive(true);
        setVideoUrl(`${PYTHON_API_URL}/video_feed?session_id=${data.sessionId}`);
      } else {
        console.error('Failed to start session:', data.error);
      }
    } catch (error) {
      console.error('Error starting session:', error);
    }
  };
  
  const stopSession = async () => {
    if (sessionId) {
      try {
        const response = await fetch(`${PYTHON_API_URL}/stop-session`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sessionId: sessionId
          }),
          credentials: 'include',
        });
        
        await response.json();
        
        setSessionActive(false);
        setSessionId(null);
        setVideoUrl(null);
      } catch (error) {
        console.error('Error stopping session:', error);
      }
    }
  };
  
  return (
    <Container>
      <Typography variant="h4" gutterBottom sx={{ mt: 3, mb: 4 }}>
        Emotion Analysis Dashboard
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Paper sx={{ width: '100%', mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          centered
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<VideocamIcon />} label="Webcam Analysis" />
          <Tab icon={<PhotoIcon />} label="Photo Analysis" />
        </Tabs>
        
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8} sx={{ margin: '0 auto' }}>
              <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
                <Box sx={{ textAlign: 'center', mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Webcam Emotion Detection
                  </Typography>
                  
                  {!sessionActive ? (
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<VideocamIcon />}
                      onClick={startSession}
                      sx={{ mt: 2 }}
                    >
                      Start Webcam Analysis
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<StopIcon />}
                      onClick={stopSession}
                      sx={{ mt: 2 }}
                    >
                      Stop Analysis
                    </Button>
                  )}
                </Box>
                
                {sessionActive && videoUrl && (
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '480px', 
                      overflow: 'hidden',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      bgcolor: 'black',
                      borderRadius: 1
                    }}
                  >
                    <img 
                      src={videoUrl} 
                      alt="Emotion Detection Feed" 
                      style={{ maxWidth: '100%', maxHeight: '100%' }}
                    />
                  </Box>
                )}
                
                {!sessionActive && (
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '480px',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      bgcolor: '#f0f0f0',
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="body1" color="textSecondary">
                      Click "Start Webcam Analysis" to begin
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <ImageAnalyzer 
            userRole="general" 
            onError={(errMsg) => setError(errMsg)}
          />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default UserDashboard;