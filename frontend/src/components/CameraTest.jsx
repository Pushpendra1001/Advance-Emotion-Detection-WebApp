import { useState, useEffect, useRef } from 'react';
import { Box, Typography, Button, Container } from '@mui/material';

const CameraTest = () => {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setError(null);
      }
    } catch (err) {
      console.error("Camera access error:", err);
      setError(`Camera error: ${err.message}`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Camera Test
      </Typography>
      
      {error && (
        <Typography color="error" align="center" gutterBottom>
          {error}
        </Typography>
      )}
      
      <Box 
        sx={{ 
          my: 2, 
          p: 1, 
          border: '1px solid #ccc', 
          borderRadius: 2,
          backgroundColor: '#000',
          height: 480,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        {cameraActive ? (
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            style={{ maxWidth: '100%', maxHeight: '100%' }}
          />
        ) : (
          <Typography variant="body1" color="#fff">
            Camera is not active
          </Typography>
        )}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button 
          variant="contained" 
          onClick={startCamera}
          disabled={cameraActive}
        >
          Start Camera
        </Button>
        <Button 
          variant="outlined" 
          onClick={stopCamera}
          disabled={!cameraActive}
        >
          Stop Camera
        </Button>
      </Box>
    </Container>
  );
};

export default CameraTest;