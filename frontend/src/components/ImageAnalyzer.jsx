import { useState } from 'react';
import { Box, Button, CircularProgress, Typography, Card, CardMedia, CardContent, Alert, Grid, Chip } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';
import { styled } from '@mui/material/styles';

const PYTHON_API_URL = 'http://localhost:5005';

const Input = styled('input')({
  display: 'none',
});

const EmotionChip = ({ emotion }) => {
  
  const getColor = (emotion) => {
    switch(emotion.toLowerCase()) {
      case 'happy': return 'success';
      case 'angry': return 'error';
      case 'neutral': return 'info';
      default: return 'default';
    }
  };
  
  return (
    <Chip 
      label={emotion} 
      color={getColor(emotion)}
      size="small"
      icon={<SentimentSatisfiedAltIcon />}
      sx={{ m: 0.5 }}
    />
  );
};

const ImageAnalyzer = ({ userRole = 'general', onError }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const handleError = (message) => {
    setError(message);
    if (onError) onError(message);
  };
  
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setError(null);
    
    if (selectedFile) {
      if (!selectedFile.type.includes('image/')) {
        handleError('Please select an image file');
        return;
      }
      
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null); 
    }
  };
  
  const handleAnalyze = async () => {
    if (!file) {
      handleError('Please select an image to analyze');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('userRole', userRole);
      
      console.log("Sending request to:", `${PYTHON_API_URL}/analyze-image`);
      
      const response = await fetch(`${PYTHON_API_URL}/analyze-image`, {
        method: 'POST',
        body: formData,
        credentials: 'include', 
      });
      
      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      
      let responseData;
      try {
        responseData = JSON.parse(responseText);
      } catch (e) {
        throw new Error(`Server responded with invalid JSON: ${responseText.substring(0, 100)}...`);
      }
      
      if (!response.ok) {
        throw new Error(responseData.error || `Server error: ${response.status}`);
      }
      
      console.log("Server response:", responseData);
      setResults(responseData);
    } catch (err) {
      console.error('Error analyzing image:', err);
      handleError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box sx={{ maxWidth: '800px', mx: 'auto' }}>
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <label htmlFor="upload-photo">
          <Input 
            accept="image/*" 
            id="upload-photo" 
            type="file"
            onChange={handleFileChange}
            disabled={loading}
          />
          <Button
            variant="contained"
            component="span"
            startIcon={<CloudUploadIcon />}
            disabled={loading}
            color="primary"
            size="large"
          >
            Upload Photo
          </Button>
        </label>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}
      
      <Grid container spacing={3}>
        {/* Preview Card */}
        {preview && (
          <Grid item xs={12} md={6}>
            <Card sx={{ mb: 2 }}>
              <Typography variant="subtitle1" sx={{ p: 1, bgcolor: 'primary.main', color: 'white' }}>
                Original Image
              </Typography>
              <CardMedia
                component="img"
                image={preview}
                alt="Preview"
                sx={{ 
                  height: 'auto',
                  maxHeight: '300px',
                  objectFit: 'contain',
                  width: '100%' 
                }}
              />
              <CardContent sx={{ textAlign: 'center', p: 2 }}>
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={handleAnalyze}
                  disabled={loading || !file}
                  sx={{ mt: 1 }}
                >
                  {loading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Emotions'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}
        
        {/* Results Card */}
        {results && (
          <Grid item xs={12} md={6}>
            <Card>
              <Typography variant="subtitle1" sx={{ p: 1, bgcolor: 'success.main', color: 'white' }}>
                Analysis Results
              </Typography>
              
              {results.resultImage && (
                <CardMedia
                  component="img"
                  image={`${PYTHON_API_URL}${results.resultImage}`}
                  alt="Analyzed Result"
                  sx={{ 
                    height: 'auto',
                    maxHeight: '300px',
                    objectFit: 'contain',
                    width: '100%'
                  }}
                />
              )}
              
              <CardContent>
                <Typography variant="body1" gutterBottom>
                  <strong>Faces Detected:</strong> {results.facesDetected}
                </Typography>
                
                <Typography variant="body1" gutterBottom>
                  <strong>Emotions:</strong>
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
                  {results.emotions && results.emotions.map((item, index) => (
                    <Box key={index} sx={{ mr: 2, mb: 2 }}>
                      <EmotionChip emotion={item.emotion} />
                      <Typography variant="body2">
                        Confidence: {(item.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  ))}
                </Box>
                
                {(!results.emotions || results.emotions.length === 0) && (
                  <Alert severity="info">No emotions detected in this image.</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
      
      {!preview && !results && (
        <Box sx={{ 
          height: '200px', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          border: '2px dashed #ccc',
          borderRadius: 2,
          mb: 2
        }}>
          <Typography variant="body1" color="text.secondary">
            Upload a photo to analyze emotions
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ImageAnalyzer;