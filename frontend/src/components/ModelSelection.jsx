import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Button,
} from '@mui/material';

// Model definitions for different roles
const modelsByRole = {
  doctor: [
    {
      id: 'doctor-patient',
      title: 'Doctor/Patient Analysis',
      description: 'Detect patient emotions during consultations for better healthcare outcomes',
      image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=500',
    }
  ],
  teacher: [
    {
      id: 'teacher-student',
      title: 'Teacher/Student Analysis',
      description: 'Monitor student engagement and emotional responses during lessons',
      image: 'https://images.unsplash.com/photo-1523580494863-6f3031224c94?auto=format&fit=crop&w=500',
    }
  ],
  general: [
    {
      id: 'general-analysis',
      title: 'General Emotion Analysis',
      description: 'Analyze emotions in various contexts and scenarios',
      image: 'https://images.unsplash.com/photo-1516387938699-a93567ec168e?auto=format&fit=crop&w=500',
    }
  ]
};

// Default models if no role is provided
const defaultModels = [
  {
    id: 'doctor-patient',
    title: 'Doctor/Patient',
    description: 'Emotion detection for healthcare interactions',
    image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=500',
  },
  {
    id: 'teacher-student',
    title: 'Teacher/Student',
    description: 'Emotion detection for educational environments',
    image: 'https://images.unsplash.com/photo-1523580494863-6f3031224c94?auto=format&fit=crop&w=500',
  }
];

export default function ModelSelection() {
  const navigate = useNavigate();
  const location = useLocation();
  const [models, setModels] = useState(defaultModels);
  const [userRole, setUserRole] = useState(null);
  
  useEffect(() => {
    // Get the role from location state
    const role = location.state?.role;
    if (role && modelsByRole[role]) {
      setModels(modelsByRole[role]);
      setUserRole(role);
    }
  }, [location]);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography 
        variant="h4" 
        align="center" 
        gutterBottom
        sx={{ mb: 1, fontWeight: 500 }}
      >
        Select Emotion Detection Model
      </Typography>
      
      {userRole && (
        <Typography 
          variant="h6" 
          align="center" 
          color="text.secondary"
          sx={{ mb: 4 }}
        >
          Specialized for {userRole.charAt(0).toUpperCase() + userRole.slice(1)}s
        </Typography>
      )}
      
      <Grid container spacing={4} justifyContent="center">
        {models.map((model) => (
          <Grid item xs={12} md={userRole ? 8 : 6} key={model.id}>
            <Card 
              sx={{ 
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'scale(1.02)',
                  boxShadow: 6
                }
              }}
            >
              <CardMedia
                component="img"
                height="250"
                image={model.image}
                alt={model.title}
                sx={{ objectFit: 'cover' }}
              />
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Typography gutterBottom variant="h5" component="div" sx={{ fontWeight: 500 }}>
                  {model.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                  {model.description}
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  fullWidth
                  onClick={() => navigate(`/detection/${model.id}`)}
                  sx={{
                    mt: 2,
                    backgroundColor: '#2c3e50',
                    '&:hover': {
                      backgroundColor: '#34495e'
                    }
                  }}
                >
                  Select Model
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}