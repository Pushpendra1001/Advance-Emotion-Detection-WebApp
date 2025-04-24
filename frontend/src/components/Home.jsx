import { 
  Container, 
  Typography, 
  Box, 
  Card, 
  CardContent, 
  CardMedia, 
  Button, 
  Grid 
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

const userTypes = [
  {
    id: 'doctor',
    title: 'Doctor',
    description: 'Analyze patient emotions during consultations',
    image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=500',
  },
  {
    id: 'teacher',
    title: 'Teacher',
    description: 'Monitor student engagement and emotional responses',
    image: 'https://images.unsplash.com/photo-1523580494863-6f3031224c94?auto=format&fit=crop&w=500',
  },
  {
    id: 'general',
    title: 'General User',
    description: 'Track and analyze emotions in various scenarios',
    image: 'https://images.unsplash.com/photo-1516387938699-a93567ec168e?auto=format&fit=crop&w=500',
  }
];

function Home() {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      <Box textAlign="center" mb={6}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          Emotion Detection Platform
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Advanced AI-powered emotion recognition for various professional environments
        </Typography>
      </Box>

      <Grid container spacing={4} justifyContent="center">
        {userTypes.map((userType) => (
          <Grid item xs={12} md={4} key={userType.id}>
            <Card 
              sx={{ 
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'scale(1.03)',
                  boxShadow: 6
                }
              }}
            >
              <CardMedia
                component="img"
                height="200"
                image={userType.image}
                alt={userType.title}
              />
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Typography gutterBottom variant="h5" component="h2">
                  {userType.title}
                </Typography>
                <Typography variant="body1" paragraph>
                  {userType.description}
                </Typography>
                <Button 
                  variant="contained" 
                  size="large"
                  onClick={() => navigate(`/select-role/${userType.id}`)}
                  sx={{ 
                    mt: 2,
                    backgroundColor: '#2c3e50',
                    '&:hover': { backgroundColor: '#34495e' }
                  }}
                >
                  Get Started
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default Home;