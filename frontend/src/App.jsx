import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate, useParams } from 'react-router-dom'
import { Box, AppBar, Toolbar, Typography, IconButton, Button } from '@mui/material'
import VideocamIcon from '@mui/icons-material/Videocam'
import BarChartIcon from '@mui/icons-material/BarChart'
import HomeIcon from '@mui/icons-material/Home'
import LogoutIcon from '@mui/icons-material/Logout'
import Home from './components/Home'
import Auth from './components/Auth'
import ModelSelection from './components/ModelSelection'
import EmotionDetection from './components/EmotionDetection'
import Analytics from './components/Analytics'
import ImageAnalysis from './components/ImageAnalyzer'
import './App.css'
import { AuthProvider, useAuth } from './context/AuthContext';


const ProtectedRoute = ({ children }) => {
  const { user } = useAuth();
  if (!user) {
    return <Navigate to="/auth" replace />;
  }
  return children;
};

function MainApp() {
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <BrowserRouter>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static" sx={{ backgroundColor: '#2c3e50' }}>
          <Toolbar sx={{ minHeight: '64px' }}>
            <IconButton 
              component={Link} 
              to="/"
              edge="start" 
              color="inherit" 
              sx={{ mr: 1 }}
            >
              <VideocamIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Emotion Detector
            </Typography>
            
            <Button 
              component={Link}
              to="/"
              color="inherit" 
              startIcon={<HomeIcon />}
              sx={{ mx: 1 }}
            >
              Home
            </Button>
            
            {user && (
              <>
                <Button 
                  component={Link}
                  to="/analytics"
                  color="inherit" 
                  startIcon={<BarChartIcon />}
                  sx={{ mx: 1 }}
                >
                  Analytics
                </Button>
                <Typography variant="body1" sx={{ mx: 1 }}>
                  {user.email}
                </Typography>
                <Button 
                  color="inherit" 
                  onClick={handleLogout}
                  startIcon={<LogoutIcon />}
                  sx={{ ml: 1 }}
                >
                  Logout
                </Button>
              </>
            )}
            
            {!user && (
              <Button 
                component={Link}
                to="/auth"
                color="inherit" 
                sx={{ ml: 1 }}
              >
                Login
              </Button>
            )}
          </Toolbar>
        </AppBar>

        <Box sx={{ flex: 1, p: 0 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/select-role/:roleId" element={<RoleRedirect />} />
            <Route path="/model-selection" element={
              <ProtectedRoute>
                <ModelSelection />
              </ProtectedRoute>
            } />
            <Route path="/detection/:modelId" element={
              <ProtectedRoute>
                <EmotionDetection />
              </ProtectedRoute>
            } />
              <Route path="/image-analysis" element={
              <ProtectedRoute>
                <ImageAnalysis />
              </ProtectedRoute>
              } />
            <Route path="/analytics" element={
              <ProtectedRoute>
                <Analytics />
              </ProtectedRoute>
            } />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Box>
      </Box>
    </BrowserRouter>
  );
}


function RoleRedirect() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const { roleId } = useParams();
  
  useEffect(() => {
    if (user) {
      
      navigate('/model-selection', { state: { role: roleId } });
    } else {
      
      navigate('/auth', { state: { redirectTo: '/model-selection', role: roleId } });
    }
  }, [user, roleId, navigate]);
  
  return <Box sx={{ p: 4, textAlign: 'center' }}>Redirecting...</Box>;
}

function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}

export default App;