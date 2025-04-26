// import { useState, useEffect } from 'react';
// import { useLocation, useNavigate } from 'react-router-dom';
// import { Container, Typography, Paper, Button, Box } from '@mui/material';
// import ArrowBackIcon from '@mui/icons-material/ArrowBack';
// import ImageAnalyzer from './ImageAnalyzer';

// const ImageAnalysis = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const [userRole, setUserRole] = useState('general');
  
//   useEffect(() => {
//     // Get user role from location state
//     if (location.state?.userRole) {
//       setUserRole(location.state.userRole);
//     }
//   }, [location]);
  
//   const handleBack = () => {
//     navigate(-1); // Go back to previous page
//   };
  
//   return (
//     <Container maxWidth="lg" sx={{ py: 4 }}>
//       <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
//         <Button 
//           startIcon={<ArrowBackIcon />} 
//           onClick={handleBack}
//           variant="outlined"
//         >
//           Back
//         </Button>
        
//         <Typography variant="h4" sx={{ fontWeight: 500 }}>
//           Image Emotion Analysis
//         </Typography>
        
//         <Box sx={{ width: 100 }}></Box> {/* Empty box for alignment */}
//       </Box>
      
//       <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
//         <Typography variant="h6" align="center" gutterBottom>
//           {userRole === 'doctor' 
//             ? 'Upload patient photos to analyze their emotional states' 
//             : userRole === 'teacher' 
//               ? 'Upload classroom photos to analyze student emotions'
//               : 'Upload any photo to detect and analyze emotions'}
//         </Typography>
        
//         <ImageAnalyzer userRole={userRole} />
//       </Paper>
//     </Container>
//   );
// };

// export default ImageAnalysis;