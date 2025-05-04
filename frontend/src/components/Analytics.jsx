import { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  CircularProgress,
  Box,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Chip,
  Tabs,
  Tab,
  TextField,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DownloadIcon from '@mui/icons-material/Download';
import PersonIcon from '@mui/icons-material/Person';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import DateRangeIcon from '@mui/icons-material/DateRange';
import AssignmentIcon from '@mui/icons-material/Assignment';
import InsightsIcon from '@mui/icons-material/Insights';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import SchoolIcon from '@mui/icons-material/School';
import GroupIcon from '@mui/icons-material/Group';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const PYTHON_API_URL = import.meta.env.VITE_PYTHON_API_URL;

const Analytics = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState({
    emotionsByTime: [],
    emotionsByModel: [],
    sessionHistory: [],
    emotionTrends: [],
    patientStats: [], 
    dailySessionCounts: [] 
  });
  const [timeRange, setTimeRange] = useState('week');
  const [selectedPatient, setSelectedPatient] = useState('all');
  const [selectedSession, setSelectedSession] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [patients, setPatients] = useState([]);
  const [userRole, setUserRole] = useState('all'); 

  useEffect(() => {
    fetchAnalytics();

    const intervalId = setInterval(() => {
      fetchAnalytics();
    }, 30000); 

    return () => clearInterval(intervalId);
  }, [timeRange, selectedPatient, userRole]); 

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  
  const handleRoleChange = (event, newRole) => {
    if (newRole !== null) {
      setUserRole(newRole);
      
      setSelectedPatient('all');
    }
  };

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      
      const url = new URL(`${PYTHON_API_URL}/analytics`);
      url.searchParams.append('time_range', timeRange);
      
      
      if (selectedPatient !== 'all') {
        url.searchParams.append('patient_name', selectedPatient);
      }
      
      
      if (userRole !== 'all') {
        url.searchParams.append('role', userRole);
      }
      
      const response = await fetch(url, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        if (response.status === 401) {
          window.location.href = '/auth'; 
          return;
        }
        throw new Error(await response.text() || 'Failed to fetch analytics data');
      }
      
      const rawData = await response.json();
      console.log('Raw analytics data:', rawData);
      
      
      let filteredPatients = [];
      if (rawData.sessionHistory && rawData.sessionHistory.length > 0) {
        filteredPatients = [...new Set(rawData.sessionHistory
          .filter(session => {
            if (userRole === 'all') return true;
            return session.modelType.includes(userRole);
          })
          .map(s => s.patientName))];
      }
      setPatients(filteredPatients);
      
      const processedData = processAnalyticsData(rawData, userRole);
      setData(processedData);
    } catch (err) {
      console.error('Analytics error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  
  const processAnalyticsData = (rawData, role) => {
    if (!rawData || !rawData.emotionsByModel) {
      return {
        emotionsByTime: [],
        emotionsByModel: [],
        sessionHistory: [],
        emotionTrends: [],
        patientStats: [],
        dailySessionCounts: []
      };
    }

    
    const filteredSessions = role !== 'all'
      ? (rawData.sessionHistory || []).filter(s => s.modelType.includes(role))
      : rawData.sessionHistory || [];

    
    const totalEmotions = rawData.emotionsByModel.reduce((acc, curr) => acc + curr.count, 0) || 0;
    const emotionsWithPercentage = rawData.emotionsByModel.map(item => ({
      ...item,
      percentage: totalEmotions > 0 ? ((item.count / totalEmotions) * 100).toFixed(2) : 0
    }));

    
    const emotionTrends = calculateEmotionTrends(rawData.emotionsByTime || []);
    
    
    const patientStats = processPatientStats(filteredSessions);
    
    
    const dailySessionCounts = calculateDailySessions(filteredSessions);

    return {
      emotionsByTime: rawData.emotionsByTime || [],
      emotionsByModel: emotionsWithPercentage,
      emotionTrends,
      patientStats,
      dailySessionCounts,
      sessionHistory: filteredSessions
    };
  };

  const processPatientStats = (sessions) => {
    if (!sessions || sessions.length === 0) return [];
    
    const patientMap = {};
    
    
    sessions.forEach(session => {
      const patientName = session.patientName || 'Unknown';
      if (!patientMap[patientName]) {
        patientMap[patientName] = {
          patientName,
          sessionCount: 0,
          totalDuration: 0,
          dominantEmotions: {},
          lastSession: null
        };
      }
      
      
      const patient = patientMap[patientName];
      patient.sessionCount++;
      patient.totalDuration += session.duration || 0;
      
      
      if (session.dominantEmotion) {
        patient.dominantEmotions[session.dominantEmotion] = 
          (patient.dominantEmotions[session.dominantEmotion] || 0) + 1;
      }
      
      
      const sessionDate = new Date(session.startTime);
      if (!patient.lastSession || sessionDate > new Date(patient.lastSession)) {
        patient.lastSession = session.startTime;
      }
    });
    
    
    return Object.values(patientMap).map(patient => {
      const dominantEmotionsEntries = Object.entries(patient.dominantEmotions);
      const mostCommonEmotion = dominantEmotionsEntries.length > 0 
        ? dominantEmotionsEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0] 
        : 'Unknown';
        
      return {
        ...patient,
        avgDuration: patient.sessionCount > 0 ? (patient.totalDuration / patient.sessionCount).toFixed(2) : 0,
        mostCommonEmotion
      };
    });
  };

  const calculateEmotionTrends = (timeData) => {
    if (!Array.isArray(timeData) || timeData.length === 0) {
      return [];
    }

    const trends = {};
    timeData.forEach(entry => {
      if (entry && entry.timestamp) {
        const hour = new Date(entry.timestamp).getHours();
        if (!trends[hour]) {
          trends[hour] = { hour, count: 0 };
        }
        trends[hour].count += entry.count || 0;
      }
    });
    return Object.values(trends);
  };

  const calculateDailySessions = (sessions) => {
    if (!sessions || sessions.length === 0) return [];
    
    const dailyMap = {};
    
    sessions.forEach(session => {
      if (session.startTime) {
        const date = new Date(session.startTime).toLocaleDateString();
        if (!dailyMap[date]) {
          dailyMap[date] = { 
            date, 
            count: 0, 
            totalDetections: 0,
            uniquePatients: new Set()
          };
        }
        
        dailyMap[date].count++;
        dailyMap[date].totalDetections += session.totalDetections || 0;
        dailyMap[date].uniquePatients.add(session.patientName || 'Unknown');
      }
    });
    
    
    return Object.values(dailyMap).map(day => ({
      ...day,
      uniquePatientCount: day.uniquePatients.size,
      uniquePatients: Array.from(day.uniquePatients)
    }));
  };

  const downloadSessionData = async (sessionId) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/download-session-data?session_id=${sessionId}`, {
        method: 'GET',
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error('Failed to download session data');
      }
      
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = `session_${sessionId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(downloadUrl);
      document.body.removeChild(a);
      
    } catch (err) {
      console.error('Error downloading session data:', err);
      setError('Failed to download session data: ' + err.message);
    }
  };

  const EmotionMetricsCard = ({ title, value, subtitle, icon }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {icon && <Box sx={{ mr: 1, color: 'primary.main' }}>{icon}</Box>}
          <Typography variant="h6">{title}</Typography>
        </Box>
        <Typography variant="h4" color="primary">{value}</Typography>
        <Typography variant="body2" color="text.secondary">{subtitle}</Typography>
      </CardContent>
    </Card>
  );

  const SessionHistoryTable = ({ sessions, onViewReport, userRole }) => {
    
    const subjectLabel = userRole === 'doctor' ? 'Patient Name' : 
                        userRole === 'teacher' ? 'Class Name' : 'Subject Name';

    return (
      <Paper elevation={3} sx={{ p: 3, mt: 4, overflowX: 'auto' }}>
        <Typography variant="h6" gutterBottom>
          {userRole === 'doctor' ? 'Consultation History' : 
           userRole === 'teacher' ? 'Lesson History' : 'Session History'}
        </Typography>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>{subjectLabel}</TableCell>
              <TableCell>Start Time</TableCell>
              <TableCell>Duration (min)</TableCell>
              <TableCell>Model Type</TableCell>
              <TableCell>Dominant Emotion</TableCell>
              <TableCell>Total Detections</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sessions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">No session data available</TableCell>
              </TableRow>
            ) : (
              sessions.map((session) => (
                <TableRow key={session.sessionId}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <PersonIcon sx={{ mr: 1, fontSize: 16, color: 'primary.main' }} />
                      {session.patientName || 'Unknown'}
                    </Box>
                  </TableCell>
                  <TableCell>{new Date(session.startTime).toLocaleString()}</TableCell>
                  <TableCell>{session.duration.toFixed(2)}</TableCell>
                  <TableCell>
                    <Chip size="small" label={session.modelType} color="primary" variant="outlined" />
                  </TableCell>
                  <TableCell>
                    <Chip 
                      size="small" 
                      label={session.dominantEmotion || 'Unknown'} 
                      color={getEmotionColor(session.dominantEmotion)} 
                    />
                  </TableCell>
                  <TableCell>{session.totalDetections}</TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => onViewReport(session)}
                        startIcon={<VisibilityIcon />}
                      >
                        View
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => downloadSessionData(session.sessionId)}
                        startIcon={<DownloadIcon />}
                      >
                        CSV
                      </Button>
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Paper>
    );
  };

  const getEmotionColor = (emotion) => {
    if (!emotion) return 'default';
    const emotionMap = {
      'happy': 'success',
      'neutral': 'info',
      'angry': 'error',
      'sad': 'warning',
      'disgust': 'error',
      'fear': 'warning',
      'surprise': 'secondary'
    };
    return emotionMap[emotion.toLowerCase()] || 'default';
  };

  const PatientStatsTable = ({ patients }) => {
    return (
      <Paper elevation={3} sx={{ p: 3, mt: 4, overflowX: 'auto' }}>
        <Typography variant="h6" gutterBottom>Patient Statistics</Typography>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Patient Name</TableCell>
              <TableCell>Sessions</TableCell>
              <TableCell>Average Duration (min)</TableCell>
              <TableCell>Most Common Emotion</TableCell>
              <TableCell>Last Session</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {patients.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">No patient data available</TableCell>
              </TableRow>
            ) : (
              patients.map((patient) => (
                <TableRow key={patient.patientName}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <PersonIcon sx={{ mr: 1, fontSize: 16, color: 'primary.main' }} />
                      {patient.patientName}
                    </Box>
                  </TableCell>
                  <TableCell>{patient.sessionCount}</TableCell>
                  <TableCell>{patient.avgDuration}</TableCell>
                  <TableCell>
                    <Chip 
                      size="small" 
                      label={patient.mostCommonEmotion} 
                      color={getEmotionColor(patient.mostCommonEmotion)} 
                    />
                  </TableCell>
                  <TableCell>{patient.lastSession ? new Date(patient.lastSession).toLocaleDateString() : 'N/A'}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Paper>
    );
  };

  const SessionReport = ({ session, onClose }) => {
    if (!session) return null;
  
    const emotionData = Object.entries(session.emotionBreakdown || {}).map(([emotion, count]) => ({
      emotion,
      count,
      percentage: session.emotionPercentages ? session.emotionPercentages[emotion] : 0
    }));
  
    return (
      <Dialog 
        open={true} 
        onClose={onClose}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Session Analysis Report - {session.patientName || 'Unknown Patient'}
          <IconButton
            onClick={onClose}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            {/* Session Details */}
            <Grid item xs={12}>
              <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                <Typography variant="h6" gutterBottom>Session Details</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography><strong>Patient Name:</strong> {session.patientName || 'Unknown'}</Typography>
                    <Typography><strong>Start Time:</strong> {new Date(session.startTime).toLocaleString()}</Typography>
                    <Typography><strong>End Time:</strong> {new Date(session.endTime).toLocaleString()}</Typography>
                    <Typography><strong>Duration:</strong> {session.duration.toFixed(2)} minutes</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography><strong>Model Type:</strong> {session.modelType}</Typography>
                    <Typography><strong>Total Detections:</strong> {session.totalDetections}</Typography>
                    <Typography><strong>Dominant Emotion:</strong> {session.dominantEmotion || 'Unknown'}</Typography>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => downloadSessionData(session.sessionId)}
                      startIcon={<DownloadIcon />}
                      sx={{ mt: 1 }}
                    >
                      Download CSV Data
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
  
            {/* Emotion Distribution Pie Chart */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Emotion Distribution</Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={emotionData}
                      dataKey="count"
                      nameKey="emotion"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ emotion, percentage }) => `${emotion} (${percentage?.toFixed(1) || 0}%)`}
                    >
                      {emotionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
  
            {/* Emotion Radar Chart */}
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Emotion Intensity</Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart outerRadius={90}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="emotion" />
                    <PolarRadiusAxis />
                    <Radar
                      name="Emotion Count"
                      dataKey="count"
                      data={emotionData}
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

           
          </Grid>
        </DialogContent>
      </Dialog>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography color="error" align="center">
          Error loading analytics: {error}
        </Typography>
      </Container>
    );
  }

  const getEmotionSummaryData = () => {
    const totalDetections = data.emotionsByModel.reduce((acc, curr) => acc + curr.count, 0) || 0;
    const dominantEmotion = data.emotionsByModel.length > 0 
        ? [...data.emotionsByModel].sort((a, b) => b.count - a.count)[0]
        : null;
    const totalPatients = data.patientStats?.length || 0;
    const totalSessions = data.sessionHistory?.length || 0;
    return { totalDetections, dominantEmotion, totalPatients, totalSessions };
  };

  
  const getDashboardTitle = () => {
    switch(userRole) {
      case 'doctor':
        return "Doctor's Analytics Dashboard";
      case 'teacher':
        return "Teacher's Analytics Dashboard";
      default:
        return "Emotion Analytics Dashboard";
    }
  };
  
  
  const getPatientLabel = () => {
    if (userRole === 'doctor') {
      return "Patient";
    } else if (userRole === 'teacher') {
      return "Class";
    }
    return "Session Subject";
  };

  const { totalDetections, dominantEmotion, totalPatients, totalSessions } = getEmotionSummaryData();

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" align="center" gutterBottom>
        <InsightsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        {getDashboardTitle()}
      </Typography>

      {/* Role selector toggle */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
        <ToggleButtonGroup
          color="primary"
          value={userRole}
          exclusive
          onChange={handleRoleChange}
          aria-label="Role selection"
        >
          <ToggleButton value="all">
            <GroupIcon sx={{ mr: 1 }} />
            All Users
          </ToggleButton>
          <ToggleButton value="doctor">
            <LocalHospitalIcon sx={{ mr: 1 }} />
            Doctors
          </ToggleButton>
          <ToggleButton value="teacher">
            <SchoolIcon sx={{ mr: 1 }} />
            Teachers
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel id="time-range-label">Time Range</InputLabel>
          <Select
            labelId="time-range-label"
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="day">Last 24 Hours</MenuItem>
            <MenuItem value="week">Last Week</MenuItem>
            <MenuItem value="month">Last Month</MenuItem>
            <MenuItem value="all">All Time</MenuItem>
          </Select>
        </FormControl>
        
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel id="patient-label">{getPatientLabel()}</InputLabel>
          <Select
            labelId="patient-label"
            value={selectedPatient}
            label={getPatientLabel()}
            onChange={(e) => setSelectedPatient(e.target.value)}
          >
            <MenuItem value="all">All {userRole === 'doctor' ? 'Patients' : userRole === 'teacher' ? 'Classes' : 'Subjects'}</MenuItem>
            {patients.map(patient => (
              <MenuItem key={patient} value={patient}>{patient}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button 
          variant="contained" 
          onClick={fetchAnalytics}
          sx={{ ml: 'auto' }}
        >
          Refresh Data
        </Button>
      </Paper>

      {/* Summary Cards - Modified based on role */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <EmotionMetricsCard
            title={userRole === 'doctor' ? "Total Patients" : userRole === 'teacher' ? "Total Classes" : "Total Subjects"}
            value={totalPatients}
            subtitle={userRole === 'doctor' ? "Number of unique patients" : userRole === 'teacher' ? "Number of unique classes" : "Number of unique subjects"}
            icon={<PersonIcon />}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <EmotionMetricsCard
            title="Total Sessions"
            value={totalSessions}
            subtitle={userRole === 'doctor' ? "Number of consultations" : userRole === 'teacher' ? "Number of lessons" : "Number of sessions"}
            icon={<AssignmentIcon />}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <EmotionMetricsCard
            title="Total Detections"
            value={totalDetections}
            subtitle="Emotion data points collected"
            icon={<InsightsIcon />}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <EmotionMetricsCard
            title="Dominant Emotion"
            value={dominantEmotion?.emotion || 'None'}
            subtitle={`${dominantEmotion?.percentage || 0}% of detections`}
            icon={<InsightsIcon />}
          />
        </Grid>
      </Grid>

      {/* Tabs */}
      <Tabs 
        value={tabValue} 
        onChange={handleTabChange} 
        aria-label="analytics tabs"
        sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab icon={<InsightsIcon />} label="Overview" />
        <Tab icon={<PersonIcon />} label="Patient Analysis" />
        <Tab icon={<CalendarTodayIcon />} label="Session History" />
      </Tabs>

      {/* Tab 1: Overview */}
      {tabValue === 0 && (
        <Grid container spacing={4}>
          <Grid item xs={12} lg={8}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Daily Session Statistics</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={data.dailySessionCounts}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="count" name="Sessions" fill="#8884d8" />
                  <Bar yAxisId="right" dataKey="uniquePatientCount" name="Unique Patients" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          <Grid item xs={12} lg={4}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Emotion Distribution</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <PieChart>
                  <Pie
                    data={data.emotionsByModel}
                    dataKey="count"
                    nameKey="emotion"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {data.emotionsByModel.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Emotion Trends Over Time</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data.emotionsByTime || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(timestamp) => {
                      if (!timestamp) return '';
                      try {
                        const date = new Date(timestamp);
                        return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                      } catch (e) {
                        console.error('Error formatting timestamp:', e);
                        return '';
                      }
                    }} 
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(timestamp) => {
                      if (!timestamp) return '';
                      try {
                        return new Date(timestamp).toLocaleString();
                      } catch (e) {
                        return timestamp;
                      }
                    }} 
                  />
                  <Legend />
                  {data.emotionsByTime.length > 0 && data.emotionsByTime[0].emotions && (
                    <>
                      <Line 
                        type="monotone" 
                        dataKey="emotions.happy" 
                        stroke="#00C49F" 
                        name="Happy" 
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 6 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="emotions.neutral" 
                        stroke="#0088FE" 
                        name="Neutral"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 6 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="emotions.sad" 
                        stroke="#FFBB28" 
                        name="Sad"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 6 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="emotions.angry" 
                        stroke="#FF8042" 
                        name="Angry"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 6 }}
                      />
                    </>
                  )}
                  {/* Keep the total count line as fallback */}
                  {(!data.emotionsByTime.length || !data.emotionsByTime[0].emotions) && (
                    <Line 
                      type="monotone" 
                      dataKey="count" 
                      stroke="#8884d8" 
                      activeDot={{ r: 8 }}
                      name="Total Detections"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Tab 2: Patient Analysis */}
      {tabValue === 1 && (
        <>
          <PatientStatsTable patients={data.patientStats} />
          
          <Grid container spacing={4} sx={{ mt: 2 }}>
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Patient Session Distribution</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={data.patientStats}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="patientName" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="sessionCount" name="Number of Sessions" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Average Session Duration by Patient</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={data.patientStats}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="patientName" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="avgDuration" name="Average Duration (min)" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
          </Grid>
        </>
      )}

      {/* Tab 3: Session History */}
      {tabValue === 2 && (
        <SessionHistoryTable 
          sessions={data.sessionHistory} 
          onViewReport={(session) => setSelectedSession(session)}
          userRole={userRole}
        />
      )}

      {selectedSession && (
        <SessionReport
          session={selectedSession}
          onClose={() => setSelectedSession(null)}
          userRole={userRole}
        />
      )}
    </Container>
  );
};

export default Analytics;