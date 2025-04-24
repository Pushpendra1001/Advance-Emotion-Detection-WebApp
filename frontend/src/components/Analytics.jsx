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
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import VisibilityIcon from '@mui/icons-material/Visibility';
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
    emotionTrends: [] // Add this default value
  });
  const [timeRange, setTimeRange] = useState('day');
  const [selectedEmotion, setSelectedEmotion] = useState('all');
  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/analytics`, {
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
          'Access-Control-Allow-Credentials': 'true'
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch analytics data');
      }
      
      const rawData = await response.json();
      const processedData = processAnalyticsData(rawData);
      setData(processedData);
    } catch (err) {
      setError(err.message);
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  const processAnalyticsData = (rawData) => {
    // Calculate emotion percentages
    const totalEmotions = rawData.emotionsByModel.reduce((acc, curr) => acc + curr.count, 0);
    const emotionsWithPercentage = rawData.emotionsByModel.map(item => ({
      ...item,
      percentage: ((item.count / totalEmotions) * 100).toFixed(2)
    }));

    // Calculate emotion trends
    const emotionTrends = calculateEmotionTrends(rawData.emotionsByTime);

    return {
      ...rawData,
      emotionsByModel: emotionsWithPercentage,
      emotionTrends
    };
  };

  const calculateEmotionTrends = (timeData) => {
    const trends = {};
    timeData.forEach(entry => {
      const hour = new Date(entry.timestamp).getHours();
      if (!trends[hour]) {
        trends[hour] = { hour, count: 0 };
      }
      trends[hour].count += entry.count;
    });
    return Object.values(trends);
  };

  const EmotionMetricsCard = ({ title, value, subtitle }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        <Typography variant="h4" color="primary">{value}</Typography>
        <Typography variant="body2" color="text.secondary">{subtitle}</Typography>
      </CardContent>
    </Card>
  );

  const SessionHistoryTable = ({ sessions, onViewReport }) => {
    return (
      <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
        <Typography variant="h6" gutterBottom>Session History</Typography>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Start Time</TableCell>
              <TableCell>Duration (min)</TableCell>
              <TableCell>Model Type</TableCell>
              <TableCell>Dominant Emotion</TableCell>
              <TableCell>Total Detections</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sessions.map((session) => (
              <TableRow key={session.sessionId}>
                <TableCell>{new Date(session.startTime).toLocaleString()}</TableCell>
                <TableCell>{session.duration.toFixed(2)}</TableCell>
                <TableCell>{session.modelType}</TableCell>
                <TableCell>{session.dominantEmotion}</TableCell>
                <TableCell>{session.totalDetections}</TableCell>
                <TableCell>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => onViewReport(session)}
                    startIcon={<VisibilityIcon />}
                  >
                    View Report
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    );
  };

  const SessionReport = ({ session, onClose }) => {
    if (!session) return null;
  
    const emotionData = Object.entries(session.emotionBreakdown).map(([emotion, count]) => ({
      emotion,
      count,
      percentage: session.emotionPercentages[emotion]
    }));
  
    return (
      <Dialog 
        open={true} 
        onClose={onClose}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Session Analysis Report
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
                    <Typography><strong>Start Time:</strong> {new Date(session.startTime).toLocaleString()}</Typography>
                    <Typography><strong>End Time:</strong> {new Date(session.endTime).toLocaleString()}</Typography>
                    <Typography><strong>Duration:</strong> {session.duration.toFixed(2)} minutes</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography><strong>Model Type:</strong> {session.modelType}</Typography>
                    <Typography><strong>Total Detections:</strong> {session.totalDetections}</Typography>
                    <Typography><strong>Dominant Emotion:</strong> {session.dominantEmotion}</Typography>
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
                      label={({ emotion, percentage }) => `${emotion} (${percentage.toFixed(1)}%)`}
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
  
            {/* Emotion Timeline */}
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

  if (!data.emotionsByModel.length && !data.sessionHistory.length) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography align="center">
          No analytics data available yet. Start an emotion detection session to generate data.
        </Typography>
      </Container>
    );
  }

  const getEmotionSummaryData = () => {
    const totalDetections = data.emotionsByModel.reduce((acc, curr) => acc + curr.count, 0) || 0;
    const dominantEmotion = data.emotionsByModel.length > 0 
        ? [...data.emotionsByModel].sort((a, b) => b.count - a.count)[0]
        : null;
    const activeSessions = data.sessionHistory?.length || 0;
    return { totalDetections, dominantEmotion, activeSessions };
  };

  const { totalDetections, dominantEmotion } = getEmotionSummaryData();

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" align="center" gutterBottom>
        Emotion Analytics Dashboard
      </Typography>

     
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <EmotionMetricsCard
            title="Total Detections"
            value={getEmotionSummaryData().totalDetections}
            subtitle="Total emotion detections"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <EmotionMetricsCard
            title="Dominant Emotion"
            value={getEmotionSummaryData().dominantEmotion?.emotion || 'None'}
            subtitle={`${getEmotionSummaryData().dominantEmotion?.percentage || 0}% of total detections`}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <EmotionMetricsCard
            title="Active Sessions"
            value={getEmotionSummaryData().activeSessions}
            subtitle="Number of detection sessions"
          />
        </Grid>
      </Grid>

      <Grid container spacing={4}>
        <Grid item xs={12} lg={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Emotion Trends Over Time</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={data.emotionsByTime}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }}
                />
              </LineChart>
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
            <Typography variant="h6" gutterBottom>Emotion Radar Analysis</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart cx="50%" cy="50%" outerRadius="80%">
                <PolarGrid />
                <PolarAngleAxis dataKey="emotion" />
                <PolarRadiusAxis />
                <Radar
                  name="Emotions"
                  dataKey="count"
                  data={data.emotionsByModel}
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

      
      </Grid>

      {data.sessionHistory && data.sessionHistory.length > 0 && (
        <SessionHistoryTable 
          sessions={data.sessionHistory} 
          onViewReport={(session) => setSelectedSession(session)}
        />
      )}

      {selectedSession && (
        <SessionReport
          session={selectedSession}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </Container>
  );
};

export default Analytics;
