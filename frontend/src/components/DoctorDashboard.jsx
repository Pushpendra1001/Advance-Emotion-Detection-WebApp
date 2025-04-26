

const startSession = async () => {
  if (!patientName) {
    setError('Please enter a patient name');
    return;
  }
  
  try {
    setError(null);
    const response = await fetch(`${PYTHON_API_URL}/start-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patientName: patientName, 
        modelType: selectedModel
      }),
      credentials: 'include',
    });
    
    const data = await response.json();
    
    if (data.success) {
      setSessionId(data.sessionId);
      setSessionActive(true);
      setVideoUrl(`${PYTHON_API_URL}/video_feed?session_id=${data.sessionId}`);
    } else {
      setError(data.error || 'Failed to start session');
    }
  } catch (error) {
    console.error('Error starting session:', error);
    setError('Network error. Please try again.');
  }
};