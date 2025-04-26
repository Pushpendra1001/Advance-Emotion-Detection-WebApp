

const startSession = async () => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/start-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        
        modelType: 'classroom'
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