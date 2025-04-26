const saveEmotionData = async (data) => {
  try {
    const response = await fetch('/api/emotions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    return await response.json();
  } catch (error) {
    console.error('Error saving emotion data:', error);
    throw error;
  }
};

const getSessionReport = async (sessionId) => {
  try {
    const response = await fetch(`/api/sessions/${sessionId}`);
    return await response.json();
  } catch (error) {
    console.error('Error getting session report:', error);
    throw error;
  }
};

const downloadSessionData = async (sessionId) => {
  try {
    const response = await fetch(`/api/sessions/${sessionId}/download`);
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session-${sessionId}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading session data:', error);
    throw error;
  }
};

export { saveEmotionData, getSessionReport, downloadSessionData };