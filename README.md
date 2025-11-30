
# Advance Emotion Detection WebApp
Real-time emotion detection and analytics for Counselor , Therapist , Psychologist sessions. Built with React, Node.js, MongoDB, and Python (Flask) using deep learning to analyze emotions from webcam video and visualize engagement.

## Features
- **Doctor/Patient Module:** Track and visualize patient emotions during sessions to help doctors understand patient feelings and reactions.
- **Teacher/Student Module:** Monitor student emotions and activeness during classes to help teachers assess engagement.
- **Real-time Emotion Detection:** Uses webcam and deep learning models for live emotion recognition.
- **Analytics Dashboard:** Visualize emotion trends and session data.
- **User Authentication:** Secure login and registration for different user roles.

## Project Structure
- `backend/` — Node.js/Express API server (auth, MongoDB, session management)
- `frontend/` — React app (UI, dashboard, video, analytics) and Python Flask server (emotion detection)

## Prerequisites
- Node.js (v16+ recommended)
- Python 3.8+
- MongoDB (cloud or local)

## Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Advance-Emotion-Detection-WebApp-main
```

### 2. Install Dependencies

#### Backend (Node.js)
```bash
cd backend
npm install
```

#### Frontend (React & Python)
```bash
cd ../frontend
npm install
pip install -r requirements.txt
```

### 3. Configure Environment Variables

- Edit `backend/.env` for MongoDB URI, JWT secret, email, etc.
- Edit `frontend/.env` for API URLs and model paths if needed.

### 4. Start the Servers

#### Start Node.js Backend (Port 5000 by default)
```bash
cd backend
npm start
```

#### Start Python Backend (Flask, Port 5005 by default)
```bash
cd ../frontend
python app.py
```

#### Start React Frontend (Vite, Port 5173 by default)
```bash
cd frontend
npm run dev
```

### 5. Access the Application

- Open your browser and go to [http://localhost:5173](http://localhost:5173)

## Usage
- Register or log in as a doctor, patient, teacher, or student.
- Start a session to record and analyze emotions in real time.
- View analytics and emotion trends on the dashboard.

## Notes
- Ensure all three servers (Node.js, Python, React) are running for full functionality.
- The Python backend handles real-time emotion detection and must be running for webcam features.
- MongoDB must be accessible as configured in `backend/.env`.


