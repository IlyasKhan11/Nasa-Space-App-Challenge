# NASA Exoplanet Prediction Frontend

A modern React-based frontend for the NASA Exoplanet Prediction System.

## Features

- 🚀 Modern, responsive UI with NASA-inspired design
- 📊 Real-time exoplanet classification
- ✅ Comprehensive form validation
- 🎨 Beautiful animations and transitions
- 📱 Mobile-friendly design
- 🔧 Easy integration with FastAPI backend

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Make sure your FastAPI backend is running on `http://localhost:8000`

## Environment Variables

Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:8000
```

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

## Backend Integration

This frontend is designed to work with the FastAPI backend in the parent directory. The backend should be running on port 8000 with the following endpoints:

- `POST /api/predict` - Make exoplanet predictions
- `GET /health` - Health check

## Technologies Used

- React 18
- Tailwind CSS
- Lucide React (icons)
- Axios (HTTP client)
- Create React App
