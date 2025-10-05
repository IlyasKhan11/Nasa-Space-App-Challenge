import React, { useState } from 'react';
import Header from './components/Header';
import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import CSVUpload from './components/CSVUpload';
import { predictExoplanet } from './services/api';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictExoplanet(formData);
      setPrediction(result);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen space-bg stars relative overflow-hidden">
      {/* Animated background planets */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-10 w-4 h-4 bg-blue-400 rounded-full animate-pulse-slow opacity-60"></div>
        <div className="absolute top-40 right-20 w-6 h-6 bg-green-400 rounded-full animate-float opacity-40"></div>
        <div className="absolute bottom-40 left-1/4 w-3 h-3 bg-yellow-400 rounded-full animate-twinkle opacity-50"></div>
        <div className="absolute top-1/3 right-1/3 w-5 h-5 bg-red-400 rounded-full animate-drift opacity-30"></div>
        <div className="absolute bottom-20 right-10 w-2 h-2 bg-purple-400 rounded-full animate-pulse-slow opacity-70"></div>
      </div>
      
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 relative">
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-32 h-32 border border-nasa-blue/20 rounded-full animate-pulse-slow"></div>
              <div className="absolute w-20 h-20 border border-yellow-400/30 rounded-full animate-orbit"></div>
            </div>
            <h1 className="text-4xl md:text-6xl font-nasa font-bold text-white mb-4 text-shadow relative z-10">
              🪐 NASA Exoplanet Prediction System
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto relative z-10">
              Advanced AI-powered system for detecting and classifying exoplanets using NASA's Kepler data
            </p>
            <div className="flex justify-center space-x-2 mt-4">
              <span className="text-2xl animate-twinkle">⭐</span>
              <span className="text-2xl animate-twinkle" style={{animationDelay: '0.5s'}}>✨</span>
              <span className="text-2xl animate-twinkle" style={{animationDelay: '1s'}}>🌟</span>
              <span className="text-2xl animate-twinkle" style={{animationDelay: '1.5s'}}>⭐</span>
            </div>
          </div>

          <div className="space-y-8">
            <PredictionForm onSubmit={handlePrediction} onReset={handleReset} />
            
            {loading && <LoadingSpinner />}
            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}
            {prediction && <ResultsDisplay prediction={prediction} />}
            
            <CSVUpload />
            
            <div className="grid lg:grid-cols-2 gap-8">
              <div className="card">
                <h3 className="text-xl font-semibold mb-4 text-nasa-blue">About This System</h3>
                <div className="space-y-3 text-gray-300">
                  <p>
                    This system uses machine learning to analyze exoplanet data and classify objects into three categories:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li><span className="text-green-400 font-semibold">CONFIRMED:</span> Verified exoplanets</li>
                    <li><span className="text-yellow-400 font-semibold">CANDIDATE:</span> Potential exoplanets requiring further study</li>
                    <li><span className="text-red-400 font-semibold">FALSE POSITIVE:</span> Objects that are not exoplanets</li>
                  </ul>
                  <p className="text-sm text-gray-400 mt-4">
                    The model analyzes 15 different parameters including orbital period, planet radius, stellar temperature, and various quality flags.
                  </p>
                </div>
              </div>
              
              <div className="card">
                <h3 className="text-xl font-semibold mb-4 text-nasa-blue">Team AstroVenture</h3>
                <div className="space-y-3 text-gray-300">
                  <p>
                    Proudly developed by team AstroVenture for the NASA Space App Challenge.
                  </p>
                  
                  {/* Country Flags */}
                  <div className="flex items-center justify-center space-x-4 my-4">
                    <div className="text-center">
                      <div className="text-2xl mb-1">🇵🇰</div>
                      <p className="text-xs text-gray-400">Pakistan</p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl mb-1">🇦🇿</div>
                      <p className="text-xs text-gray-400">Azerbaijan</p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl mb-1">🇰🇿</div>
                      <p className="text-xs text-gray-400">Kazakhstan</p>
                    </div>
                  </div>
                  
                  <p>
                    Our mission is to advance exoplanet detection and classification using cutting-edge machine learning techniques.
                  </p>
                  <div className="bg-nasa-blue/20 rounded-lg p-4 mt-4">
                    <p className="text-sm text-nasa-blue font-semibold">
                      🚀 Exploring the cosmos, one prediction at a time
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
