import React from 'react';
import { CheckCircle, AlertCircle, XCircle, BarChart3, Info } from 'lucide-react';

const ResultsDisplay = ({ prediction }) => {
  const getPredictionStyle = (prediction) => {
    switch (prediction) {
      case 'CONFIRMED':
        return {
          icon: CheckCircle,
          color: 'text-green-400',
          bgColor: 'prediction-confirmed',
          title: 'Confirmed Exoplanet',
          description: 'This object has been verified as a genuine exoplanet through multiple observations and analysis.'
        };
      case 'CANDIDATE':
        return {
          icon: AlertCircle,
          color: 'text-yellow-400',
          bgColor: 'prediction-candidate',
          title: 'Exoplanet Candidate',
          description: 'This object shows promising characteristics of an exoplanet but requires further study for confirmation.'
        };
      case 'FALSE POSITIVE':
        return {
          icon: XCircle,
          color: 'text-red-400',
          bgColor: 'prediction-false-positive',
          title: 'False Positive',
          description: 'This object is not an exoplanet. It may be a stellar binary, instrumental artifact, or other non-planetary phenomenon.'
        };
      default:
        return {
          icon: AlertCircle,
          color: 'text-gray-400',
          bgColor: '',
          title: 'Unknown',
          description: 'Unable to determine classification.'
        };
    }
  };

  const predictionStyle = getPredictionStyle(prediction.prediction);
  const IconComponent = predictionStyle.icon;

  const getConfidenceLevel = (prediction) => {
    if (prediction.prediction === 'CONFIRMED') return 'High';
    if (prediction.prediction === 'CANDIDATE') return 'Medium';
    if (prediction.prediction === 'FALSE POSITIVE') return 'High';
    return 'Medium';
  };

  const getNextSteps = (prediction) => {
    switch (prediction) {
      case 'CONFIRMED':
        return [
          'Continue monitoring for additional transits',
          'Analyze atmospheric composition if possible',
          'Study orbital dynamics and stability',
          'Compare with known exoplanet populations'
        ];
      case 'CANDIDATE':
        return [
          'Schedule follow-up observations',
          'Analyze stellar activity and variability',
          'Check for stellar companions',
          'Perform radial velocity measurements'
        ];
      case 'FALSE POSITIVE':
        return [
          'Investigate stellar binary hypothesis',
          'Check for instrumental artifacts',
          'Analyze light curve for stellar variability',
          'Consider other astrophysical explanations'
        ];
      default:
        return ['Further analysis required'];
    }
  };

  return (
    <div className={`prediction-card ${predictionStyle.bgColor} relative overflow-hidden`}>
      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-4 right-4 w-2 h-2 bg-white rounded-full animate-twinkle opacity-60"></div>
        <div className="absolute bottom-4 left-4 w-1 h-1 bg-yellow-300 rounded-full animate-pulse-slow opacity-40"></div>
        <div className="absolute top-1/2 right-8 w-1 h-1 bg-blue-300 rounded-full animate-drift opacity-50"></div>
      </div>
      
      <div className="flex items-start space-x-4 relative z-10">
        <div className="flex-shrink-0">
          <IconComponent className={`h-12 w-12 ${predictionStyle.color} animate-pulse-slow`} />
        </div>
        
        <div className="flex-1">
          <h3 className="text-2xl font-bold text-white mb-2">
            {predictionStyle.title}
          </h3>
          
          <div className="space-y-4">
            <p className="text-gray-300 leading-relaxed">
              {predictionStyle.description}
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-black/20 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <BarChart3 className="h-5 w-5 text-nasa-blue" />
                  <span className="font-semibold text-white">Confidence Level</span>
                </div>
                <p className="text-lg font-bold text-nasa-blue">
                  {getConfidenceLevel(prediction.prediction)}
                </p>
              </div>
              
              <div className="bg-black/20 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Info className="h-5 w-5 text-nasa-blue" />
                  <span className="font-semibold text-white">Prediction Code</span>
                </div>
                <p className="text-lg font-bold text-nasa-blue">
                  {prediction.prediction_code}
                </p>
              </div>
            </div>
            
            <div className="bg-black/20 rounded-lg p-4">
              <h4 className="font-semibold text-white mb-3">Recommended Next Steps:</h4>
              <ul className="space-y-2">
                {getNextSteps(prediction.prediction).map((step, index) => (
                  <li key={index} className="flex items-start space-x-2 text-gray-300">
                    <span className="text-nasa-blue font-bold mt-1">•</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
