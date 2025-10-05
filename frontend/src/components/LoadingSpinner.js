import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = () => {
  return (
    <div className="card relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-4 left-4 w-2 h-2 bg-nasa-blue rounded-full animate-pulse-slow opacity-60"></div>
        <div className="absolute bottom-4 right-4 w-1 h-1 bg-yellow-400 rounded-full animate-twinkle opacity-40"></div>
        <div className="absolute top-1/2 left-1/4 w-1 h-1 bg-green-400 rounded-full animate-drift opacity-50"></div>
      </div>
      
      <div className="flex items-center justify-center space-x-3 relative z-10">
        <div className="relative">
          <Loader2 className="h-8 w-8 text-nasa-blue animate-spin" />
          <div className="absolute inset-0 border-2 border-nasa-blue/30 rounded-full animate-ping"></div>
        </div>
        <div className="text-center">
          <h3 className="text-lg font-semibold text-white">Analyzing Data</h3>
          <p className="text-gray-400">Processing exoplanet parameters...</p>
        </div>
      </div>
      
      <div className="mt-6 space-y-2">
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <div className="w-2 h-2 bg-nasa-blue rounded-full animate-pulse"></div>
          <span>Validating input parameters</span>
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <div className="w-2 h-2 bg-nasa-blue rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
          <span>Applying preprocessing transformations</span>
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <div className="w-2 h-2 bg-nasa-blue rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
          <span>Running machine learning model</span>
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <div className="w-2 h-2 bg-nasa-blue rounded-full animate-pulse" style={{animationDelay: '1.5s'}}></div>
          <span>Generating classification result</span>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;
