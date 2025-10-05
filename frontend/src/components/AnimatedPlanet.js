import React from 'react';

const AnimatedPlanet = ({ size = 'w-8 h-8', color = 'bg-blue-400', orbitRadius = 'w-16 h-16' }) => {
  return (
    <div className={`relative ${orbitRadius} animate-orbit`}>
      <div className={`absolute top-0 left-1/2 transform -translate-x-1/2 ${size} ${color} rounded-full animate-pulse-slow shadow-lg`}>
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/20 to-transparent"></div>
      </div>
    </div>
  );
};

export default AnimatedPlanet;
