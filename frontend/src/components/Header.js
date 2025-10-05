import React from 'react';
import { Rocket, Star } from 'lucide-react';
import AnimatedPlanet from './AnimatedPlanet';

const Header = () => {
  return (
    <header className="glass-effect border-b border-gray-700">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Rocket className="h-8 w-8 text-nasa-blue animate-float" />
              <Star className="h-4 w-4 text-yellow-400 absolute -top-1 -right-1 animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl font-nasa font-bold text-white">NASA Exoplanet AI</h1>
              <p className="text-sm text-gray-400">Advanced Detection System</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <div className="text-right">
              <div className="text-sm text-gray-300 font-medium">Proudly presented by</div>
              <div className="text-xl font-bold text-nasa-blue bg-nasa-blue/10 px-3 py-1 rounded-lg border border-nasa-blue/30">
                team AstroVenture
              </div>
            </div>
            <div className="relative">
              <AnimatedPlanet 
                size="w-6 h-6" 
                color="bg-nasa-blue" 
                orbitRadius="w-12 h-12"
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-white font-bold text-lg animate-pulse">🚀</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
