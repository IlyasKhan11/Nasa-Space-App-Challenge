import React, { useState } from 'react';
import { Send, RotateCcw } from 'lucide-react';

const PredictionForm = ({ onSubmit, onReset }) => {
  const [formData, setFormData] = useState({
    koi_period: '',
    koi_duration: '',
    koi_depth: '',
    koi_prad: '',
    koi_teq: '',
    koi_insol: '',
    koi_model_snr: '',
    koi_steff: '',
    koi_slogg: '',
    koi_srad: '',
    koi_kepmag: '',
    koi_fpflag_nt: '',
    koi_fpflag_ss: '',
    koi_fpflag_co: '',
    koi_fpflag_ec: ''
  });

  const [errors, setErrors] = useState({});

  const fieldDescriptions = {
    koi_period: 'Orbital period in days',
    koi_duration: 'Transit duration in hours',
    koi_depth: 'Transit depth (dimming fraction)',
    koi_prad: 'Planet radius in Earth radii',
    koi_teq: 'Equilibrium temperature in Kelvin',
    koi_insol: 'Stellar insolation in Earth units',
    koi_model_snr: 'Model signal-to-noise ratio',
    koi_steff: 'Stellar effective temperature in Kelvin',
    koi_slogg: 'Stellar surface gravity (log g)',
    koi_srad: 'Stellar radius in solar radii',
    koi_kepmag: 'Kepler magnitude',
    koi_fpflag_nt: 'Not transit-like flag (0/1)',
    koi_fpflag_ss: 'Stellar eclipse flag (0/1)',
    koi_fpflag_co: 'Centroid offset flag (0/1)',
    koi_fpflag_ec: 'Ephemeris match flag (0/1)'
  };

  const validateField = (name, value) => {
    const numValue = parseFloat(value);
    
    if (value === '') return 'This field is required';
    if (isNaN(numValue)) return 'Must be a valid number';
    
    if (name.includes('fpflag')) {
      if (numValue !== 0 && numValue !== 1) {
        return 'Must be 0 or 1';
      }
    } else {
      if (numValue < 0) {
        return 'Must be positive';
      }
    }
    
    return null;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    const error = validateField(name, value);
    setErrors(prev => ({ ...prev, [name]: error }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const newErrors = {};
    let hasErrors = false;
    
    Object.keys(formData).forEach(key => {
      const error = validateField(key, formData[key]);
      if (error) {
        newErrors[key] = error;
        hasErrors = true;
      }
    });
    
    setErrors(newErrors);
    
    if (!hasErrors) {
      const numericData = {};
      Object.keys(formData).forEach(key => {
        numericData[key] = parseFloat(formData[key]);
      });
      onSubmit(numericData);
    }
  };

  const handleReset = () => {
    setFormData({
      koi_period: '',
      koi_duration: '',
      koi_depth: '',
      koi_prad: '',
      koi_teq: '',
      koi_insol: '',
      koi_model_snr: '',
      koi_steff: '',
      koi_slogg: '',
      koi_srad: '',
      koi_kepmag: '',
      koi_fpflag_nt: '',
      koi_fpflag_ss: '',
      koi_fpflag_co: '',
      koi_fpflag_ec: ''
    });
    setErrors({});
    onReset();
  };

  const inputFields = [
    { name: 'koi_period', label: 'Orbital Period (days)', type: 'number', step: '0.01' },
    { name: 'koi_duration', label: 'Transit Duration (hours)', type: 'number', step: '0.01' },
    { name: 'koi_depth', label: 'Transit Depth', type: 'number', step: '0.000001' },
    { name: 'koi_prad', label: 'Planet Radius (Earth radii)', type: 'number', step: '0.01' },
    { name: 'koi_teq', label: 'Equilibrium Temperature (K)', type: 'number', step: '1' },
    { name: 'koi_insol', label: 'Stellar Insolation', type: 'number', step: '0.01' },
    { name: 'koi_model_snr', label: 'Model SNR', type: 'number', step: '0.01' },
    { name: 'koi_steff', label: 'Stellar Temperature (K)', type: 'number', step: '1' },
    { name: 'koi_slogg', label: 'Stellar Surface Gravity', type: 'number', step: '0.01' },
    { name: 'koi_srad', label: 'Stellar Radius (Solar radii)', type: 'number', step: '0.01' },
    { name: 'koi_kepmag', label: 'Kepler Magnitude', type: 'number', step: '0.01' },
    { name: 'koi_fpflag_nt', label: 'Not Transit-like Flag', type: 'number', step: '1' },
    { name: 'koi_fpflag_ss', label: 'Stellar Eclipse Flag', type: 'number', step: '1' },
    { name: 'koi_fpflag_co', label: 'Centroid Offset Flag', type: 'number', step: '1' },
    { name: 'koi_fpflag_ec', label: 'Ephemeris Match Flag', type: 'number', step: '1' }
  ];

  return (
    <div className="card relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-6 right-6 w-1 h-1 bg-blue-400 rounded-full animate-twinkle opacity-60"></div>
        <div className="absolute bottom-6 left-6 w-1 h-1 bg-yellow-400 rounded-full animate-pulse-slow opacity-40"></div>
        <div className="absolute top-1/3 right-1/4 w-1 h-1 bg-green-400 rounded-full animate-drift opacity-50"></div>
      </div>
      
      <div className="relative z-10">
        <h2 className="text-2xl font-semibold mb-6 text-white flex items-center">
          <span className="mr-2">🪐</span>
          Exoplanet Data Input
          <span className="ml-2 text-lg animate-twinkle">✨</span>
        </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {inputFields.map(field => (
            <div key={field.name} className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                {field.label}
                <span className="text-red-400 ml-1">*</span>
              </label>
              <input
                type={field.type}
                name={field.name}
                value={formData[field.name]}
                onChange={handleInputChange}
                step={field.step}
                className={`input-field ${errors[field.name] ? 'border-red-500 focus:ring-red-500' : ''}`}
                placeholder={`Enter ${field.label.toLowerCase()}`}
              />
              {errors[field.name] && (
                <p className="text-red-400 text-sm">{errors[field.name]}</p>
              )}
              <p className="text-xs text-gray-500">{fieldDescriptions[field.name]}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-col sm:flex-row gap-4 pt-6">
          <button
            type="submit"
            className="btn-primary flex items-center justify-center space-x-2"
          >
            <Send className="h-5 w-5" />
            <span>Predict Exoplanet</span>
          </button>
          
          <button
            type="button"
            onClick={handleReset}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            <RotateCcw className="h-5 w-5" />
            <span>Reset Form</span>
          </button>
        </div>
      </form>
      </div>
    </div>
  );
};

export default PredictionForm;
