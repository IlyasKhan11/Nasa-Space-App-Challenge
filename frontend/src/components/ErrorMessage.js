import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

const ErrorMessage = ({ message, onClose }) => {
  return (
    <div className="card border-red-500 bg-red-500/10">
      <div className="flex items-start space-x-3">
        <AlertTriangle className="h-6 w-6 text-red-400 flex-shrink-0 mt-1" />
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-red-400 mb-2">Prediction Error</h3>
          <p className="text-gray-300 mb-4">{message}</p>
          
          <div className="bg-red-900/20 rounded-lg p-3">
            <h4 className="font-semibold text-red-300 mb-2">Possible Solutions:</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Check that all numeric fields contain valid numbers</li>
              <li>• Ensure flag fields (fpflag_*) contain only 0 or 1</li>
              <li>• Verify all values are positive where required</li>
              <li>• Try refreshing the page and submitting again</li>
            </ul>
          </div>
        </div>
        
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
