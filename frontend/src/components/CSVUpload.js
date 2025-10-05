import React, { useState } from 'react';
import { Upload, FileText, Download, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const CSVUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid CSV file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file first');
      return;
    }

    setUploading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/api/upload-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error uploading file');
    } finally {
      setUploading(false);
    }
  };

  const downloadPDF = () => {
    if (results?.pdf_download_url) {
      window.open(`http://localhost:8000${results.pdf_download_url}`, '_blank');
    }
  };

  const downloadCSV = () => {
    if (results?.csv_download_url) {
      window.open(`http://localhost:8000${results.csv_download_url}`, '_blank');
    }
  };

  return (
    <div className="card p-6 space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2 flex items-center justify-center space-x-2">
          <FileText className="w-6 h-6 text-nasa-blue" />
          <span>Batch Prediction</span>
        </h2>
        <p className="text-gray-300">Upload a CSV file to get predictions for multiple exoplanets</p>
      </div>

      {/* File Upload Area */}
      <div className="border-2 border-dashed border-nasa-blue/30 rounded-lg p-8 text-center hover:border-nasa-blue/50 transition-colors">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="hidden"
          id="csv-upload"
        />
        <label
          htmlFor="csv-upload"
          className="cursor-pointer flex flex-col items-center space-y-4"
        >
          <Upload className="w-12 h-12 text-nasa-blue" />
          <div>
            <p className="text-white font-medium">
              {file ? file.name : 'Click to select CSV file'}
            </p>
            <p className="text-gray-400 text-sm mt-1">
              CSV files only • Max 10MB
            </p>
          </div>
        </label>
      </div>

      {/* Upload Button */}
      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="w-full bg-nasa-blue hover:bg-nasa-blue/80 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {uploading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              <span>Upload & Predict</span>
            </>
          )}
        </button>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-300">{error}</span>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="space-y-4">
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 flex items-center space-x-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-green-300">Predictions completed successfully!</span>
          </div>

          {/* Summary */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3">Prediction Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{results.total_predictions}</div>
                <div className="text-gray-400 text-sm">Total</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400">{results.prediction_summary['FALSE POSITIVE']}</div>
                <div className="text-gray-400 text-sm">False Positive</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-400">{results.prediction_summary['CANDIDATE']}</div>
                <div className="text-gray-400 text-sm">Candidate</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">{results.prediction_summary['CONFIRMED']}</div>
                <div className="text-gray-400 text-sm">Confirmed</div>
              </div>
            </div>
          </div>

          {/* Download Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <button
              onClick={downloadCSV}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Download className="w-5 h-5" />
              <span>Download CSV Results</span>
            </button>
            
            <button
              onClick={downloadPDF}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Download className="w-5 h-5" />
              <span>Download PDF Report</span>
            </button>
          </div>

          {/* Results Preview */}
          {results.results_preview && results.results_preview.length > 0 && (
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-3">Preview (First 10 Results)</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-300">Period</th>
                      <th className="text-left py-2 px-3 text-gray-300">Radius</th>
                      <th className="text-left py-2 px-3 text-gray-300">SNR</th>
                      <th className="text-left py-2 px-3 text-gray-300">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.results_preview.map((result, index) => (
                      <tr key={index} className="border-b border-gray-700/50">
                        <td className="py-2 px-3 text-gray-300">{result.koi_period?.toFixed(2)}</td>
                        <td className="py-2 px-3 text-gray-300">{result.koi_prad?.toFixed(2)}</td>
                        <td className="py-2 px-3 text-gray-300">{result.koi_model_snr?.toFixed(2)}</td>
                        <td className="py-2 px-3">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            result.prediction_label === 'CONFIRMED' ? 'bg-green-500/20 text-green-300' :
                            result.prediction_label === 'CANDIDATE' ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>
                            {result.prediction_label}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* CSV Format Info */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="text-blue-300 font-semibold mb-2">Required CSV Format</h4>
        <p className="text-gray-300 text-sm mb-2">
          Your CSV file must contain these columns:
        </p>
        <div className="text-xs text-gray-400 font-mono bg-gray-900/50 p-2 rounded">
          koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
        </div>
      </div>
    </div>
  );
};

export default CSVUpload;