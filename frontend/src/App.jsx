import React, { useState } from 'react';
import './index.css';

function App() {
  const [mode, setMode] = useState('single'); // 'single' or 'batch'
  
  // Single State
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [patientResult, setPatientResult] = useState(null);
  
  // Batch State
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState([]);

  const [isLoading, setIsLoading] = useState(false);

  const API_URL = 'http://localhost:8000';

  // --- Handlers ---
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreviewUrl(URL.createObjectURL(file));
      setPatientResult(null);
    }
  };

  const handleBatchFilesChange = (e) => {
    const files = Array.from(e.target.files);
    setBatchFiles(files);
    setBatchResults([]);
  };

  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) {
      alert("Please upload a Patient Ultrasound Image first.");
      return;
    }

    setIsLoading(true);
    setPatientResult(null);

    const payload = new FormData();
    payload.append('file', selectedImage);
    
    try {
      const response = await fetch(`${API_URL}/api/predict/comprehensive`, {
        method: 'POST',
        body: payload,
      });
      const data = await response.json();
      setPatientResult(data);
    } catch (error) {
      console.error('Error fetching comprehensive prediction:', error);
      alert("Failed to connect to the Multimodal API Server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    if (batchFiles.length === 0) {
      alert("Please select at least one image for batch processing.");
      return;
    }

    setIsLoading(true);
    setBatchResults([]);

    const payload = new FormData();
    batchFiles.forEach(file => {
      payload.append('files', file);
    });
    
    try {
      const response = await fetch(`${API_URL}/api/predict/batch`, {
        method: 'POST',
        body: payload,
      });
      const data = await response.json();
      setBatchResults(data);
    } catch (error) {
      console.error('Error:', error);
      alert("Failed to connect to the Multimodal API Server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = () => {
    // This allows downloading PDF via native browser print
    window.print();
  };

  return (
    <div>
      {/* Hide controls from print view and enforce clean white styling for PDFs */}
      <style>
        {`
          @media print {
            .no-print { display: none !important; }
            body { background: white !important; color: black !important; }
            .card { background: white !important; border: 1px solid #ddd !important; color: black !important; box-shadow: none !important; margin: 0 !important; }
            .result-value.text-danger { color: red !important; }
            .result-value.text-success { color: green !important; }
            .result-value.text-warning { color: orange !important; }
            .title, .subtitle { color: black !important; }
          }
        `}
      </style>

      <div className="no-print" style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 className="title">Visionary Oncologist Dashboard</h1>
        <p className="subtitle">True Single-Pass Multimodal Deep Learning</p>
        
        {/* Toggle Mode */}
        <div style={{ marginTop: '1rem' }}>
          <button 
            className="btn-primary" 
            style={{ 
              background: mode === 'single' ? 'var(--primary-color)' : 'transparent',
              border: '1px solid var(--primary-color)',
              marginRight: '1rem'
            }}
            onClick={() => { setMode('single'); setBatchResults([]); setPatientResult(null); }}
          >
            Single Patient Mode
          </button>
          <button 
            className="btn-primary" 
            style={{ 
              background: mode === 'batch' ? 'var(--primary-color)' : 'transparent',
              border: '1px solid var(--primary-color)'
            }}
            onClick={() => { setMode('batch'); setBatchResults([]); setPatientResult(null); }}
          >
            Batch Processing Mode
          </button>
        </div>
      </div>

      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        
        {/* --- SINGLE PATIENT MODE --- */}
        {mode === 'single' && (
          <div className="no-print">
            <div className="card">
              <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>Patient File Submission</h2>
              
              <form onSubmit={handleSingleSubmit} style={{ textAlign: 'center' }}>
                <div 
                  className="upload-area" 
                  onClick={() => document.getElementById('imageUpload').click()}
                  style={{ maxWidth: '500px', margin: '0 auto' }}
                >
                  <input 
                    type="file" 
                    id="imageUpload" 
                    accept="image/*" 
                    style={{ display: 'none' }}
                    onChange={handleImageChange}
                  />
                  {imagePreviewUrl ? (
                    <div>
                      <p>Image selected. Click to change.</p>
                      <img src={imagePreviewUrl} alt="Preview" className="image-preview" style={{ maxHeight: '180px' }} />
                    </div>
                  ) : (
                    <div>
                      <h4 style={{ margin: '0 0 1rem 0' }}>Click to upload patient scan</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>(JPG or PNG)</p>
                    </div>
                  )}
                </div>

                <div style={{ marginTop: '2rem' }}>
                  <button 
                    type="submit" 
                    className="btn-primary" 
                    style={{ width: '100%', maxWidth: '400px', fontSize: '1.2rem', padding: '1rem' }}
                    disabled={!selectedImage || isLoading}
                  >
                    {isLoading ? 'Processing Neural Network...' : 'Analyze Patient Profile'}
                  </button>
                </div>
              </form>
            </div>
            
            {isLoading && <div className="loading" style={{ marginTop: '2rem' }}>Fusing Modal Data & Extracting Gradients...</div>}
          </div>
        )}

        {/* --- BATCH PROCESSING MODE --- */}
        {mode === 'batch' && (
          <div className="no-print">
             <div className="card">
              <h2 style={{ textAlign: 'center', marginBottom: '1rem' }}>Advanced Batch Triage</h2>
              <p style={{ textAlign: 'center', marginBottom: '2rem', color: 'var(--text-secondary)' }}>
                Upload multiple patient ultrasounds simultaneously. The AI will prioritize high-risk tumors automatically.
              </p>
              
              <form onSubmit={handleBatchSubmit} style={{ textAlign: 'center' }}>
                <div 
                  className="upload-area" 
                  onClick={() => document.getElementById('batchUpload').click()}
                  style={{ maxWidth: '500px', margin: '0 auto' }}
                >
                  <input 
                    type="file" 
                    id="batchUpload" 
                    accept="image/*" 
                    multiple
                    style={{ display: 'none' }}
                    onChange={handleBatchFilesChange}
                  />
                  <div>
                    <h4 style={{ margin: '0 0 1rem 0' }}>Select Folder or Multiple Images</h4>
                    <p style={{ margin: 0, color: 'var(--primary-color)', fontWeight: 'bold' }}>
                      {batchFiles.length > 0 ? `${batchFiles.length} Scans Selected` : 'Click Here'}
                    </p>
                  </div>
                </div>

                <div style={{ marginTop: '2rem' }}>
                  <button 
                    type="submit" 
                    className="btn-primary" 
                    style={{ width: '100%', maxWidth: '400px', fontSize: '1.2rem', padding: '1rem' }}
                    disabled={batchFiles.length === 0 || isLoading}
                  >
                    {isLoading ? 'Batch Processing in Progress...' : `Analyze ${batchFiles.length} Scans`}
                  </button>
                </div>
              </form>
            </div>
            {isLoading && <div className="loading" style={{ marginTop: '2rem' }}>Parallelizing Matrix Computations...</div>}
          </div>
        )}

        {/* --- RESULTS DASHBOARD (Only shows after success in Single Mode) --- */}
        {patientResult && mode === 'single' && (
          <div className="card" id="patient-report" style={{ marginTop: '2rem', animation: 'fadeIn 0.5s ease' }}>
            <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: 'var(--primary-color)' }}>Final Diagnostic Report</h2>
            
            {/* Top Row: Hard Classifications */}
            <div className="results-grid" style={{ marginBottom: '2rem' }}>
              <div className="result-box">
                <h3 style={{ textAlign: 'center' }}>Image Subtype</h3>
                <div className={`result-value ${patientResult.subtype === 'Malignant' ? 'text-danger' : 'text-success'}`}>
                  {patientResult.subtype}
                </div>
                <p style={{ textAlign: 'center' }}>ResNet50 Confidence: {(patientResult.confidence * 100).toFixed(1)}%</p>
              </div>

              <div className="result-box">
                <h3 style={{ textAlign: 'center' }}>Recurrence Risk</h3>
                <div className={`result-value ${patientResult.recurrence_prob > 0.65 ? 'text-danger' : patientResult.recurrence_prob > 0.35 ? 'text-warning' : 'text-success'}`}>
                  {(patientResult.recurrence_prob * 100).toFixed(1)}%
                </div>
                <p style={{ textAlign: 'center' }}>XGBoost Profile: {patientResult.risk_group}</p>
              </div>
            </div>

            {/* Bottom Row: AI Explanations */}
            <div className="results-grid">
              <div className="result-box">
                <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>Grad-CAM Visualization</h3>
                <img src={patientResult.gradcam_image} alt="Grad-CAM" className="image-preview" style={{ maxHeight: '250px', objectFit: 'contain' }} />
                <p style={{ textAlign: 'center', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Model focus regions highlighted</p>
              </div>

              <div className="result-box">
                <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>SHAP Drivers (Avg Cohort)</h3>
                <div style={{ padding: '0 1rem' }}>
                  {patientResult.shap_features.length > 0 ? (
                    patientResult.shap_features.map((feat, idx) => (
                      <div key={idx} style={{ marginBottom: '1.2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                          <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{feat.feature.toUpperCase()}</span>
                          <span style={{ fontWeight: 600 }}>{feat.value.toFixed(2)}</span>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.1)', height: '10px', borderRadius: '5px', overflow: 'hidden' }}>
                          <div style={{ 
                            width: `${(feat.value / 0.5) * 100}%`, 
                            height: '100%', 
                            background: 'linear-gradient(90deg, var(--primary-color), white)' 
                          }}></div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p style={{ color: 'var(--text-secondary)' }}>SHAP data unavailable.</p>
                  )}
                </div>
              </div>
            </div>

            <div className="no-print" style={{ textAlign: 'center', marginTop: '2rem' }}>
               <button className="btn-primary" onClick={handleDownloadReport} style={{ background: 'darkgreen', border: '1px solid lime' }}>
                  Download PDF Report
               </button>
            </div>
            
          </div>
        )}

        {/* --- BATCH TABLE OUTPUT --- */}
        {batchResults.length > 0 && mode === 'batch' && (
          <div className="card" style={{ marginTop: '2rem', animation: 'fadeIn 0.5s ease' }}>
             <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: 'var(--primary-color)' }}>Batch Triage Results</h2>
             <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                    <th style={{ padding: '1rem' }}>Patient Filename</th>
                    <th style={{ padding: '1rem' }}>Subtype</th>
                    <th style={{ padding: '1rem' }}>Confidence</th>
                    <th style={{ padding: '1rem' }}>Recurrence Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.map((res, index) => (
                    <tr key={index} style={{ 
                        borderBottom: '1px solid rgba(255,255,255,0.05)',
                        background: res.recurrence_prob > 0.65 ? 'rgba(255,0,0,0.1)' : 'transparent'
                      }}>
                      <td style={{ padding: '1rem' }}>
                        {res.filename}
                        <br/>
                        <img src={res.gradcam_image} style={{ height: '60px', borderRadius: '5px', marginTop: '0.5rem' }} alt="Grad-CAM" />
                      </td>
                      <td style={{ padding: '1rem', fontWeight: 'bold' }} className={res.subtype === 'Malignant' ? 'text-danger' : 'text-success'}>
                        {res.subtype}
                      </td>
                      <td style={{ padding: '1rem' }}>{(res.confidence * 100).toFixed(1)}%</td>
                      <td style={{ padding: '1rem', fontWeight: 'bold' }} className={res.recurrence_prob > 0.65 ? 'text-danger' : 'text-warning'}>
                        {(res.recurrence_prob * 100).toFixed(1)}% ({res.risk_group})
                      </td>
                    </tr>
                  ))}
                </tbody>
             </table>
             
             <div className="no-print" style={{ textAlign: 'center', marginTop: '2rem' }}>
               <button className="btn-primary" onClick={handleDownloadReport} style={{ background: 'darkgreen', border: '1px solid lime' }}>
                  Download Triage Report
               </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
