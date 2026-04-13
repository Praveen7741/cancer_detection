import React, { useState } from 'react';
import './index.css';

function App() {
  // Auth State
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [username, setUsername] = useState(localStorage.getItem('username'));
  const [authMode, setAuthMode] = useState('login');
  const [authForm, setAuthForm] = useState({ username: '', password: '', email: '', full_name: '', hospital_branch: '' });
  const [otpForm, setOtpForm] = useState({ email: '', otp: '', new_password: '' });
  const [authError, setAuthError] = useState('');

  // Single State
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [patientResult, setPatientResult] = useState(null);

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

  const handleAuthChange = (e) => {
    setAuthForm({ ...authForm, [e.target.name]: e.target.value });
  };

  const handleOtpChange = (e) => {
    setOtpForm({ ...otpForm, [e.target.name]: e.target.value });
  };

  const handleForgotSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: otpForm.email }),
      });
      if (!response.ok) throw new Error('Failed to send OTP');
      setAuthMode('reset');
      setAuthError('OTP sent to your email.');
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(otpForm),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to reset password');
      }
      setAuthMode('login');
      setAuthError('Password reset successful! Please log in.');
      setOtpForm({ email: '', otp: '', new_password: '' });
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    setIsLoading(true);

    try {
      if (authMode === 'signup') {
        const response = await fetch(`${API_URL}/api/signup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(authForm),
        });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Signup failed');
        }
        setAuthMode('login');
        setAuthError('Signup successful! Please log in.');
        setAuthForm({ username: '', password: '', email: '', full_name: '', hospital_branch: '' });
      } else {
        const formData = new FormData();
        formData.append('username', authForm.username);
        formData.append('password', authForm.password);

        const response = await fetch(`${API_URL}/api/login`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error('Invalid username or password');
        }
        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('username', data.username);
        setToken(data.access_token);
        setUsername(data.username);
      }
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setToken(null);
    setUsername('');
    setPatientResult(null);
    setSelectedImage(null);
    setImagePreviewUrl('');
    setAuthError('');
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
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: payload,
      });
      if (response.status === 401) {
        handleLogout();
        alert("Session expired. Please log in again.");
        return;
      }
      const data = await response.json();
      setPatientResult(data);
    } catch (error) {
      console.error('Error fetching comprehensive prediction:', error);
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

      <div className="no-print" style={{ textAlign: 'center', marginBottom: '2rem', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
        <div>
          <h1 className="title">Visionary Oncologist Dashboard</h1>
          <p className="subtitle">True Single-Pass Multimodal Deep Learning</p>
        </div>
        {token && (
          <div style={{ position: 'absolute', right: '0', top: '10px' }}>
            <span style={{ marginRight: '1rem', color: 'var(--text-secondary)' }}>Dr. {username}</span>
            <button className="btn-primary" onClick={handleLogout} style={{ background: 'transparent', border: '1px solid var(--primary-color)', padding: '0.5rem 1rem' }}>Logout</button>
          </div>
        )}
      </div>

      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        
        {!token ? (
          <div className="card" style={{ maxWidth: '400px', margin: '0 auto' }}>
            <h2 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
              {authMode === 'login' && 'Secure System Login'}
              {authMode === 'signup' && 'Register Account'}
              {authMode === 'forgot' && 'Forgot Password'}
              {authMode === 'reset' && 'Enter Verification OTP'}
            </h2>
            {authError && <div style={{ color: authError.includes('successful') || authError.includes('sent') ? 'var(--primary-color)' : 'red', textAlign: 'center', marginBottom: '1rem' }}>{authError}</div>}
            
            {(authMode === 'login' || authMode === 'signup') && (
              <form onSubmit={handleAuthSubmit}>
              <div style={{ marginBottom: '1rem' }}>
                <input
                  type="text"
                  name="username"
                  placeholder="Username"
                  value={authForm.username}
                  onChange={handleAuthChange}
                  style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                  required
                />
              </div>
              {authMode === 'signup' && (
                <>
                  <div style={{ marginBottom: '1rem' }}>
                    <input
                      type="email"
                      name="email"
                      placeholder="Email Address"
                      value={authForm.email}
                      onChange={handleAuthChange}
                      style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                      required
                    />
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <input
                      type="text"
                      name="full_name"
                      placeholder="Full Name"
                      value={authForm.full_name}
                      onChange={handleAuthChange}
                      style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                      required
                    />
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <input
                      type="text"
                      name="hospital_branch"
                      placeholder="Hospital Branch"
                      value={authForm.hospital_branch}
                      onChange={handleAuthChange}
                      style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                      required
                    />
                  </div>
                </>
              )}
              <div style={{ marginBottom: '1.5rem' }}>
                <input
                  type="password"
                  name="password"
                  placeholder="Password"
                  value={authForm.password}
                  onChange={handleAuthChange}
                  style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                  required
                />
              </div>
              <button 
                type="submit" 
                className="btn-primary" 
                style={{ width: '100%', padding: '0.8rem' }}
                disabled={isLoading}
              >
                {isLoading ? 'Processing...' : (authMode === 'login' ? 'Login' : 'Sign Up')}
              </button>
              </form>
            )}

            {authMode === 'forgot' && (
              <form onSubmit={handleForgotSubmit}>
                <div style={{ marginBottom: '1.5rem' }}>
                  <input
                    type="email"
                    name="email"
                    placeholder="Registered Email Address"
                    value={otpForm.email}
                    onChange={handleOtpChange}
                    style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                    required
                  />
                </div>
                <button type="submit" className="btn-primary" style={{ width: '100%', padding: '0.8rem' }} disabled={isLoading}>
                  {isLoading ? 'Processing...' : 'Send OTP'}
                </button>
              </form>
            )}

            {authMode === 'reset' && (
              <form onSubmit={handleResetSubmit}>
                <div style={{ marginBottom: '1rem' }}>
                  <input
                    type="text"
                    name="otp"
                    placeholder="6-Digit OTP"
                    value={otpForm.otp}
                    onChange={handleOtpChange}
                    style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white', letterSpacing: '2px' }}
                    required
                  />
                </div>
                <div style={{ marginBottom: '1.5rem' }}>
                  <input
                    type="password"
                    name="new_password"
                    placeholder="New Password"
                    value={otpForm.new_password}
                    onChange={handleOtpChange}
                    style={{ width: '100%', padding: '0.8rem', borderRadius: '5px', border: '1px solid #444', background: '#222', color: 'white' }}
                    required
                  />
                </div>
                <button type="submit" className="btn-primary" style={{ width: '100%', padding: '0.8rem' }} disabled={isLoading}>
                  {isLoading ? 'Processing...' : 'Reset Password'}
                </button>
              </form>
            )}
            
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              {(authMode === 'login' || authMode === 'signup') && (
                <>
                  <button 
                    type="button"
                    onClick={() => {
                      setAuthMode(authMode === 'login' ? 'signup' : 'login');
                      setAuthError('');
                    }}
                    style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', textDecoration: 'underline', marginBottom: '0.5rem', display: 'block', width: '100%' }}
                  >
                    {authMode === 'login' ? "Don't have an account? Sign up" : "Already have an account? Login"}
                  </button>
                  {authMode === 'login' && (
                    <button 
                      type="button"
                      onClick={() => { setAuthMode('forgot'); setAuthError(''); }}
                      style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', textDecoration: 'underline' }}
                    >
                      Forgot Password?
                    </button>
                  )}
                </>
              )}
              {(authMode === 'forgot' || authMode === 'reset') && (
                <button 
                  type="button"
                  onClick={() => { setAuthMode('login'); setAuthError(''); setOtpForm({ email: '', otp: '', new_password: '' }); }}
                  style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', textDecoration: 'underline' }}
                >
                  Back to Login
                </button>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* --- SINGLE PATIENT MODE --- */}
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

        {/* --- RESULTS DASHBOARD --- */}
        {patientResult && (
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
          </>
        )}

      </div>
    </div>
  );
}

export default App;
