import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  Check,
  Clock3,
  Database,
  Droplets,
  Fish,
  LoaderCircle,
  Play,
  RotateCcw,
  Waves,
} from 'lucide-react';

const featureFields = [
  { key: 'DO', label: 'Dissolved oxygen', unit: 'mg/L', value: '7.0' },
  { key: 'temperature', label: 'Water temperature', unit: '°C', value: '27.0' },
  { key: 'pH', label: 'pH', unit: '', value: '7.5' },
  { key: 'turbidity', label: 'Turbidity', unit: 'NTU', value: '15.0' },
  { key: 'ammonia', label: 'Ammonia', unit: 'mg/L', value: '0.10' },
  { key: 'fish_weight', label: 'Average fish weight', unit: 'g', value: '300.0' },
];

const riskStyles = {
  Low: 'risk-low',
  Moderate: 'risk-moderate',
  High: 'risk-high',
};

function App() {
  const [values, setValues] = useState(() => Object.fromEntries(featureFields.map(({ key, value }) => [key, value])));
  const [sample, setSample] = useState(null);
  const [summary, setSummary] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    fetch('/api/dataset/summary')
      .then((response) => {
        if (!response.ok) throw new Error('API unavailable');
        return response.json();
      })
      .then(setSummary)
      .catch(() => setSummary(null));
  }, []);

  function updateValue(event) {
    setValues((current) => ({ ...current, [event.target.name]: event.target.value }));
    setSample(null);
    setResult(null);
  }

  async function loadDatasetWindow() {
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`/api/dataset/sample?split=TEST&offset=${offset}`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Could not load a dataset window');
      setSample(body);
      setValues(Object.fromEntries(featureFields.map(({ key }) => [key, String(body.readings.at(-1)[key])])));
      setOffset((current) => current + 1);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the model API.');
    } finally {
      setBusy(false);
    }
  }

  async function runPrediction(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    setResult(null);
    const readings = sample
      ? sample.readings
      : [Object.fromEntries(featureFields.map(({ key }) => [key, Number(values[key])]))];
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ readings }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Prediction failed');
      setResult(body);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the model API.');
    } finally {
      setBusy(false);
    }
  }

  const sortedProbabilities = result
    ? Object.entries(result.probabilities).sort((left, right) => right[1] - left[1])
    : [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AquaGuard home">
          <span className="brand-mark"><Waves size={21} strokeWidth={2.2} /></span>
          <span>AQUAGUARD<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-status"><span className="status-dot" /> 5-MINUTE MODEL</div>
      </header>

      <section className="intro" id="top">
        <div>
          <p className="eyebrow"><span /> POND MONITORING / EARLY WARNING</p>
          <h1>Water, in <em>good</em> time.</h1>
          <p className="intro-copy">A 12-hour fish-risk forecast, trained on five-minute pond readings.</p>
        </div>
        <div className="intro-mark" aria-hidden="true"><Fish size={68} strokeWidth={1.15} /></div>
      </section>

      <section className="metrics-strip" aria-label="Model specifications">
        <div className="metric"><Clock3 /><span><strong>5 min</strong><small>SAMPLING INTERVAL</small></span></div>
        <div className="metric"><Activity /><span><strong>144 steps</strong><small>12-HOUR LOOKBACK</small></span></div>
        <div className="metric"><Waves /><span><strong>12 hours</strong><small>FORECAST HORIZON</small></span></div>
        <div className="metric metric-data"><Database /><span><strong>{summary ? summary.rows.toLocaleString() : 'Dataset'}</strong><small>5-MINUTE READINGS</small></span></div>
      </section>

      <div className="workspace-grid">
        <section className="input-panel">
          <div className="panel-heading">
            <div>
              <p className="section-index">01 / INPUT</p>
              <h2>Read the pond</h2>
            </div>
            <Droplets className="heading-icon" size={23} />
          </div>

          <button className={`dataset-button ${sample ? 'is-selected' : ''}`} type="button" onClick={loadDatasetWindow} disabled={busy}>
            <span className="dataset-icon"><Database size={17} /></span>
            <span className="dataset-copy"><strong>{sample ? `Test window · ${sample.pond}` : 'Try a real test window'}</strong><small>{sample ? `${sample.start_time} → ${sample.end_time}` : '144 consecutive rows from one held-out pond'}</small></span>
            {sample ? <Check size={17} /> : <ArrowDownToLine size={17} />}
          </button>

          <form onSubmit={runPrediction}>
            <div className="field-grid">
              {featureFields.map(({ key, label, unit }) => (
                <label className="field" key={key}>
                  <span>{label}</span>
                  <span className="input-wrap">
                    <input name={key} type="number" step="any" value={values[key]} onChange={updateValue} required />
                    {unit && <small>{unit}</small>}
                  </span>
                </label>
              ))}
            </div>

            <div className="mode-note">
              {sample ? <><Check size={15} /> Sequence mode uses 144 actual sensor readings.</> : <><RotateCcw size={15} /> Preview mode repeats this reading across the 12-hour window.</>}
            </div>
            <button className="run-button" type="submit" disabled={busy}>
              {busy ? <LoaderCircle className="spin" size={18} /> : <Play size={17} fill="currentColor" />}
              {busy ? 'Working…' : 'Forecast risk'}
            </button>
            {error && <p className="error-message" role="alert">{error}</p>}
          </form>

          <div className="dataset-footer">
            <span><Database size={15} /> {summary ? `${summary.rows.toLocaleString()} rows · ${Object.keys(summary.splits).length} splits` : 'Bundled 5-minute dataset'}</span>
            <a href="/api/dataset/download" download>Download CSV <ArrowDownToLine size={14} /></a>
          </div>
        </section>

        <section className="result-panel" aria-live="polite">
          <div className="panel-heading">
            <div>
              <p className="section-index">02 / FORECAST</p>
              <h2>Risk outlook</h2>
            </div>
            {result && <span className="latency">{result.latency_ms} ms</span>}
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="empty-rings"><Waves size={32} /></div>
              <strong>The next 12 hours are unwritten.</strong>
              <span>Run a forecast to see the model’s risk assessment.</span>
            </div>
          ) : (
            <div className={`forecast-result ${riskStyles[result.risk_label]}`}>
              <div className="result-kicker"><span className="result-dot" /> RISK CLASSIFICATION</div>
              <div className="risk-word">{result.risk_label}<span> risk</span></div>
              <div className="confidence-line">
                <span>Confidence</span><strong>{(result.confidence * 100).toFixed(1)}%</strong>
              </div>
              <div className="confidence-track"><span style={{ width: `${result.confidence * 100}%` }} /></div>

              <div className="result-details">
                <div><span>BiLSTM output</span><strong>{result.model_label}</strong></div>
                <div><span>Safety assessment</span><strong>{result.biological_label}</strong></div>
                <div><span>Input</span><strong>{sample ? 'Real 144-step window' : 'Single reading preview'}</strong></div>
              </div>

              {sample && <p className="ground-truth">Held-out target at 12h: <strong>{sample.expected_label}</strong></p>}

              <div className="probability-list">
                <p>CLASS PROBABILITIES</p>
                {sortedProbabilities.map(([label, probability]) => (
                  <div className="probability-row" key={label}>
                    <span>{label}</span><div className="probability-track"><span className={riskStyles[label]} style={{ width: `${probability * 100}%` }} /></div><strong>{(probability * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      <footer className="page-footer"><span>AQUAGUARD / 5-MINUTE BILSTM</span><span>12-HOUR EARLY WARNING · RESEARCH PROTOTYPE</span></footer>
    </main>
  );
}

export default App;