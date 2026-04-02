import { useEffect, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";
const HISTORY_STORAGE_KEY = "ai-stethoscope-history";

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value) {
  return new Date(value).toLocaleString();
}

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [analysisStarted, setAnalysisStarted] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [apiMessage, setApiMessage] = useState("Checking backend availability.");
  const [history, setHistory] = useState(() => {
    if (typeof window === "undefined") {
      return [];
    }

    try {
      return JSON.parse(window.localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  });
  const inputRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setAudioUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setAudioUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
          throw new Error("Health check failed.");
        }

        const payload = await response.json();
        if (cancelled) {
          return;
        }

        if (payload.model_ready) {
          setApiStatus("ready");
          setApiMessage("Backend reachable. Model artifact detected.");
          return;
        }

        setApiStatus("degraded");
        setApiMessage("Backend reachable, but the model artifact is not available.");
      } catch (err) {
        if (!cancelled) {
          setApiStatus("offline");
          setApiMessage("Backend unavailable. Start the API before running analysis.");
        }
      }
    }

    checkHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    if (!file || !canvasRef.current) {
      return undefined;
    }

    let cancelled = false;

    async function drawWaveform() {
      try {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          return;
        }

        const context = new AudioContextCtor();
        const arrayBuffer = await file.arrayBuffer();
        const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
        if (cancelled) {
          await context.close();
          return;
        }

        const raw = buffer.getChannelData(0);
        const canvas = canvasRef.current;
        if (!canvas) {
          await context.close();
          return;
        }

        const width = canvas.width;
        const height = canvas.height;
        const ctx = canvas.getContext("2d");
        const step = Math.max(1, Math.floor(raw.length / width));

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#fffaf3";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "#c8663b";
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let x = 0; x < width; x += 1) {
          let peak = 0;
          const start = x * step;
          const end = Math.min(start + step, raw.length);
          for (let i = start; i < end; i += 1) {
            peak = Math.max(peak, Math.abs(raw[i]));
          }
          const barHeight = Math.max(2, peak * height * 0.9);
          const y = (height - barHeight) / 2;
          ctx.moveTo(x, y);
          ctx.lineTo(x, y + barHeight);
        }

        ctx.stroke();
        await context.close();
      } catch {
        // Canvas waveform is progressive enhancement; keep the rest of the flow usable.
      }
    }

    drawWaveform();
    return () => {
      cancelled = true;
    };
  }, [file]);

  function clearSelection() {
    setFile(null);
    setResult(null);
    setError("");
    setAnalysisStarted(false);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function applyFile(nextFile) {
    if (!nextFile) {
      clearSelection();
      return;
    }

    if (!nextFile.name.toLowerCase().endsWith(".wav")) {
      setError("Please choose a WAV recording. Other audio formats are not accepted by the current pipeline.");
      setFile(null);
      setResult(null);
      setAnalysisStarted(false);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
      return;
    }

    setFile(nextFile);
    setResult(null);
    setError("");
  }

  async function handleAnalyze() {
    if (!file) {
      setError("Upload a WAV recording before running analysis.");
      return;
    }

    if (apiStatus === "offline") {
      setError("The backend API is currently unavailable. Start the server and retry the analysis.");
      return;
    }

    setLoading(true);
    setAnalysisStarted(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }

      setResult(payload);
      setHistory((current) => [
        {
          id: `${Date.now()}-${payload.predicted_class}`,
          timestamp: new Date().toISOString(),
          filename: payload.filename,
          predicted_display_name: payload.predicted_display_name,
          predicted_class: payload.predicted_class,
          confidence: payload.confidence,
        },
        ...current,
      ].slice(0, 6));
    } catch (err) {
      setError(err.message || "Prediction failed. Check the backend logs for details.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] ?? null;
    applyFile(nextFile);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setDragActive(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    const nextFile = event.dataTransfer.files?.[0] ?? null;
    applyFile(nextFile);
  }

  function openFilePicker() {
    inputRef.current?.click();
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">AI Stethoscope MVP</p>
          <h1>Heart sound analysis in one guided upload flow</h1>
          <p className="lede">
            Upload a WAV recording, preview the audio, and run the classifier to
            view the predicted heart sound category and confidence scores.
          </p>
        </div>
        <div className="hero-note">
          <strong>Academic use</strong>
          <p>This interface is intended for research demonstration and model inspection, not diagnosis.</p>
        </div>
      </section>

      <section className="workspace-grid">
        <article className="panel upload-panel">
          <div className={`status-banner status-${apiStatus}`}>
            <strong>System status</strong>
            <p>{apiMessage}</p>
          </div>
          <h2>1. Upload recording</h2>
          <div
            className={`upload-dropzone${dragActive ? " is-drag-active" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onClick={openFilePicker}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                openFilePicker();
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".wav,audio/wav"
              onChange={handleFileChange}
            />
            <span>{dragActive ? "Drop the WAV file here" : "Select or drag a `.wav` heart sound file"}</span>
            <small>Mono or stereo WAV supported. The backend converts to mono.</small>
            <button type="button" className="ghost-button upload-button">
              Choose file
            </button>
          </div>

          {file ? (
            <div className="file-summary">
              <div>
                <strong>{file.name}</strong>
                <p>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <button type="button" className="ghost-button" onClick={clearSelection}>
                Remove
              </button>
            </div>
          ) : null}

          <div className="audio-preview">
            <h3>2. Preview audio</h3>
            {audioUrl ? (
              <>
                <audio controls src={audioUrl}>
                  Your browser does not support audio playback.
                </audio>
                <canvas
                  ref={canvasRef}
                  className="waveform-canvas"
                  width="640"
                  height="120"
                  aria-label="Waveform preview"
                />
              </>
            ) : (
              <p>No recording selected yet.</p>
            )}
          </div>

          <button
            type="button"
            className="primary-button"
            onClick={handleAnalyze}
            disabled={!file || loading || apiStatus === "offline"}
          >
            {loading ? "Running analysis..." : analysisStarted ? "Run analysis again" : "3. Analyze recording"}
          </button>

          <p className="action-hint">
            {loading
              ? "The classifier is processing the uploaded phonocardiogram."
              : "Analysis uses the exported YAMNet-based classifier currently configured in the backend."}
          </p>

          {error ? <p className="error-banner">{error}</p> : null}
        </article>

        <article className="panel results-panel">
          <h2>Results</h2>
          {!result ? (
            <p className="placeholder-copy">
              Prediction output will appear here after a successful analysis.
            </p>
          ) : (
            <>
              <div className="result-highlight">
                <p className="result-label">Predicted class</p>
                <h3>{result.predicted_display_name}</h3>
                <p className="result-code">Code: {result.predicted_class}</p>
                <p>{formatPercent(result.confidence)} confidence</p>
              </div>

              <div className="message-card interpretation-card">
                <p>{result.message}</p>
                <small>{result.interpretation}</small>
              </div>

              <div className="meta-grid">
                <div>
                  <span>Filename</span>
                  <strong>{result.filename}</strong>
                </div>
                <div>
                  <span>Duration</span>
                  <strong>{result.audio_seconds}s</strong>
                </div>
                <div>
                  <span>Sample rate</span>
                  <strong>{result.sample_rate_hz} Hz</strong>
                </div>
              </div>

              <div className="score-list">
                {result.ranked_predictions.map((item) => (
                  <div className="score-row" key={item.label}>
                    <div className="score-head">
                      <span>{item.rank}. {item.display_name}</span>
                      <strong>{formatPercent(item.probability)}</strong>
                    </div>
                    <div className="score-bar">
                      <div style={{ width: `${item.probability * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="message-card">
                <small>{result.disclaimer}</small>
              </div>
            </>
          )}
        </article>
      </section>

      <section className="panel history-panel">
        <div className="history-head">
          <h2>Recent local analyses</h2>
          <button
            type="button"
            className="ghost-button"
            onClick={() => setHistory([])}
            disabled={history.length === 0}
          >
            Clear history
          </button>
        </div>
        {history.length === 0 ? (
          <p className="placeholder-copy">
            Completed analyses are stored in this browser only. No history has been recorded yet.
          </p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <article className="history-card" key={item.id}>
                <div>
                  <strong>{item.predicted_display_name}</strong>
                  <p>{item.filename}</p>
                </div>
                <div className="history-metrics">
                  <span>{formatPercent(item.confidence)}</span>
                  <small>{formatTimestamp(item.timestamp)}</small>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

export default App;
