"use client";

import { useEffect, useRef, useState } from "react";

const samples = Array.from(
  { length: 20 },
  (_, index) => `/assets/mind/samples/sample-${String(index + 1).padStart(2, "0")}.png`,
);

const trajectory = [
  { step: 0, t: 0.0010, phase: "soft-top-k", cfg: 1, entropy: 7.6643, confidence: 0.0151, gap: 0.5372, x: 1.0000, y: 0.2788, frame: "frame_000.png" },
  { step: 1, t: 0.0050, phase: "soft-top-k", cfg: 1, entropy: 7.7431, confidence: 0.0138, gap: 0.5338, x: 0.9632, y: 0.3121, frame: "frame_001.png" },
  { step: 5, t: 0.0210, phase: "soft-top-k", cfg: 1, entropy: 8.0066, confidence: 0.0102, gap: 0.5285, x: 0.8556, y: 0.3433, frame: "frame_005.png" },
  { step: 12, t: 0.0491, phase: "soft-top-k", cfg: 1, entropy: 8.3019, confidence: 0.0071, gap: 0.5117, x: 0.5394, y: 0.0689, frame: "frame_012.png" },
  { step: 24, t: 0.0967, phase: "soft-top-k", cfg: 1, entropy: 8.5245, confidence: 0.0057, gap: 0.5186, x: 0.5704, y: -0.1003, frame: "frame_024.png" },
  { step: 25, t: 0.1011, phase: "stochastic-token", cfg: 1, entropy: 8.4976, confidence: 0.0061, gap: 0.4845, x: 0.5657, y: -0.1141, frame: "frame_025.png" },
  { step: 50, t: 0.2012, phase: "stochastic-token", cfg: 3, entropy: 7.0206, confidence: 0.0388, gap: 0.5487, x: 0.1859, y: -0.2835, frame: "frame_050.png" },
  { step: 80, t: 0.3203, phase: "stochastic-token", cfg: 3, entropy: 5.8928, confidence: 0.0835, gap: 0.4999, x: 0.1585, y: -1.0000, frame: "frame_080.png" },
  { step: 120, t: 0.4805, phase: "stochastic-token", cfg: 3, entropy: 3.7724, confidence: 0.2521, gap: 0.4462, x: -0.5123, y: -0.3592, frame: "frame_120.png" },
  { step: 160, t: 0.6406, phase: "stochastic-token", cfg: 1, entropy: 2.0438, confidence: 0.5155, gap: 0.3631, x: -0.8159, y: -0.2084, frame: "frame_160.png" },
  { step: 200, t: 0.8008, phase: "stochastic-token", cfg: 1, entropy: 0.2449, confidence: 0.9258, gap: 0.2638, x: -0.7907, y: 0.2462, frame: "frame_200.png" },
  { step: 224, t: 0.8945, phase: "stochastic-token", cfg: 1, entropy: 0.0004, confidence: 1.0000, gap: 0.1952, x: -0.8903, y: 0.3114, frame: "frame_224.png" },
  { step: 225, t: 0.8984, phase: "greedy-projection", cfg: 1, entropy: 0.0004, confidence: 1.0000, gap: 0.1918, x: -0.8970, y: 0.3281, frame: "frame_225.png" },
  { step: 249, t: 0.9961, phase: "greedy-projection", cfg: 1, entropy: 0.0000, confidence: 1.0000, gap: 0.0376, x: -0.9327, y: 0.1769, frame: "frame_249.png" },
];

const phaseLabels: Record<string, string> = {
  "soft-top-k": "Soft top-k projection",
  "stochastic-token": "Stochastic token sampling",
  "greedy-projection": "Greedy token projection",
};

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>;
}

function TrajectoryPlot({ active }: { active: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    const margin = 30;
    const point = (record: (typeof trajectory)[number]) => ({
      x: margin + ((record.x + 1) / 2) * (width - margin * 2),
      y: margin + ((1 - record.y) / 2) * (height - margin * 2),
    });

    context.strokeStyle = "rgba(23,23,20,.16)";
    context.lineWidth = 1;
    for (let index = 0; index < 5; index += 1) {
      const position = margin + index * ((width - margin * 2) / 4);
      context.beginPath();
      context.moveTo(position, margin);
      context.lineTo(position, height - margin);
      context.stroke();
    }
    for (let index = 0; index < 4; index += 1) {
      const position = margin + index * ((height - margin * 2) / 3);
      context.beginPath();
      context.moveTo(margin, position);
      context.lineTo(width - margin, position);
      context.stroke();
    }

    context.strokeStyle = "#171714";
    context.lineWidth = 2;
    context.beginPath();
    trajectory.forEach((record, index) => {
      const position = point(record);
      if (index === 0) context.moveTo(position.x, position.y);
      else context.lineTo(position.x, position.y);
    });
    context.stroke();

    trajectory.forEach((record, index) => {
      const position = point(record);
      context.beginPath();
      context.arc(position.x, position.y, index === active ? 7 : 3.5, 0, Math.PI * 2);
      context.fillStyle = index <= active ? "#171714" : "#f1efe8";
      context.fill();
      context.strokeStyle = "#171714";
      context.stroke();
    });

    const current = point(trajectory[active]);
    context.beginPath();
    context.arc(current.x, current.y, 12, 0, Math.PI * 2);
    context.strokeStyle = "#d8ff38";
    context.lineWidth = 4;
    context.stroke();
  }, [active]);

  return <canvas ref={canvasRef} className="trajectory-canvas" aria-label="PCA projection of the recorded continuous-state trajectory" />;
}

function RecordedTrajectory() {
  const frameRef = useRef<number | null>(null);
  const lastAdvanceRef = useRef(0);
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const animate = (time: number) => {
      if (!lastAdvanceRef.current) lastAdvanceRef.current = time;
      if (time - lastAdvanceRef.current > 850) {
        setActive((value) => (value + 1) % trajectory.length);
        lastAdvanceRef.current = time;
      }
      frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [playing]);

  const current = trajectory[active];

  return (
    <div className="recorded-shell">
      <div className="recorded-visual">
        <img
          src={`/assets/mind/trajectory/${current.frame}`}
          alt={`Decoded MIND prediction at sampling step ${current.step}`}
        />
        <div className="recorded-badge">Recorded run · 250 steps</div>
        <div className="recorded-step">
          <strong>{String(current.step).padStart(3, "0")}</strong>
          <span>/ 249</span>
        </div>
      </div>
      <div className="recorded-data">
        <div className="recorded-data-head">
          <span>Current sampler state</span>
          <strong>{phaseLabels[current.phase]}</strong>
        </div>
        <div className="trajectory-chart">
          <TrajectoryPlot active={active} />
          <div className="axis-label axis-y">PC2</div>
          <div className="axis-label axis-x">PC1</div>
          <span>Continuous-state trajectory · PCA projection</span>
        </div>
        <div className="metric-grid">
          <div><span>Schedule t</span><strong>{current.t.toFixed(4)}</strong></div>
          <div><span>Token entropy</span><strong>{current.entropy.toFixed(3)}</strong></div>
          <div><span>Mean confidence</span><strong>{(current.confidence * 100).toFixed(1)}%</strong></div>
          <div><span>Projection gap</span><strong>{current.gap.toFixed(3)}</strong></div>
        </div>
        <div className="phase-bar" aria-label="Actual sampler phases">
          <button className={current.phase === "soft-top-k" ? "active" : ""} onClick={() => { setActive(0); setPlaying(false); }}>
            <span>0–24</span> Soft top-k
          </button>
          <button className={current.phase === "stochastic-token" ? "active" : ""} onClick={() => { setActive(5); setPlaying(false); }}>
            <span>25–224</span> Stochastic
          </button>
          <button className={current.phase === "greedy-projection" ? "active" : ""} onClick={() => { setActive(12); setPlaying(false); }}>
            <span>225–249</span> Greedy
          </button>
        </div>
        <div className="recorded-controls">
          <button className="play-button" onClick={() => {
            lastAdvanceRef.current = 0;
            setPlaying((value) => !value);
          }}>
            {playing ? "Pause" : "Play"} <span>{playing ? "Ⅱ" : "▶"}</span>
          </button>
          <input
            aria-label="Recorded sampling step"
            type="range"
            min="0"
            max={trajectory.length - 1}
            value={active}
            onChange={(event) => {
              setPlaying(false);
              setActive(Number(event.target.value));
            }}
          />
          <div className={current.cfg > 1 ? "cfg-status active" : "cfg-status"}>
            CFG {current.cfg > 1 ? "×3 active" : "off"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [lightbox, setLightbox] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <main id="top">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="MIND home">
          <span className="brand-mark">M</span>
          <span>MIND</span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#overview">Overview</a>
          <a href="#method">Method</a>
          <a href="#process">Process</a>
          <a href="#results">Results</a>
        </nav>
        <a
          className="header-paper"
          href="https://arxiv.org/abs/2606.00094"
          target="_blank"
          rel="noreferrer"
        >
          Paper <ArrowIcon />
        </a>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">Research project · 2026</p>
          <h1 id="hero-title">
            <strong>MIND</strong>
            <span>Diffusion on the</span>
            <span>data manifold.</span>
          </h1>
          <p className="hero-deck">
            Image generation with explicit modeling of data manifold geometry.
          </p>
          <div className="author-line">
            <a href="https://scholar.google.com/citations?user=hN_465sAAAAJ" target="_blank" rel="noreferrer">
              Duoduo Xue
            </a>
            <span>·</span>
            <span>Zhiyu Zhu</span>
            <span>·</span>
            <span>Junhui Hou</span>
          </div>
          <div className="hero-actions">
            <a className="primary-action" href="#process">
              Explore the process <span>↓</span>
            </a>
            <a
              className="secondary-action"
              href="https://arxiv.org/pdf/2606.00094"
              target="_blank"
              rel="noreferrer"
            >
              Read paper <ArrowIcon />
            </a>
          </div>
        </div>
        <figure className="hero-figure">
          <img src="/assets/mind/hero.jpg" alt="An overview of images generated by MIND-B" />
          <figcaption>
            <span>01 / Generated samples</span>
            <span>ImageNet 256 × 256</span>
          </figcaption>
        </figure>
        <div className="hero-index">01</div>
      </section>

      <section className="statement" id="overview">
        <div className="section-number">01</div>
        <div className="section-kicker">Overview · Abstract</div>
        <div className="statement-grid">
          <h2>A geometry-aware route from noise to image.</h2>
          <div className="abstract-copy">
            <p className="lead">
              Generative models seek to sample from a dense, low-dimensional and
              compact data manifold. <strong>MIND</strong> makes that geometry explicit.
            </p>
            <p>
              The framework integrates discrete patch tokenization into the score
              function of a continuous diffusion model, combining the structural
              quantization of discrete tokens with the parallel generation flexibility
              of continuous diffusion.
            </p>
            <p>
              A differentiable soft top-k aggregation mechanism enables end-to-end
              training, while dual-branch high-frequency embeddings counter the
              spectral bias of transformer backbones. At inference, multi-stage
              transition sampling adapts the sampling behavior over time.
            </p>
          </div>
        </div>
        <div className="metric-strip" aria-label="Headline results">
          <div><strong>2.06</strong><span>FID · MIND-B</span></div>
          <div><strong>130M</strong><span>Parameters</span></div>
          <div><strong>1.95</strong><span>FID · MIND-XL</span></div>
          <div><strong>256²</strong><span>ImageNet resolution</span></div>
        </div>
      </section>

      <section className="method" id="method">
        <div className="section-number">02</div>
        <div className="section-kicker">Architecture · Key ideas</div>
        <div className="method-heading">
          <h2>Discrete structure.<br />Continuous motion.</h2>
          <p>
            MIND inserts manifold-aware structure where a continuous diffusion
            transformer needs it most—inside the score function itself.
          </p>
        </div>
        <div className="method-flow" aria-label="MIND method overview">
          <article>
            <span className="method-id">01</span>
            <div className="method-symbol token-grid" aria-hidden="true">
              {Array.from({ length: 16 }).map((_, index) => <i key={index} />)}
            </div>
            <h3>Discrete patch tokenization</h3>
            <p>Structural anchors form a compact parameterization of the image manifold.</p>
          </article>
          <div className="flow-arrow">→</div>
          <article>
            <span className="method-id">02</span>
            <div className="method-symbol top-k" aria-hidden="true">
              <i /><i /><i /><i /><i />
            </div>
            <h3>Soft top-k aggregation</h3>
            <p>A differentiable bridge lets discrete assignments learn end-to-end.</p>
          </article>
          <div className="flow-arrow">→</div>
          <article>
            <span className="method-id">03</span>
            <div className="method-symbol frequency" aria-hidden="true">
              <i /><i />
            </div>
            <h3>High-frequency embedding</h3>
            <p>Dual branches recover detail that low-dimensional transformer inputs can miss.</p>
          </article>
          <div className="flow-arrow">→</div>
          <article>
            <span className="method-id">04</span>
            <div className="method-symbol transition" aria-hidden="true"><i /></div>
            <h3>Transition sampling</h3>
            <p>The sampler changes strategy across time as topology turns into texture.</p>
          </article>
        </div>
      </section>

      <section className="process" id="process">
        <div className="section-number">03</div>
        <div className="section-kicker">Recorded run · Sampling trajectory</div>
        <div className="process-heading">
          <h2>Inside one<br />MIND sample.</h2>
          <p>
            A real 250-step run from the 1.4M-step checkpoint. Each frame decodes
            the token prediction at that exact sampler step.
          </p>
        </div>
        <RecordedTrajectory />
        <p className="visualization-note">
          The 2D path is a PCA projection of the per-step mean continuous
          token-embedding state. Projection gap is the RMS distance from that
          continuous state to its predicted token-embedding projection.
        </p>
      </section>

      <section className="results" id="results">
        <div className="section-number">04</div>
        <div className="section-kicker">Generated samples · MIND-B</div>
        <div className="results-heading">
          <h2>Compact model.<br />High-fidelity manifold.</h2>
          <p>
            Twenty ImageNet generations from the 1.4M-step checkpoint.
          </p>
        </div>
        <div className="gallery">
          {samples.map((src, index) => (
            <button key={src} className="gallery-item" onClick={() => setLightbox(src)}>
              <img src={src} alt={`MIND-B generation ${index + 1}`} loading="lazy" />
              <span>{String(index + 1).padStart(2, "0")}</span>
              <i aria-hidden="true">↗</i>
            </button>
          ))}
        </div>
      </section>

      <section className="conclusion">
        <p className="eyebrow">Conclusion</p>
        <h2>Model the manifold.<br />Then move through it.</h2>
        <div className="conclusion-actions">
          <a href="https://arxiv.org/abs/2606.00094" target="_blank" rel="noreferrer">
            arXiv paper <ArrowIcon />
          </a>
          <a href="#top">Back to top ↑</a>
        </div>
      </section>

      <footer>
        <div className="brand"><span className="brand-mark">M</span><span>MIND</span></div>
        <p>Data Manifold-aware Image diffusioN moDel</p>
        <p>Research project page · 2026</p>
      </footer>

      {lightbox && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label="Expanded result">
          <button className="lightbox-backdrop" onClick={() => setLightbox(null)} aria-label="Close expanded image" />
          <img src={lightbox} alt="Expanded MIND-B generation" />
          <button className="lightbox-close" onClick={() => setLightbox(null)}>Close ×</button>
        </div>
      )}
    </main>
  );
}
