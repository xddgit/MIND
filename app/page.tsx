"use client";

import { useEffect, useRef, useState } from "react";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const assetPath = (path: string) => `${basePath}${path}`;

const samples = Array.from(
  { length: 20 },
  (_, index) => assetPath(`/assets/mind/samples/sample-${String(index + 1).padStart(2, "0")}.png`),
);

const phaseLabels: Record<string, string> = {
  "soft-argmax": "Soft state · argmax visualization",
  sampled: "Original top-k / top-p sampling",
  greedy: "Original greedy sampling",
};

const trajectorySamples = [1, 5, 6, 8, 12];

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>;
}

function RecordedTrajectory() {
  const frameRef = useRef<number | null>(null);
  const lastAdvanceRef = useRef(0);
  const trajectoryCacheRef = useRef<HTMLImageElement[]>([]);
  const [active, setActive] = useState(0);
  const [sample, setSample] = useState(trajectorySamples[0]);
  const [playing, setPlaying] = useState(false);
  const [loadedSample, setLoadedSample] = useState<number | null>(null);
  const [loadProgress, setLoadProgress] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let loaded = 0;
    setPlaying(false);
    setLoadedSample(null);
    setLoadProgress(0);
    setActive(0);
    lastAdvanceRef.current = 0;
    trajectoryCacheRef.current = [];

    const sampleName = `sample_${String(sample).padStart(2, "0")}`;
    Array.from({ length: 250 }, (_, index) => index + 1).forEach((step) => {
      const image = new Image();
      const complete = () => {
        if (cancelled) return;
        loaded += 1;
        setLoadProgress(loaded);
        if (loaded === 250) {
          setLoadedSample(sample);
          setPlaying(true);
        }
      };
      image.onload = () => {
        image.decode().catch(() => undefined).finally(complete);
      };
      image.onerror = complete;
      image.src = assetPath(`/assets/mind/trajectories/${sampleName}/frame_${String(step).padStart(3, "0")}.jpg`);
      trajectoryCacheRef.current.push(image);
    });

    return () => {
      cancelled = true;
    };
  }, [sample]);

  useEffect(() => {
    if (!playing || loadedSample !== sample) return;
    const animate = (time: number) => {
      if (!lastAdvanceRef.current) lastAdvanceRef.current = time;
      if (time - lastAdvanceRef.current > 140) {
        setActive((value) => (value + 1) % 250);
        lastAdvanceRef.current = time;
      }
      frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [playing, loadedSample, sample]);

  const step = active + 1;
  const phase = step <= 25 ? "soft-argmax" : step >= 226 ? "greedy" : "sampled";
  const cfgActive = active / 250 >= 0.2 && active / 250 <= 0.6;
  const scheduleT = 0.001 + (active / 250) * 0.999;
  const sampleName = `sample_${String(sample).padStart(2, "0")}`;
  const frameName = `frame_${String(step).padStart(3, "0")}.jpg`;

  return (
    <div className="recorded-shell">
      <div className="recorded-visual">
        <img
          src={assetPath(`/assets/mind/trajectories/${sampleName}/${frameName}`)}
          alt={`Decoded token prediction for trajectory ${sample + 1} at sampling step ${step}`}
          decoding="sync"
          fetchPriority="high"
        />
        {loadedSample !== sample && (
          <div className="trajectory-loading" role="status">
            <span>Loading complete trajectory</span>
            <strong>{Math.round((loadProgress / 250) * 100)}%</strong>
            <i><b style={{ width: `${(loadProgress / 250) * 100}%` }} /></i>
          </div>
        )}
        <div className="recorded-badge">v90.31 recorded run · sample {sample + 1}</div>
        <div className="recorded-step">
          <strong>{String(step).padStart(3, "0")}</strong>
          <span>/ 250</span>
        </div>
      </div>
      <div className="recorded-data">
        <div className="recorded-data-head">
          <span>Current sampler state</span>
          <strong>{phaseLabels[phase]}</strong>
        </div>
        <div className="trajectory-picker" aria-label="Choose one of five recorded generations">
          {trajectorySamples.map((index) => (
            <button
              key={index}
              className={sample === index ? "active" : ""}
              onClick={() => {
                setSample(index);
              }}
            >
              <img
                src={assetPath(`/assets/mind/trajectories/sample_${String(index).padStart(2, "0")}/frame_250.jpg`)}
                alt=""
              />
              <span>{String(index + 1).padStart(2, "0")}</span>
            </button>
          ))}
        </div>
        <div className="metric-grid">
          <div><span>Schedule t</span><strong>{scheduleT.toFixed(4)}</strong></div>
          <div><span>Visualized tokens</span><strong>16 × 16</strong></div>
          <div><span>Top-k / top-p</span><strong>100 / 0.70</strong></div>
          <div><span>CFG</span><strong>{cfgActive ? "2.5×" : "1.0×"}</strong></div>
        </div>
        <div className="phase-bar" aria-label="Actual sampler phases">
          <button className={phase === "soft-argmax" ? "active" : ""} onClick={() => { setActive(0); setPlaying(false); }}>
            <span>1–25</span> Soft · argmax view
          </button>
          <button className={phase === "sampled" ? "active" : ""} onClick={() => { setActive(25); setPlaying(false); }}>
            <span>26–225</span> Original sampling
          </button>
          <button className={phase === "greedy" ? "active" : ""} onClick={() => { setActive(225); setPlaying(false); }}>
            <span>226–250</span> Greedy
          </button>
        </div>
        <div className="recorded-controls">
          <button className="play-button" onClick={() => {
            if (loadedSample !== sample) return;
            lastAdvanceRef.current = 0;
            setPlaying((value) => !value);
          }} disabled={loadedSample !== sample}>
            {playing ? "Pause" : "Play"} <span>{playing ? "Ⅱ" : "▶"}</span>
          </button>
          <input
            aria-label="Recorded sampling step"
            type="range"
            min="0"
            max="249"
            value={active}
            onChange={(event) => {
              setPlaying(false);
              setActive(Number(event.target.value));
            }}
          />
          <div className={cfgActive ? "cfg-status active" : "cfg-status"}>
            CFG {cfgActive ? "×2.5 active" : "off"}
          </div>
        </div>
      </div>
    </div>
  );
}

function OneStepShowcase() {
  const [sample, setSample] = useState(0);
  const [decoded, setDecoded] = useState(false);

  useEffect(() => {
    let revealTimer: ReturnType<typeof setTimeout>;
    let resetTimer: ReturnType<typeof setTimeout>;

    const play = () => {
      setDecoded(false);
      revealTimer = setTimeout(() => setDecoded(true), 1100);
      resetTimer = setTimeout(play, 3600);
    };

    play();
    return () => {
      clearTimeout(revealTimer);
      clearTimeout(resetTimer);
    };
  }, [sample]);

  const samplePath = assetPath(`/assets/mind/onestep/sample-${String(sample + 1).padStart(2, "0")}`);

  return (
    <div className="one-step-block">
      <div className="one-step-heading">
        <div>
          <span>One-step generation · 600M</span>
          <h3>Noise in.<br />Image out.</h3>
        </div>
        <p>
          One network evaluation maps a 16 × 16 × 16 continuous latent-noise
          state to image-token logits, followed by a single tokenizer decode.
        </p>
      </div>
      <div className="one-step-shell">
        <div className={decoded ? "one-step-stage decoded" : "one-step-stage"}>
          <img
            className="noise-frame"
            src={`${samplePath}/noise.png`}
            alt={`Three-channel projection of the initial latent noise for one-step sample ${sample + 1}`}
          />
          <img
            className="decoded-frame"
            src={`${samplePath}/final.png`}
            alt={`One-step MIND generation ${sample + 1}`}
          />
          <div className="one-step-state">
            <span>{decoded ? "Output" : "Input"}</span>
            <strong>{decoded ? "Decoded image" : "Latent noise"}</strong>
          </div>
          <div className="one-step-count">01</div>
        </div>
        <div className="one-step-data">
          <div className="one-step-evaluation">
            <span className={decoded ? "complete" : ""} />
            <div>
              <small>Single transition</small>
              <strong>{decoded ? "Decode complete" : "Model evaluation"}</strong>
            </div>
            <b>→</b>
          </div>
          <div className="one-step-metrics">
            <div><span>FID</span><strong>0.90</strong></div>
            <div><span>Network evaluations</span><strong>1</strong></div>
            <div><span>CFG</span><strong>1.5×</strong></div>
            <div><span>Checkpoint</span><strong>260K</strong></div>
          </div>
          <div className="one-step-picker" aria-label="Choose one of twenty one-step generations">
            {Array.from({ length: 20 }, (_, index) => (
              <button
                key={index}
                className={sample === index ? "active" : ""}
                onClick={() => setSample(index)}
                aria-label={`Show one-step generation ${index + 1}`}
              >
                <img
                  src={assetPath(`/assets/mind/onestep/sample-${String(index + 1).padStart(2, "0")}/final.png`)}
                  alt=""
                  loading="lazy"
                />
                <span>{String(index + 1).padStart(2, "0")}</span>
              </button>
            ))}
          </div>
          <button className="one-step-replay" onClick={() => {
            setDecoded(false);
            setTimeout(() => setDecoded(true), 1100);
          }}>
            Replay one step <span>↻</span>
          </button>
        </div>
      </div>
      <p className="visualization-note">
        The input is an exact three-channel projection of the CUDA latent noise
        used for each selected result—not simulated image-space noise. The brief
        transition denotes one model evaluation; no intermediate denoising
        steps are implied.
      </p>
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
          <a href="#method">Method</a>
          <a href="#mechanisms">Architecture</a>
          <a href="#process">Process</a>
          <a href="#results">Results</a>
          <a href="https://github.com/xddgit/MIND" target="_blank" rel="noreferrer">GitHub</a>
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
            <a
              className="secondary-action"
              href="https://github.com/xddgit/MIND"
              target="_blank"
              rel="noreferrer"
            >
              GitHub <ArrowIcon />
            </a>
          </div>
        </div>
        <figure className="hero-figure">
          <img
            src={assetPath("/assets/mind/hero-manifold-v2.png")}
            alt="A cartoon visualization of noisy tokens traveling along a folded data manifold and becoming a coherent image"
          />
          <figcaption>
            <span>01 / Manifold-aware diffusion</span>
            <span>Noise → structure → image</span>
          </figcaption>
        </figure>
        <div className="hero-index">01</div>
      </section>

      <section className="method-intro" id="method">
        <div className="section-number">01</div>
        <div className="section-kicker">Method · At a glance</div>
        <div className="method-intro-heading">
          <h2>Diffuse through a<br />parameterized manifold.</h2>
          <div>
            <p className="lead">
              MIND explicitly brings the geometry of the image data manifold
              into a continuous diffusion model.
            </p>
            <p>
              A tokenizer first maps images to discrete patch tokens. Their
              embeddings define a compact parameterization space in which the
              score network learns to denoise, reducing the metric entropy of
              the learning problem while retaining parallel generation.
            </p>
          </div>
        </div>

        <figure className="method-figure">
          <div className="method-figure-scroll">
            <img
              src={assetPath("/assets/mind/figure1-extended.png")}
              alt="MIND overview showing manifold-aware training, multi-stage inference, one-step FD distillation, and one-step inference"
            />
          </div>
          <figcaption>
            <span>Figure 01 · Complete MIND pipeline</span>
            <span>Training · Inference · One-step distillation</span>
          </figcaption>
        </figure>

        <div className="pipeline-guide">
          <article>
            <span>A</span>
            <h3>Manifold-aware training</h3>
            <p>
              Discrete tokens are projected into a continuous parameterized
              manifold. Forward diffusion produces noisy latents; a DiT predicts
              logits that are supervised through CE and manifold-space MSE losses.
            </p>
          </article>
          <article>
            <span>B</span>
            <h3>Multi-stage inference</h3>
            <p>
              Sampling changes with timestep: differentiable soft sampling,
              entropy-driven hybrid sampling, and greedy projection guide the
              latent toward valid discrete tokens before decoding.
            </p>
          </article>
          <article>
            <span>C</span>
            <h3>One-step FD distillation</h3>
            <p>
              Conditional and unconditional predictions are combined by CFG.
              A greedy straight-through estimator preserves gradients through
              token selection and the decoder to optimize the FD loss.
            </p>
          </article>
          <article>
            <span>D</span>
            <h3>One-step inference</h3>
            <p>
              The distilled model maps initial noise to token logits in one
              network evaluation. Greedy sampling and the tokenizer decoder
              directly produce the final image.
            </p>
          </article>
        </div>

        <div className="contribution-strip">
          <div><strong>Lower metric entropy</strong><span>Explicit manifold parameterization</span></div>
          <div><strong>Soft top-k</strong><span>End-to-end differentiable token projection</span></div>
          <div><strong>High-frequency branches</strong><span>Reduced transformer spectral bias</span></div>
          <div><strong>Greedy STE + FD</strong><span>Differentiable one-step generation</span></div>
        </div>
      </section>

      <section className="method" id="mechanisms">
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
          <h2>Inside five<br />MIND samples.</h2>
          <p>
            Five images generated by MIND-XL-G. Every frame decodes the token
            prediction recorded at that exact step.
          </p>
        </div>
        <RecordedTrajectory />
        <p className="visualization-note">
          The high-quality sampler is unchanged. During the first 25 soft steps,
          argmax tokens are decoded only for visualization; from step 26 onward,
          the frames show the tokens selected by the original sampling logic.
        </p>
        <OneStepShowcase />
      </section>

      <section className="results" id="results">
        <div className="section-number">04</div>
        <div className="section-kicker">Performance · Generated samples</div>
        <div className="results-heading">
          <h2>One manifold.<br />Two sampling regimes.</h2>
          <p>
            One-step FD distillation prioritizes speed; multi-step MIND-XL-G
            achieves stronger distributional fidelity on ImageNet 256 × 256.
          </p>
        </div>
        <div className="performance-compare" aria-label="One-step and multi-step performance comparison">
          <article>
            <div className="performance-label"><span>Multi-step</span><b>250 NFE</b></div>
            <div className="performance-score"><span>FID ↓</span><strong>1.84</strong></div>
            <div className="performance-score"><span>FDr⁶ ↓</span><strong>4.87</strong></div>
            <p>MIND-XL-G · 715M parameters</p>
          </article>
          <article className="one-step-performance">
            <div className="performance-label"><span>One-step</span><b>1 NFE</b></div>
            <div className="performance-score"><span>FID ↓</span><strong>0.90</strong></div>
            <div className="performance-score"><span>FDr⁶ ↓</span><strong>6.77</strong></div>
            <p>FD-distilled MIND · 600M parameters</p>
          </article>
        </div>
        <p className="performance-note">FID and FDr⁶: lower is better.</p>
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
          <a href="https://github.com/xddgit/MIND" target="_blank" rel="noreferrer">
            GitHub code <ArrowIcon />
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
