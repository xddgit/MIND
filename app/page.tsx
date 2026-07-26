"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const samples = Array.from(
  { length: 20 },
  (_, index) => `/assets/mind/samples/sample-${String(index + 1).padStart(2, "0")}.png`,
);

const stages = [
  { label: "Signal", step: "t = 1.00" },
  { label: "Topology", step: "t = 0.72" },
  { label: "Structure", step: "t = 0.43" },
  { label: "Semantics", step: "t = 0.18" },
  { label: "Image", step: "t = 0.00" },
];

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>;
}

function NoiseReveal() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const frameRef = useRef<number | null>(null);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [sample, setSample] = useState(0);

  const paintNoise = useCallback((value: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const size = 256;
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(size, size);
    const block = Math.max(1, Math.round(8 - value * 7));
    for (let y = 0; y < size; y += block) {
      for (let x = 0; x < size; x += block) {
        const tone = Math.floor(Math.random() * 255);
        for (let by = 0; by < block && y + by < size; by += 1) {
          for (let bx = 0; bx < block && x + bx < size; bx += 1) {
            const offset = ((y + by) * size + x + bx) * 4;
            image.data[offset] = tone;
            image.data[offset + 1] = Math.min(255, tone + 8);
            image.data[offset + 2] = Math.max(0, tone - 10);
            image.data[offset + 3] = 255;
          }
        }
      }
    }
    context.putImageData(image, 0, 0);
  }, []);

  useEffect(() => {
    paintNoise(progress);
  }, [paintNoise, progress]);

  useEffect(() => {
    if (!playing) return;
    let start: number | undefined;
    const animate = (time: number) => {
      if (start === undefined) start = time - progress * 7000;
      const elapsed = (time - start) % 7800;
      if (elapsed > 7000) {
        setProgress(1);
      } else {
        setProgress(elapsed / 7000);
      }
      frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [playing, sample]);

  const currentStage = Math.min(4, Math.floor(progress * 5));

  return (
    <div className="process-shell">
      <div className="process-visual">
        <img
          ref={imageRef}
          src={samples[sample]}
          alt={`MIND generated ImageNet sample ${sample + 1}`}
          style={{
            opacity: Math.max(0.05, progress * 1.18),
            filter: `blur(${Math.max(0, 18 - progress * 18)}px) saturate(${0.25 + progress * 0.75}) contrast(${0.7 + progress * 0.3})`,
          }}
        />
        <canvas
          ref={canvasRef}
          aria-hidden="true"
          style={{ opacity: Math.pow(1 - progress, 1.55) }}
        />
        <div className="scan-line" style={{ left: `${progress * 100}%` }} />
        <div className="process-corner process-corner-top">MIND / SAMPLE {String(sample + 1).padStart(2, "0")}</div>
        <div className="process-corner process-corner-bottom">
          {stages[currentStage].step}
        </div>
      </div>

      <div className="process-controls">
        <div className="stage-row" aria-label="Generation stages">
          {stages.map((stage, index) => (
            <button
              className={index === currentStage ? "stage active" : "stage"}
              key={stage.label}
              onClick={() => {
                setPlaying(false);
                setProgress(index / 4);
              }}
            >
              <span>0{index + 1}</span>
              {stage.label}
            </button>
          ))}
        </div>
        <input
          aria-label="Sampling progress"
          className="timeline"
          type="range"
          min="0"
          max="100"
          value={Math.round(progress * 100)}
          onChange={(event) => {
            setPlaying(false);
            setProgress(Number(event.target.value) / 100);
          }}
        />
        <div className="control-footer">
          <button className="play-button" onClick={() => setPlaying((value) => !value)}>
            {playing ? "Pause" : "Play"} <span>{playing ? "Ⅱ" : "▶"}</span>
          </button>
          <div className="sample-switcher">
            <span>Sample</span>
            {[0, 1, 2, 3, 4].map((index) => (
              <button
                aria-label={`Show sample ${index + 1}`}
                className={sample === index ? "active" : ""}
                key={index}
                onClick={() => {
                  setSample(index);
                  setProgress(0);
                  setPlaying(true);
                }}
              >
                {String(index + 1).padStart(2, "0")}
              </button>
            ))}
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
          <img src="/assets/mind/hero.jpg" alt="A curated overview of images generated by MIND-B" />
          <figcaption>
            <span>01 / Selected generations</span>
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
        <div className="section-kicker">Interactive · Sampling visualization</div>
        <div className="process-heading">
          <h2>Watch structure<br />emerge from signal.</h2>
          <p>
            Move through an explanatory reconstruction of the generation path—from
            stochastic signal to the final semantic image.
          </p>
        </div>
        <NoiseReveal />
        <p className="visualization-note">
          Visualization reconstructed from final samples for communication; it is not
          a recorded tensor trajectory from the sampler.
        </p>
      </section>

      <section className="results" id="results">
        <div className="section-number">04</div>
        <div className="section-kicker">Curated samples · MIND-B</div>
        <div className="results-heading">
          <h2>Compact model.<br />High-fidelity manifold.</h2>
          <p>
            Twenty manually selected ImageNet generations from the 1.4M-step
            checkpoint, ordered by semantic quality.
          </p>
        </div>
        <div className="gallery">
          {samples.map((src, index) => (
            <button key={src} className="gallery-item" onClick={() => setLightbox(src)}>
              <img src={src} alt={`Selected MIND-B generation ${index + 1}`} loading="lazy" />
              <span>{String(index + 1).padStart(2, "0")}</span>
              <i aria-hidden="true">↗</i>
            </button>
          ))}
        </div>
      </section>

      <section className="result-overview">
        <img src="/assets/mind/grid.jpg" alt="Grid of forty semantically selected MIND-B samples" loading="lazy" />
        <div>
          <span>05 / Semantic selection</span>
          <h2>One manifold.<br />Many visual worlds.</h2>
          <p>
            Category diversity emerges without trading away local detail, texture,
            or recognizable object structure.
          </p>
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
