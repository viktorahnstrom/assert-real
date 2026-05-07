import { useState } from 'react';
import { ChevronLeft, X } from 'lucide-react';
import type { DetectionResult, ExplanationResult } from '@/lib/api';

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      onClick={onClose}
    >
      <button onClick={onClose} className="absolute right-6 top-6 text-white/70 hover:text-white">
        <X className="h-6 w-6" />
      </button>
      <div
        className="flex h-[90vh] w-[90vw] items-center justify-center"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={src}
          alt="Full size"
          className="h-full w-full rounded-xl object-contain shadow-2xl"
        />
      </div>
    </div>
  );
}

function TechnicalDetails({
  result,
  isFake,
  explanation,
}: {
  result: DetectionResult;
  isFake: boolean;
  explanation: ExplanationResult | null;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-black/[0.06] bg-white">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between px-6 py-4 text-left"
      >
        <p className="text-[11px] font-semibold uppercase tracking-widest text-xade-charcoal/40">
          Technical Details
        </p>
        <ChevronLeft
          className={`h-4 w-4 text-xade-charcoal/40 transition-transform duration-200 ${
            open ? '-rotate-90' : 'rotate-180'
          }`}
        />
      </button>

      {open && (
        <div className="border-t border-xade-charcoal/10 px-6 py-5">
          <div className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-3">
            <div>
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">Model</p>
              <p className="mt-1 font-medium text-xade-charcoal">{result.model}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                Trained Accuracy
              </p>
              <p className="mt-1 font-medium text-xade-charcoal">{result.accuracy}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">Prediction</p>
              <p className={`mt-1 font-medium ${isFake ? 'text-red-500' : 'text-green-500'}`}>
                {result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1)}
              </p>
            </div>
          </div>

          {result.explanation && (
            <div className="mt-4 grid grid-cols-3 gap-4 border-t border-xade-charcoal/10 pt-4 text-sm">
              <div>
                <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                  Explanation Provider
                </p>
                <p className="mt-1 font-medium text-xade-charcoal">
                  {result.explanation.provider} / {result.explanation.model}
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                  Explanation Time
                </p>
                <p className="mt-1 font-medium text-xade-charcoal">
                  {(result.explanation.processing_time_ms / 1000).toFixed(1)}s
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                  Estimated Cost
                </p>
                <p className="mt-1 font-medium text-xade-charcoal">
                  ${result.explanation.estimated_cost_usd.toFixed(4)}
                </p>
              </div>
            </div>
          )}

          {explanation?.detailed_analysis && (
            <div className="mt-4 border-t border-xade-charcoal/10 pt-4">
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">Full Analysis</p>
              <p className="mt-2 text-sm leading-relaxed text-xade-charcoal/70">
                {explanation.detailed_analysis}
              </p>
            </div>
          )}

          {explanation?.technical_notes && (
            <div className="mt-4 border-t border-xade-charcoal/10 pt-4">
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                Technical Notes
              </p>
              <p className="mt-2 text-sm leading-relaxed text-xade-charcoal/70">
                {explanation.technical_notes}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface AnalysisResultBodyProps {
  result: DetectionResult;
  previewUrl: string | null;
}

export function AnalysisResultBody({ result, previewUrl }: AnalysisResultBodyProps) {
  const isFake = result.prediction === 'fake';
  const confidencePct = Math.round(result.confidence * 100);
  const fakePct = Math.round(result.probabilities.fake * 100);
  const realPct = Math.round(result.probabilities.real * 100);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  const explanation = result.explanation;
  const [hoveredMetric, setHoveredMetric] = useState<{ idx: number; metric: string } | null>(null);

  const METRIC_LABELS: Record<string, string> = {
    sharpness_z: 'Sharpness',
    hf_energy_z: 'HF Energy',
    ela_intensity_z: 'ELA',
  };
  const METRIC_KEYS = ['sharpness_z', 'hf_energy_z', 'ela_intensity_z'] as const;

  return (
    <>
      {lightboxSrc && <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}

      {/* Row 1: Verdict */}
      <div className="mb-4 rounded-2xl border border-black/[0.06] bg-white p-5 sm:p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-8">
          <div className="shrink-0">
            <p
              className={`text-[11px] font-semibold uppercase tracking-widest ${isFake ? 'text-red-400' : 'text-emerald-500'}`}
            >
              {isFake ? 'Deepfake detected' : 'Authentic'}
            </p>
            <p
              className={`mt-0.5 text-5xl font-bold tabular-nums leading-none sm:text-6xl ${isFake ? 'text-red-500' : 'text-emerald-500'}`}
            >
              {confidencePct}%
            </p>
            <p className="mt-1 text-sm text-xade-charcoal/40">confidence</p>
          </div>
          <div className="flex-1 w-full">
            <div className="h-2 w-full overflow-hidden rounded-full bg-xade-charcoal/8">
              <div
                className={`h-full rounded-full ${isFake ? 'bg-red-400' : 'bg-emerald-400'}`}
                style={{ width: `${fakePct}%` }}
              />
            </div>
            <div className="mt-1.5 flex justify-between text-xs text-xade-charcoal/35">
              <span>{fakePct}% fake</span>
              <span>{realPct}% real</span>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Three image tiles */}
      <div className="mb-4 rounded-2xl border border-black/[0.06] bg-white p-5">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-xade-charcoal/40">
          Visual Analysis
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {/* Original */}
          <div>
            <p className="mb-1.5 text-xs text-xade-charcoal/40">Original</p>
            {previewUrl ? (
              <div
                className="w-full cursor-zoom-in overflow-hidden rounded-lg"
                style={{ aspectRatio: '1 / 1' }}
                onClick={() => setLightboxSrc(previewUrl)}
              >
                <img
                  src={previewUrl}
                  alt="Original"
                  className="h-full w-full object-cover transition-opacity hover:opacity-75"
                />
              </div>
            ) : (
              <div
                className="flex w-full items-center justify-center rounded-lg bg-xade-charcoal/5 text-xs text-xade-charcoal/30"
                style={{ aspectRatio: '1 / 1' }}
              >
                Unavailable
              </div>
            )}
          </div>
          {/* Heatmap */}
          <div>
            <p className="mb-1.5 text-xs text-xade-charcoal/40">
              Heatmap
              <span className="ml-1 text-xade-charcoal/25">· red = focus</span>
            </p>
            {result.gradcam_heatmap_url ? (
              <div
                className="w-full cursor-zoom-in overflow-hidden rounded-lg"
                style={{ aspectRatio: '1 / 1' }}
                onClick={() => setLightboxSrc(result.gradcam_heatmap_url!)}
              >
                <img
                  src={result.gradcam_heatmap_url}
                  alt="GradCAM heatmap"
                  className="h-full w-full object-cover transition-opacity hover:opacity-75"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ) : (
              <div
                className="flex w-full items-center justify-center rounded-lg bg-xade-charcoal/5 text-xs text-xade-charcoal/30"
                style={{ aspectRatio: '1 / 1' }}
              >
                Unavailable
              </div>
            )}
          </div>
          {/* ELA */}
          <div>
            <p className="mb-1.5 text-xs text-xade-charcoal/40">
              ELA
              <span className="ml-1 text-xade-charcoal/25">· bright = tampered</span>
            </p>
            {result.ela_heatmap_url ? (
              <div
                className="w-full cursor-zoom-in overflow-hidden rounded-lg"
                style={{ aspectRatio: '1 / 1' }}
                onClick={() => setLightboxSrc(result.ela_heatmap_url!)}
              >
                <img
                  src={result.ela_heatmap_url}
                  alt="ELA overlay"
                  className="h-full w-full object-cover transition-opacity hover:opacity-75"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ) : (
              <div
                className="flex w-full items-center justify-center rounded-lg bg-xade-charcoal/5 text-xs text-xade-charcoal/30"
                style={{ aspectRatio: '1 / 1' }}
              >
                Unavailable
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Row 3: Explanation */}
      <div className="mb-4 rounded-2xl border border-black/[0.06] bg-white p-5">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-xade-charcoal/40">
          Explanation
        </p>
        {explanation ? (
          <p className="text-sm leading-relaxed text-xade-charcoal/70">
            {explanation.summary}
            {explanation.detailed_analysis ? ' ' + explanation.detailed_analysis : ''}
          </p>
        ) : (
          <p className="text-sm text-xade-charcoal/35">
            No explanation available. Select a VLM provider in dev settings.
          </p>
        )}
      </div>

      {/* Row 4: Facial Regions — stacked list */}
      {result.evidence_regions && result.evidence_regions.length > 0 && (
        <div className="mb-4 rounded-2xl border border-black/[0.06] bg-white p-6">
          <div className="mb-4 flex items-baseline justify-between">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-xade-charcoal/40">
              Facial Regions
            </p>
            <span className="text-xs text-xade-charcoal/30">
              {Math.min(result.evidence_regions.length, 3)} regions · by activation
            </span>
          </div>

          <div className="flex flex-col divide-y divide-black/[0.04]">
            {result.evidence_regions.slice(0, 3).map((region, idx) => {
              const actPct = Math.round(region.activation_score * 100);
              const suspicious = isFake && region.activation_score > 0.5;
              const verdictLabel = isFake
                ? region.activation_score > 0.5
                  ? 'Suspicious'
                  : 'Low signal'
                : 'Looks natural';
              const verdictStyle = suspicious
                ? 'bg-red-50 text-red-500'
                : isFake
                  ? 'bg-orange-50 text-orange-400'
                  : 'bg-emerald-50 text-emerald-600';

              return (
                <div
                  key={idx}
                  className="flex items-start gap-3 py-4 first:pt-0 last:pb-0 sm:gap-4"
                >
                  {/* Zoomed crop */}
                  <div
                    className="h-16 w-16 shrink-0 cursor-zoom-in overflow-hidden rounded-xl border border-black/[0.06] sm:h-24 sm:w-24"
                    onClick={() => setLightboxSrc(region.url)}
                  >
                    <img
                      src={region.url}
                      alt={region.label}
                      className="h-full w-full object-cover transition-opacity hover:opacity-75"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>

                  {/* Content */}
                  <div className="flex flex-1 flex-col gap-1.5">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {region.category_label && (
                        <span className="rounded-full bg-xade-blue/8 px-2 py-0.5 text-[11px] font-medium text-xade-blue">
                          {region.category_label}
                        </span>
                      )}
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${verdictStyle}`}
                      >
                        {verdictLabel}
                      </span>
                    </div>

                    <p className="text-sm font-medium text-xade-charcoal/80">{region.label}</p>

                    {region.explanation ? (
                      <p
                        className="cursor-default text-xs leading-relaxed text-xade-charcoal/60"
                        onMouseEnter={() => {
                          if (region.evidence_type === 'metric' && region.evidence_ref) {
                            const metric = region.evidence_ref.split('=')[0].trim();
                            setHoveredMetric({ idx, metric });
                          }
                        }}
                        onMouseLeave={() => setHoveredMetric(null)}
                      >
                        {region.explanation}
                      </p>
                    ) : (
                      <p className="text-xs text-xade-charcoal/35">
                        No region-level explanation available.
                      </p>
                    )}

                    {/* Forensic z-score strip */}
                    {region.z_scores && METRIC_KEYS.some((k) => region.z_scores![k] != null) && (
                      <div className="mt-2 flex flex-col gap-1">
                        {METRIC_KEYS.map((key) => {
                          const z = region.z_scores![key];
                          if (z == null) return null;
                          const clampedZ = Math.max(-3, Math.min(3, z));
                          const fillPct = (Math.abs(clampedZ) / 3) * 50;
                          const isNeg = clampedZ < 0;
                          const absZ = Math.abs(z);
                          const isHighlighted =
                            hoveredMetric?.idx === idx && hoveredMetric?.metric === key;
                          const barColor = isHighlighted
                            ? 'bg-xade-blue'
                            : absZ >= 2.5
                              ? 'bg-red-400'
                              : absZ >= 1.5
                                ? 'bg-orange-400'
                                : 'bg-emerald-400';
                          return (
                            <div key={key} className="flex items-center gap-2">
                              <span className="w-16 shrink-0 text-right text-[10px] text-xade-charcoal/40">
                                {METRIC_LABELS[key]}
                              </span>
                              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-xade-charcoal/8">
                                <div className="absolute inset-y-0 left-1/2 w-px bg-xade-charcoal/25" />
                                <div
                                  className={`absolute inset-y-0 ${barColor} transition-colors`}
                                  style={{
                                    left: isNeg ? `${50 - fillPct}%` : '50%',
                                    width: `${fillPct}%`,
                                  }}
                                />
                              </div>
                              <span className="w-10 shrink-0 text-[10px] tabular-nums text-xade-charcoal/40">
                                {z >= 0 ? '+' : ''}
                                {z.toFixed(1)}σ
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Activation */}
                  <div className="flex shrink-0 flex-col items-end gap-1.5 pt-0.5">
                    <span className="text-sm font-semibold tabular-nums text-xade-charcoal/60">
                      {actPct}%
                    </span>
                    <div className="h-1 w-16 overflow-hidden rounded-full bg-xade-charcoal/8">
                      <div
                        className={`h-full rounded-full ${suspicious ? 'bg-red-400' : 'bg-xade-charcoal/20'}`}
                        style={{ width: `${actPct}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-xade-charcoal/30">activation</span>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="mt-4 text-[10px] leading-relaxed text-xade-charcoal/30">
            Bars compare each region to the real-face distribution (0 = average real face). Positive
            = above average · Negative = below average. <span className="text-red-400">Red</span> =
            unusual (&gt;2.5σ) · <span className="text-orange-400">Orange</span> = moderate ·{' '}
            <span className="text-emerald-500">Green</span> = normal. Hover a claim to highlight its
            cited metric.
          </p>
        </div>
      )}

      <TechnicalDetails result={result} isFake={isFake} explanation={explanation} />
    </>
  );
}
