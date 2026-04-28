import { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle, Loader2, XCircle } from 'lucide-react';
import { studyAnalyzeImage, saveStudyResults, type StudyAnalysisResult } from '@/lib/api';

// ============================================
// Study images
// Note: thesis specifies 15 images; add 3 more to /public/quiz-images/ when available.
// ============================================
interface StudyImage {
  id: number;
  url: string;
  label: 'real' | 'fake';
}

const ALL_IMAGES: StudyImage[] = [
  { id: 1, url: '/quiz-images/Fake1.jpg', label: 'fake' },
  { id: 2, url: '/quiz-images/Fake2.jpg', label: 'fake' },
  { id: 3, url: '/quiz-images/Fake3.jpg', label: 'fake' },
  { id: 4, url: '/quiz-images/Fake4.jpg', label: 'fake' },
  { id: 5, url: '/quiz-images/Fake5.jpg', label: 'fake' },
  { id: 6, url: '/quiz-images/Fake6.jpg', label: 'fake' },
  { id: 7, url: '/quiz-images/Real1.jpg', label: 'real' },
  { id: 8, url: '/quiz-images/Real2.jpg', label: 'real' },
  { id: 9, url: '/quiz-images/Real3.jpg', label: 'real' },
  { id: 10, url: '/quiz-images/Real4.jpg', label: 'real' },
  { id: 11, url: '/quiz-images/Real5.jpg', label: 'real' },
  { id: 12, url: '/quiz-images/Real6.jpg', label: 'real' },
];

function shuffleArray<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function generateParticipantId(): string {
  return 'p-' + Math.random().toString(36).slice(2, 10);
}

// ============================================
// Types
// ============================================
type StudyPhase =
  | 'intro'
  | 'confidence'
  | 'classification'
  | 'analyzing'
  | 'explanation'
  | 'survey'
  | 'complete';

interface ClassificationRecord {
  image: StudyImage;
  answer: 'real' | 'fake';
  isCorrect: boolean;
}

interface ProviderAssignment {
  A: string;
  B: string;
  C: string;
  D: string;
}

interface ExplanationItem {
  image: StudyImage;
  userAnswer: 'real' | 'fake';
  analysis: StudyAnalysisResult | null;
  assignment: ProviderAssignment;
  preferredExplanation: 'A' | 'B' | 'C' | 'D' | 'all' | 'none' | null;
  understandingRating: number | null;
  mostUsefulPart: 'heatmap' | 'text' | 'confidence' | null;
}

// ============================================
// Shared UI helpers
// ============================================
function RatingButtons({
  value,
  onChange,
  labels,
}: {
  value: number | null;
  onChange: (v: number) => void;
  labels?: [string, string];
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className={`flex h-10 w-10 items-center justify-center rounded-lg border-2 text-sm font-semibold transition-colors ${
              value === n
                ? 'border-xade-blue bg-xade-blue text-white'
                : 'border-xade-charcoal/15 bg-white text-xade-charcoal/60 hover:border-xade-blue/40 hover:text-xade-charcoal'
            }`}
          >
            {n}
          </button>
        ))}
      </div>
      {labels && (
        <div className="flex justify-between text-[10px] text-xade-charcoal/40">
          <span>{labels[0]}</span>
          <span>{labels[1]}</span>
        </div>
      )}
    </div>
  );
}

// ============================================
// Phase: Intro
// ============================================
function IntroScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-lg">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-xade-blue">XADE</h1>
          <p className="mt-1 text-[11px] font-medium uppercase tracking-[0.2em] text-xade-charcoal/40">
            Deepfake Detection · User Study
          </p>
        </div>

        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <h2 className="text-xl font-semibold text-xade-charcoal">Welcome to the study</h2>
          <p className="mt-2 text-sm leading-relaxed text-xade-charcoal/60">
            This study evaluates how well AI-generated explanations help people understand deepfake
            detection decisions. It takes approximately <strong>10–15 minutes</strong>.
          </p>

          <div className="mt-6 space-y-4">
            {[
              {
                step: '1',
                title: 'Baseline Classification',
                desc: 'Classify 12 images as real or fake — no feedback or hints.',
              },
              {
                step: '2',
                title: 'Explanation Evaluation',
                desc: 'For images you got wrong, compare four AI explanations and rate their usefulness.',
              },
              {
                step: '3',
                title: 'Closing Survey',
                desc: 'Answer a short questionnaire about your overall experience.',
              },
            ].map(({ step, title, desc }) => (
              <div key={step} className="flex gap-3">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-xade-blue/10 text-xs font-bold text-xade-blue">
                  {step}
                </div>
                <div>
                  <p className="text-sm font-medium text-xade-charcoal">{title}</p>
                  <p className="text-xs leading-relaxed text-xade-charcoal/50">{desc}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-lg bg-xade-charcoal/3 px-4 py-3 text-xs leading-relaxed text-xade-charcoal/50">
            <strong className="text-xade-charcoal/70">Privacy:</strong> Participation is voluntary
            and anonymous. No personally identifiable information is collected. Results are used
            solely for academic research.
          </div>

          <button
            onClick={onStart}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            I understand — start the study
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>

        <button
          onClick={() => {
            localStorage.setItem('xade-test-completed', 'skipped');
            window.location.reload();
          }}
          className="mt-4 block w-full text-center text-xs text-xade-charcoal/30 hover:text-xade-charcoal/50"
        >
          Skip for now
        </button>
      </div>
    </div>
  );
}

// ============================================
// Phase: Self-confidence rating
// ============================================
function ConfidenceScreen({ onNext }: { onNext: (rating: number) => void }) {
  const [rating, setRating] = useState<number | null>(null);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-md">
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <p className="text-xs font-medium uppercase tracking-widest text-xade-blue/60">
            Before we start
          </p>
          <h2 className="mt-2 text-xl font-semibold text-xade-charcoal">
            How confident are you at spotting deepfakes?
          </h2>
          <p className="mt-2 text-sm text-xade-charcoal/50">
            Be honest — there are no right or wrong answers here.
          </p>

          <div className="mt-8">
            <RatingButtons
              value={rating}
              onChange={setRating}
              labels={['Not confident at all', 'Very confident']}
            />
          </div>

          <button
            onClick={() => rating !== null && onNext(rating)}
            disabled={rating === null}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark disabled:opacity-40"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Phase: Classification (Phase 1)
// ============================================
function ClassificationScreen({
  image,
  current,
  total,
  onAnswer,
}: {
  image: StudyImage;
  current: number;
  total: number;
  onAnswer: (answer: 'real' | 'fake') => void;
}) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-lg">
        <div className="mb-2 flex items-center justify-between text-xs text-xade-charcoal/40">
          <span>
            Image {current} of {total}
          </span>
          <span>{Math.round((current / total) * 100)}%</span>
        </div>
        <div className="mb-6 h-1.5 w-full rounded-full bg-xade-charcoal/10">
          <div
            className="h-1.5 rounded-full bg-xade-blue transition-all duration-300"
            style={{ width: `${(current / total) * 100}%` }}
          />
        </div>

        <div className="overflow-hidden rounded-2xl border border-xade-charcoal/6 bg-white shadow-lg shadow-xade-charcoal/4">
          <img
            src={image.url}
            alt={`Study image ${current}`}
            className="aspect-square w-full object-cover"
          />
        </div>

        <p className="mt-4 text-center text-xs text-xade-charcoal/40">
          Is this image real or a deepfake?
        </p>

        <div className="mt-3 flex gap-3">
          <button
            onClick={() => onAnswer('real')}
            className="flex-1 rounded-lg border-2 border-green-200 bg-white px-6 py-3.5 text-sm font-semibold text-green-600 transition-colors hover:border-green-400 hover:bg-green-50"
          >
            Real
          </button>
          <button
            onClick={() => onAnswer('fake')}
            className="flex-1 rounded-lg border-2 border-red-200 bg-white px-6 py-3.5 text-sm font-semibold text-red-500 transition-colors hover:border-red-400 hover:bg-red-50"
          >
            Fake
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Phase: Analyzing (loading screen between P1 and P2)
// ============================================
function AnalyzingScreen({ done, total }: { done: number; total: number }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-sm text-center">
        <Loader2 className="mx-auto h-10 w-10 animate-spin text-xade-blue" />
        <h2 className="mt-6 text-xl font-semibold text-xade-charcoal">Analysing your responses</h2>
        <p className="mt-2 text-sm text-xade-charcoal/50">
          Running our AI model and generating explanations for each image you got wrong.
          {total > 0 && (
            <>
              {' '}
              ({done}/{total} done)
            </>
          )}
        </p>
        <p className="mt-4 text-xs text-xade-charcoal/30">This may take up to a minute.</p>
      </div>
    </div>
  );
}

// ============================================
// Phase: Explanation evaluation (Phase 2)
// ============================================
function ExplanationScreen({
  item,
  current,
  total,
  onSubmit,
}: {
  item: ExplanationItem;
  current: number;
  total: number;
  onSubmit: (answers: {
    preferredExplanation: 'A' | 'B' | 'C' | 'D' | 'all' | 'none';
    understandingRating: number;
    mostUsefulPart: 'heatmap' | 'text' | 'confidence';
  }) => void;
}) {
  const [preferred, setPreferred] = useState<'A' | 'B' | 'C' | 'D' | 'all' | 'none' | null>(null);
  const [understanding, setUnderstanding] = useState<number | null>(null);
  const [usefulPart, setUsefulPart] = useState<'heatmap' | 'text' | 'confidence' | null>(null);

  const canSubmit = preferred !== null && understanding !== null && usefulPart !== null;
  const { analysis, assignment } = item;

  function explanationText(label: 'A' | 'B' | 'C' | 'D') {
    const providerId = assignment[label];
    const exp = analysis?.explanations[providerId];
    if (!exp) return { summary: 'Unavailable.', detailed: '' };
    return { summary: exp.summary, detailed: exp.detailed_analysis };
  }

  const gradcamUrl = analysis?.gradcam_url ?? null;

  const fakeScore = analysis ? Math.round(analysis.deepfake_score * 100) : null;

  return (
    <div className="min-h-screen bg-xade-cream px-6 py-8">
      <div className="mx-auto max-w-6xl">
        {/* Header */}
        <div className="mb-2 flex items-center justify-between text-xs text-xade-charcoal/40">
          <span>
            Incorrect image {current} of {total}
          </span>
          <span>{Math.round((current / total) * 100)}%</span>
        </div>
        <div className="mb-6 h-1.5 w-full rounded-full bg-xade-charcoal/10">
          <div
            className="h-1.5 rounded-full bg-xade-blue transition-all duration-300"
            style={{ width: `${(current / total) * 100}%` }}
          />
        </div>

        <p className="mb-4 text-xs font-medium uppercase tracking-widest text-xade-blue/60">
          Phase 2 — Explanation Evaluation
        </p>

        {/* Image row: original + heatmap + confidence */}
        <div className="grid grid-cols-3 gap-4">
          {/* Original image */}
          <div className="rounded-xl border border-xade-charcoal/6 bg-white p-3 shadow-sm">
            <p className="mb-2 text-center text-[10px] font-medium uppercase tracking-wider text-xade-charcoal/40">
              Original
            </p>
            <img
              src={item.image.url}
              alt="Original"
              className="aspect-square w-full rounded-lg object-cover"
            />
            <p className="mt-2 text-center text-xs text-xade-charcoal/50">
              Was:{' '}
              <span
                className={
                  item.image.label === 'fake'
                    ? 'font-semibold text-red-500'
                    : 'font-semibold text-green-600'
                }
              >
                {item.image.label}
              </span>{' '}
              · You said:{' '}
              <span className="font-semibold text-xade-charcoal/70">{item.userAnswer}</span>
            </p>
          </div>

          {/* Grad-CAM heatmap */}
          <div className="rounded-xl border border-xade-charcoal/6 bg-white p-3 shadow-sm">
            <p className="mb-2 text-center text-[10px] font-medium uppercase tracking-wider text-xade-charcoal/40">
              Grad-CAM Heatmap
            </p>
            {gradcamUrl ? (
              <img
                src={gradcamUrl}
                alt="Grad-CAM heatmap"
                className="aspect-square w-full rounded-lg object-cover"
              />
            ) : (
              <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-xade-charcoal/5 text-xs text-xade-charcoal/30">
                Unavailable
              </div>
            )}
            <p className="mt-2 text-center text-[10px] leading-relaxed text-xade-charcoal/40">
              Highlighted areas are where the model focused most
            </p>
          </div>

          {/* Confidence score */}
          <div className="flex flex-col rounded-xl border border-xade-charcoal/6 bg-white p-3 shadow-sm">
            <p className="mb-2 text-center text-[10px] font-medium uppercase tracking-wider text-xade-charcoal/40">
              AI Confidence
            </p>
            <div className="flex flex-1 flex-col items-center justify-center gap-3">
              {fakeScore !== null ? (
                <>
                  <p className="text-4xl font-bold text-xade-charcoal">{fakeScore}%</p>
                  <p className="text-xs text-xade-charcoal/50">probability of being fake</p>
                  <div className="w-full">
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-xade-charcoal/10">
                      <div
                        className="h-2.5 rounded-full bg-red-400 transition-all"
                        style={{ width: `${fakeScore}%` }}
                      />
                    </div>
                    <div className="mt-1 flex justify-between text-[10px] text-xade-charcoal/30">
                      <span>Real</span>
                      <span>Fake</span>
                    </div>
                  </div>
                  <p className="text-xs font-medium text-xade-charcoal/70">
                    Classified as:{' '}
                    <span
                      className={
                        analysis?.classification === 'fake' ? 'text-red-500' : 'text-green-600'
                      }
                    >
                      {analysis?.classification}
                    </span>
                  </p>
                </>
              ) : (
                <p className="text-xs text-xade-charcoal/30">Unavailable</p>
              )}
            </div>
          </div>
        </div>

        {/* Four explanations */}
        <div className="mt-6 grid grid-cols-4 gap-4">
          {(['A', 'B', 'C', 'D'] as const).map((label) => {
            const { summary } = explanationText(label);
            return (
              <div
                key={label}
                className="rounded-xl border border-xade-charcoal/6 bg-white p-4 shadow-sm"
              >
                <p className="mb-3 text-xs font-bold uppercase tracking-widest text-xade-blue">
                  Explanation {label}
                </p>
                <p className="text-xs font-medium leading-relaxed text-xade-charcoal/80">
                  {summary}
                </p>
              </div>
            );
          })}
        </div>

        {/* Questions */}
        <div className="mt-6 rounded-2xl border border-xade-charcoal/6 bg-white px-6 py-6 shadow-sm">
          {/* Q1 */}
          <div className="mb-6">
            <p className="text-sm font-medium text-xade-charcoal">
              1. Which explanation helped you understand the detection decision best?
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {(['A', 'B', 'C', 'D', 'all', 'none'] as const).map((opt) => (
                <button
                  key={opt}
                  onClick={() => setPreferred(opt)}
                  className={`rounded-lg border-2 px-4 py-2 text-xs font-semibold transition-colors ${
                    preferred === opt
                      ? 'border-xade-blue bg-xade-blue text-white'
                      : 'border-xade-charcoal/15 text-xade-charcoal/60 hover:border-xade-blue/40'
                  }`}
                >
                  {opt === 'all' ? 'All equally' : opt === 'none' ? 'None' : `Explanation ${opt}`}
                </button>
              ))}
            </div>
          </div>

          {/* Q2 */}
          <div className="mb-6">
            <p className="text-sm font-medium text-xade-charcoal">
              2. How well do you now understand the detection decision?
            </p>
            <div className="mt-3">
              <RatingButtons
                value={understanding}
                onChange={setUnderstanding}
                labels={['Not at all', 'Completely']}
              />
            </div>
          </div>

          {/* Q3 */}
          <div>
            <p className="text-sm font-medium text-xade-charcoal">
              3. Which part of the explanation did you find most useful?
            </p>
            <div className="mt-3 flex gap-2">
              {(
                [
                  { value: 'heatmap', label: 'Heatmap' },
                  { value: 'text', label: 'Text description' },
                  { value: 'confidence', label: 'Confidence score' },
                ] as const
              ).map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setUsefulPart(value)}
                  className={`rounded-lg border-2 px-4 py-2 text-xs font-semibold transition-colors ${
                    usefulPart === value
                      ? 'border-xade-blue bg-xade-blue text-white'
                      : 'border-xade-charcoal/15 text-xade-charcoal/60 hover:border-xade-blue/40'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          disabled={!canSubmit}
          onClick={() =>
            canSubmit &&
            onSubmit({
              preferredExplanation: preferred!,
              understandingRating: understanding!,
              mostUsefulPart: usefulPart!,
            })
          }
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark disabled:opacity-40"
        >
          {current < total ? 'Next image' : 'Continue to survey'}
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

// ============================================
// Phase: Closing survey (Phase 3)
// ============================================
function SurveyScreen({
  onSubmit,
}: {
  onSubmit: (answers: { trustRating: number; willingnessToUse: string; comments: string }) => void;
}) {
  const [trust, setTrust] = useState<number | null>(null);
  const [willingness, setWillingness] = useState<string | null>(null);
  const [comments, setComments] = useState('');

  const canSubmit = trust !== null && willingness !== null;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-lg">
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <p className="text-xs font-medium uppercase tracking-widest text-xade-blue/60">
            Phase 3 — Closing Survey
          </p>
          <h2 className="mt-2 text-xl font-semibold text-xade-charcoal">Almost done</h2>
          <p className="mt-1 text-sm text-xade-charcoal/50">
            A few final questions about your overall experience.
          </p>

          {/* Q1: Trust */}
          <div className="mt-8">
            <p className="text-sm font-medium text-xade-charcoal">
              1. How much do you trust the XADE system&apos;s ability to detect deepfakes?
            </p>
            <div className="mt-3">
              <RatingButtons
                value={trust}
                onChange={setTrust}
                labels={['No trust', 'Full trust']}
              />
            </div>
          </div>

          {/* Q2: Willingness */}
          <div className="mt-6">
            <p className="text-sm font-medium text-xade-charcoal">
              2. Would you use a tool like XADE to check whether images are deepfakes?
            </p>
            <div className="mt-3 flex gap-2">
              {(['yes', 'no', 'maybe'] as const).map((opt) => (
                <button
                  key={opt}
                  onClick={() => setWillingness(opt)}
                  className={`rounded-lg border-2 px-5 py-2 text-xs font-semibold transition-colors ${
                    willingness === opt
                      ? 'border-xade-blue bg-xade-blue text-white'
                      : 'border-xade-charcoal/15 text-xade-charcoal/60 hover:border-xade-blue/40'
                  }`}
                >
                  {opt.charAt(0).toUpperCase() + opt.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Q3: Open text */}
          <div className="mt-6">
            <p className="text-sm font-medium text-xade-charcoal">
              3. Any additional comments? (optional)
            </p>
            <textarea
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Share any thoughts about the explanations, the interface, or the study..."
              rows={4}
              className="mt-2 w-full rounded-lg border border-xade-charcoal/10 bg-white px-4 py-3 text-sm text-xade-charcoal placeholder:text-xade-charcoal/30 focus:border-xade-blue/40 focus:outline-none focus:ring-2 focus:ring-xade-blue/10"
            />
          </div>

          <button
            disabled={!canSubmit}
            onClick={() =>
              canSubmit &&
              onSubmit({
                trustRating: trust!,
                willingnessToUse: willingness!,
                comments,
              })
            }
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark disabled:opacity-40"
          >
            Submit
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Phase: Complete
// ============================================
function CompleteScreen({
  classificationRecords,
  onContinue,
}: {
  classificationRecords: ClassificationRecord[];
  onContinue: () => void;
}) {
  const correct = classificationRecords.filter((r) => r.isCorrect).length;
  const total = classificationRecords.length;
  const pct = Math.round((correct / total) * 100);

  let message: string;
  if (pct >= 80) {
    message = 'Impressive! But AI-generated deepfakes are getting harder to spot every day.';
  } else if (pct >= 50) {
    message =
      "Not bad — but as you can see, it's tricky. That's exactly why tools like XADE exist.";
  } else {
    message =
      "Don't worry — most people struggle with this. Deepfakes are designed to fool the human eye.";
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-md">
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <div className="text-center">
            <CheckCircle className="mx-auto h-10 w-10 text-green-500" />
            <h2 className="mt-4 text-xl font-semibold text-xade-charcoal">
              Thank you for participating!
            </h2>
            <p className="mt-1 text-sm text-xade-charcoal/50">Your responses have been recorded.</p>
          </div>

          <div className="mt-6 text-center">
            <p className="text-5xl font-bold text-xade-blue">
              {correct}/{total}
            </p>
            <p className="mt-1 text-sm text-xade-charcoal/50">{pct}% correct in Phase 1</p>
          </div>

          <p className="mt-4 text-center text-sm leading-relaxed text-xade-charcoal/60">
            {message}
          </p>

          <div className="mt-6 space-y-2">
            {classificationRecords.map((r, i) => (
              <div
                key={r.image.id}
                className="flex items-center justify-between rounded-lg bg-xade-charcoal/3 px-3 py-2 text-xs"
              >
                <span className="text-xade-charcoal/50">Image {i + 1}</span>
                <div className="flex items-center gap-2">
                  <span className="text-xade-charcoal/40">
                    was {r.image.label} · you said {r.answer}
                  </span>
                  {r.isCorrect ? (
                    <CheckCircle className="h-3.5 w-3.5 text-green-500" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-red-400" />
                  )}
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={onContinue}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            Continue to XADE
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Main orchestrator
// ============================================
interface DeepfakeTestProps {
  onComplete: () => void;
}

// Precomputed analyses structure saved by the /api/v1/study/precompute endpoint
interface PrecomputedData {
  generated_at: string;
  analyses: Record<string, StudyAnalysisResult>;
}

export default function DeepfakeTest({ onComplete }: DeepfakeTestProps) {
  const participantId = useRef(generateParticipantId());
  const [images] = useState<StudyImage[]>(() => shuffleArray(ALL_IMAGES));

  const [phase, setPhase] = useState<StudyPhase>('intro');
  const [selfConfidence, setSelfConfidence] = useState<number | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [classificationRecords, setClassificationRecords] = useState<ClassificationRecord[]>([]);
  const [explanationItems, setExplanationItems] = useState<ExplanationItem[]>([]);
  const [currentExplanationIndex, setCurrentExplanationIndex] = useState(0);
  const [analyzeProgress, setAnalyzeProgress] = useState({ done: 0, total: 0 });

  // Precomputed analyses — loaded at mount, null means not yet checked
  const [precomputed, setPrecomputed] = useState<PrecomputedData | null | 'unavailable'>(
    'unavailable'
  );

  useEffect(() => {
    fetch('/study-analyses.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setPrecomputed(data ?? 'unavailable'))
      .catch(() => setPrecomputed('unavailable'));
  }, []);

  // ---- Phase 1 → analysis/explanation ----
  async function startAnalysis(records: ClassificationRecord[]) {
    const wrong = records.filter((r) => !r.isCorrect);

    if (wrong.length === 0) {
      setPhase('survey');
      return;
    }

    const hasPrecomputed = precomputed !== null && precomputed !== 'unavailable';

    if (!hasPrecomputed) {
      // Fallback: live analysis (slow — researcher should run /precompute first)
      setAnalyzeProgress({ done: 0, total: wrong.length });
      setPhase('analyzing');
    }

    const items: ExplanationItem[] = [];
    for (const record of wrong) {
      const providers = shuffleArray(['openai', 'google', 'anthropic', 'rule_based']);
      const assignment: ProviderAssignment = {
        A: providers[0],
        B: providers[1],
        C: providers[2],
        D: providers[3],
      };

      let analysis: StudyAnalysisResult | null = null;

      if (hasPrecomputed) {
        // Instant — use the precomputed JSON
        analysis = (precomputed as PrecomputedData).analyses[String(record.image.id)] ?? null;
      } else {
        try {
          analysis = await studyAnalyzeImage(record.image.url);
        } catch (err) {
          console.warn('[XADE study] Live analysis failed for', record.image.url, err);
        }
        setAnalyzeProgress((p) => ({ ...p, done: p.done + 1 }));
      }

      items.push({
        image: record.image,
        userAnswer: record.answer,
        analysis,
        assignment,
        preferredExplanation: null,
        understandingRating: null,
        mostUsefulPart: null,
      });
    }

    setExplanationItems(items);
    setCurrentExplanationIndex(0);
    setPhase('explanation');
  }

  // ---- Classification answer ----
  function handleClassificationAnswer(answer: 'real' | 'fake') {
    const image = images[currentImageIndex];
    const isCorrect = answer === image.label;
    const newRecords = [...classificationRecords, { image, answer, isCorrect }];
    setClassificationRecords(newRecords);

    if (currentImageIndex + 1 < images.length) {
      setCurrentImageIndex(currentImageIndex + 1);
    } else {
      void startAnalysis(newRecords);
    }
  }

  // ---- Explanation answer ----
  function handleExplanationSubmit(answers: {
    preferredExplanation: 'A' | 'B' | 'C' | 'D' | 'all' | 'none';
    understandingRating: number;
    mostUsefulPart: 'heatmap' | 'text' | 'confidence';
  }) {
    const updated = explanationItems.map((item, i) =>
      i === currentExplanationIndex ? { ...item, ...answers } : item
    );
    setExplanationItems(updated);

    if (currentExplanationIndex + 1 < explanationItems.length) {
      setCurrentExplanationIndex(currentExplanationIndex + 1);
    } else {
      setPhase('survey');
    }
  }

  // ---- Survey submit ----
  async function handleSurveySubmit(answers: {
    trustRating: number;
    willingnessToUse: string;
    comments: string;
  }) {
    const correct = classificationRecords.filter((r) => r.isCorrect).length;
    const payload = {
      participant_id: participantId.current,
      self_confidence_rating: selfConfidence ?? 0,
      baseline_accuracy: classificationRecords.length ? correct / classificationRecords.length : 0,
      total_images: classificationRecords.length,
      correct_count: correct,
      incorrect_count: classificationRecords.length - correct,
      explanation_answers: explanationItems.map((item) => ({
        image_id: item.image.id,
        image_label: item.image.label,
        user_answer: item.userAnswer,
        assignment: item.assignment,
        preferred_explanation: item.preferredExplanation,
        understanding_rating: item.understandingRating,
        most_useful_part: item.mostUsefulPart,
      })),
      trust_rating: answers.trustRating,
      willingness_to_use: answers.willingnessToUse,
      comments: answers.comments,
      completed_at: new Date().toISOString(),
    };

    await saveStudyResults(payload);
    setPhase('complete');
  }

  function handleComplete() {
    localStorage.setItem('xade-test-completed', 'true');
    onComplete();
  }

  // ---- Render ----
  if (phase === 'intro') return <IntroScreen onStart={() => setPhase('confidence')} />;

  if (phase === 'confidence')
    return (
      <ConfidenceScreen
        onNext={(rating) => {
          setSelfConfidence(rating);
          setPhase('classification');
        }}
      />
    );

  if (phase === 'classification')
    return (
      <ClassificationScreen
        image={images[currentImageIndex]}
        current={currentImageIndex + 1}
        total={images.length}
        onAnswer={handleClassificationAnswer}
      />
    );

  if (phase === 'analyzing')
    return <AnalyzingScreen done={analyzeProgress.done} total={analyzeProgress.total} />;

  if (phase === 'explanation' && explanationItems[currentExplanationIndex])
    return (
      <ExplanationScreen
        key={currentExplanationIndex}
        item={explanationItems[currentExplanationIndex]}
        current={currentExplanationIndex + 1}
        total={explanationItems.length}
        onSubmit={handleExplanationSubmit}
      />
    );

  if (phase === 'survey') return <SurveyScreen onSubmit={handleSurveySubmit} />;

  return (
    <CompleteScreen classificationRecords={classificationRecords} onContinue={handleComplete} />
  );
}
