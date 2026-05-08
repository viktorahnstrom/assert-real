import { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle, Loader2, XCircle } from 'lucide-react';
import {
  studyAnalyzeImage,
  saveStudyResults,
  type DetectionResult,
  type StudyAnalysisResult,
} from '@/lib/api';
import { AnalysisResultBody } from '@/components/AnalysisResultBody';

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
  { id: 1, url: '/quiz-images/sg3_psi070_seed0002378.webp', label: 'fake' },
  { id: 2, url: '/quiz-images/sg3_psi070_seed0002384.webp', label: 'fake' },
  { id: 3, url: '/quiz-images/sg3_psi070_seed0002763.webp', label: 'fake' },
  { id: 4, url: '/quiz-images/sg3_psi070_seed0003361.webp', label: 'fake' },
  { id: 5, url: '/quiz-images/sg3_psi070_seed0003379.webp', label: 'fake' },
  { id: 6, url: '/quiz-images/sg3_psi070_seed0003409.webp', label: 'fake' },
  { id: 7, url: '/quiz-images/00093.webp', label: 'real' },
  { id: 8, url: '/quiz-images/00408.webp', label: 'real' },
  { id: 9, url: '/quiz-images/00764.webp', label: 'real' },
  { id: 10, url: '/quiz-images/00818.webp', label: 'real' },
  { id: 11, url: '/quiz-images/01770.webp', label: 'real' },
  { id: 12, url: '/quiz-images/02213.webp', label: 'real' },
];

// Phase 3 retest images. Must be disjoint from ALL_IMAGES so participants
// see them for the first time after Phase 2 explanations.
const RETEST_IMAGES: StudyImage[] = [
  { id: 13, url: '/quiz-images/sg3_psi070_seed0001025.webp', label: 'fake' },
  { id: 14, url: '/quiz-images/sg3_psi070_seed0001242.webp', label: 'fake' },
  { id: 15, url: '/quiz-images/00999.webp', label: 'real' },
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
  | 'retest'
  | 'survey'
  | 'complete';

type UsefulComponent =
  | 'heatmap_only'
  | 'text_only'
  | 'facial_regions_only'
  | 'heatmap_text'
  | 'text_facial_regions'
  | 'heatmap_facial_regions'
  | 'everything';

const PHASE_2_MAX_IMAGES = 3;

interface ClassificationRecord {
  image: StudyImage;
  answer: 'real' | 'fake';
  isCorrect: boolean;
}

// Phase 3 — retest classification records. Same shape as
// ClassificationRecord; kept as a separate type so saveStudyResults can
// distinguish Phase 1 baseline answers from Phase 3 post-explanation
// answers when writing to Supabase.
//
// Schema agreement (#118 timer work will extend each entry):
//   { image_id, image_label, user_answer, is_correct, time_ms?, idle_discarded? }
interface RetestRecord {
  image: StudyImage;
  answer: 'real' | 'fake';
  isCorrect: boolean;
}

interface ExplanationItem {
  image: StudyImage;
  userAnswer: 'real' | 'fake';
  analysis: StudyAnalysisResult | null;
  provider: string;
  mostUsefulComponent: UsefulComponent | null;
  understandingRating: number | null;
  mostUsefulComment: string | null;
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
            This study tests how well AI explanations help people understand why an image is real or
            fake. It takes about <strong>10 to 15 minutes</strong>.
          </p>

          <div className="mt-6 space-y-4">
            {[
              {
                step: '1',
                title: 'Look at 12 images',
                desc: 'For each image, choose if it is real or fake. You get no hints.',
              },
              {
                step: '2',
                title: 'See the AI explanations',
                desc: 'For up to 3 images you got wrong, see how the AI explains them. Tell us what helped you understand best.',
              },
              {
                step: '3',
                title: 'Try 3 more images',
                desc: 'Now that you have seen our explanations, try classifying 3 new images.',
              },
              {
                step: '4',
                title: 'A few final questions',
                desc: 'Answer a few short questions about your experience.',
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
            <strong className="text-xade-charcoal/70">Privacy:</strong> Joining is voluntary and
            anonymous. We do not collect any personal information. Your answers are only used for
            research.
          </div>

          <button
            onClick={onStart}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            I understand. Start the study
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
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
            How good are you at spotting deepfakes?
          </h2>
          <p className="mt-2 text-sm text-xade-charcoal/50">
            Be honest. There are no right or wrong answers.
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
        <h2 className="mt-6 text-xl font-semibold text-xade-charcoal">Looking at your answers</h2>
        <p className="mt-2 text-sm text-xade-charcoal/50">
          Getting AI explanations ready for the images you got wrong.
          {total > 0 && (
            <>
              {' '}
              ({done}/{total} done)
            </>
          )}
        </p>
        <p className="mt-4 text-xs text-xade-charcoal/30">This can take up to a minute.</p>
      </div>
    </div>
  );
}

// ============================================
// Phase: Explanation evaluation (Phase 2)
// ============================================
function studyAnalysisToDetectionResult(item: ExplanationItem): DetectionResult {
  const { analysis, provider } = item;
  const fakeScore = analysis?.deepfake_score ?? 0;
  const explanation = analysis?.explanations[provider] ?? null;
  const prediction: 'fake' | 'real' = analysis?.classification === 'fake' ? 'fake' : 'real';
  const isFake = prediction === 'fake';

  return {
    prediction,
    confidence: isFake ? fakeScore : 1 - fakeScore,
    probabilities: { fake: fakeScore, real: 1 - fakeScore },
    model: 'EfficientNet-B4',
    accuracy: '98.48%',
    gradcam_heatmap_url: analysis?.gradcam_url ?? null,
    ela_heatmap_url: analysis?.ela_heatmap_url ?? null,
    evidence_regions: analysis?.evidence_regions ?? [],
    explanation: explanation
      ? {
          summary: explanation.summary,
          detailed_analysis: explanation.detailed_analysis,
          technical_notes: explanation.technical_notes,
          provider: explanation.provider,
          model: explanation.model,
          processing_time_ms: explanation.processing_time_ms,
          estimated_cost_usd: 0,
        }
      : null,
  };
}

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
    mostUsefulComponent: UsefulComponent;
    understandingRating: number;
    mostUsefulComment: string | null;
  }) => void;
}) {
  const [usefulComponent, setUsefulComponent] = useState<UsefulComponent | null>(null);
  const [understanding, setUnderstanding] = useState<number | null>(null);
  const [comment, setComment] = useState('');

  const canSubmit = usefulComponent !== null && understanding !== null;

  const detectionResult = studyAnalysisToDetectionResult(item);
  const userSaidLabel = item.userAnswer === 'fake' ? 'FAKE' : 'REAL';
  const actualLabel = item.image.label === 'fake' ? 'FAKE' : 'REAL';

  return (
    <div className="min-h-screen bg-xade-cream px-6 py-8">
      <div className="mx-auto max-w-4xl">
        {/* Progress */}
        <div className="mb-2 flex items-center justify-between text-xs text-xade-charcoal/40">
          <span>
            Explanation {current} of {total}
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
          Phase 2: See the AI Explanation
        </p>

        {/* Correction banner */}
        <div className="mb-5 flex items-center gap-3 rounded-xl border border-red-100 bg-red-50 px-5 py-4">
          <XCircle className="h-5 w-5 shrink-0 text-red-400" />
          <p className="text-sm text-red-700">
            You said this was <span className="font-bold">{userSaidLabel}</span>. It was actually{' '}
            <span
              className={`font-bold ${item.image.label === 'fake' ? 'text-red-600' : 'text-green-700'}`}
            >
              {actualLabel}
            </span>
            . Here is what our AI saw.
          </p>
        </div>

        {/* Shared layout from the production result view */}
        <AnalysisResultBody result={detectionResult} previewUrl={item.image.url} />

        {/* Questions */}
        <div className="mt-4 mb-6 rounded-2xl border border-xade-charcoal/6 bg-white px-6 py-6 shadow-sm">
          {/* Q1 */}
          <div className="mb-6">
            <p className="text-sm font-medium text-xade-charcoal">
              1. Which parts helped you understand the AI&apos;s choice best?
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {(
                [
                  { value: 'heatmap_only', label: 'Heatmap only' },
                  { value: 'text_only', label: 'Text only' },
                  { value: 'facial_regions_only', label: 'Facial regions only' },
                  { value: 'heatmap_text', label: 'Heatmap + Text' },
                  { value: 'text_facial_regions', label: 'Text + Facial regions' },
                  { value: 'heatmap_facial_regions', label: 'Heatmap + Facial regions' },
                  { value: 'everything', label: 'Everything together' },
                ] as const
              ).map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setUsefulComponent(value)}
                  className={`rounded-lg border-2 px-4 py-2 text-xs font-semibold transition-colors ${
                    usefulComponent === value
                      ? 'border-xade-blue bg-xade-blue text-white'
                      : 'border-xade-charcoal/15 text-xade-charcoal/60 hover:border-xade-blue/40'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Q2 */}
          <div className="mb-6">
            <p className="text-sm font-medium text-xade-charcoal">
              2. How well do you understand why the AI made this choice?
            </p>
            <div className="mt-3">
              <RatingButtons
                value={understanding}
                onChange={setUnderstanding}
                labels={['Not at all', 'Completely']}
              />
            </div>
          </div>

          {/* Q3 — optional free-text comment */}
          <div>
            <p className="text-sm font-medium text-xade-charcoal">
              3. Anything specific about why this combination helped most?{' '}
              <span className="font-normal text-xade-charcoal/50">(optional)</span>
            </p>
            <p className="mt-1 text-xs text-xade-charcoal/40">
              You can write in Swedish if you prefer.
            </p>
            <input
              type="text"
              value={comment}
              maxLength={200}
              onChange={(e) => setComment(e.target.value)}
              placeholder="e.g. the heatmap pointed straight at the eyes…"
              className="mt-3 w-full rounded-lg border-2 border-xade-charcoal/15 bg-white px-3 py-2 text-sm text-xade-charcoal placeholder:text-xade-charcoal/30 focus:border-xade-blue focus:outline-none"
            />
          </div>
        </div>

        <button
          disabled={!canSubmit}
          onClick={() => {
            if (!canSubmit) return;
            onSubmit({
              mostUsefulComponent: usefulComponent!,
              understandingRating: understanding!,
              mostUsefulComment: comment.trim() ? comment.trim() : null,
            });
            window.scrollTo({ top: 0, behavior: 'instant' });
          }}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark disabled:opacity-40"
        >
          {current < total ? 'Next image' : 'Continue to the final questions'}
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

// ============================================
// Phase: Retest classification (Phase 3)
// ============================================
function RetestScreen({
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
        <p className="mb-3 text-xs font-medium uppercase tracking-widest text-xade-blue/60">
          Phase 3: Try Again
        </p>
        {current === 1 && (
          <div className="mb-5 rounded-xl border border-xade-blue/20 bg-xade-blue/5 px-5 py-4 text-sm leading-relaxed text-xade-charcoal/70">
            Now that you have seen our explanations, try classifying these 3 new images.
          </div>
        )}

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
            alt={`Retest image ${current}`}
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
// Phase: Closing survey (Phase 4)
// ============================================
function SurveyScreen({
  didRetest,
  onSubmit,
}: {
  didRetest: boolean;
  onSubmit: (answers: {
    trustRating: number;
    willingnessToUse: string;
    explanationsHelpedInRetest: number | null;
    comments: string;
  }) => void;
}) {
  const [trust, setTrust] = useState<number | null>(null);
  const [willingness, setWillingness] = useState<string | null>(null);
  // Only required when the participant actually took Phase 3.
  const [helpedInRetest, setHelpedInRetest] = useState<number | null>(null);
  const [comments, setComments] = useState('');

  const canSubmit =
    trust !== null && willingness !== null && (!didRetest || helpedInRetest !== null);

  // Q-numbers shift by one when the retest question is hidden.
  const commentsQNumber = didRetest ? 4 : 3;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-lg">
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <p className="text-xs font-medium uppercase tracking-widest text-xade-blue/60">
            Phase 4: Final Questions
          </p>
          <h2 className="mt-2 text-xl font-semibold text-xade-charcoal">Almost done</h2>
          <p className="mt-1 text-sm text-xade-charcoal/50">
            A few last questions about your experience.
          </p>

          {/* Q1: Trust */}
          <div className="mt-8">
            <p className="text-sm font-medium text-xade-charcoal">
              1. How much do you trust XADE to detect deepfakes?
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
              2. Would you use a tool like XADE to check if images are deepfakes?
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

          {/* Q3: Did our explanations help in the retest? — only shown when
              the participant actually took Phase 3. */}
          {didRetest && (
            <div className="mt-6">
              <p className="text-sm font-medium text-xade-charcoal">
                3. Did our explanations help you in the second round of images?
              </p>
              <div className="mt-3">
                <RatingButtons
                  value={helpedInRetest}
                  onChange={setHelpedInRetest}
                  labels={['Not at all', 'Definitely']}
                />
              </div>
            </div>
          )}

          {/* Comments — Q3 if no retest, Q4 otherwise. */}
          <div className="mt-6">
            <p className="text-sm font-medium text-xade-charcoal">
              {commentsQNumber}. Any other comments? (optional)
            </p>
            <p className="mt-1 text-xs text-xade-charcoal/40">
              You can write in Swedish if you prefer.
            </p>
            <textarea
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Tell us what you thought about the explanations, the design, or anything else. Svenska går bra."
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
                explanationsHelpedInRetest: didRetest ? helpedInRetest : null,
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

// Production deploys the study-only experience. In that mode CompleteScreen
// is the terminal state — no Finish button, the participant just closes
// the window. Local dev with VITE_STUDY_ONLY=false keeps the button so
// the full app (auth + upload) stays reachable.
const STUDY_ONLY_MODE = import.meta.env.VITE_STUDY_ONLY !== 'false';

// ============================================
// Phase: Complete
// ============================================
function CompleteScreen({
  classificationRecords,
  retestRecords,
  onContinue,
}: {
  classificationRecords: ClassificationRecord[];
  retestRecords: RetestRecord[];
  onContinue: () => void;
}) {
  const correct = classificationRecords.filter((r) => r.isCorrect).length;
  const total = classificationRecords.length;
  const pct = Math.round((correct / total) * 100);

  const retestCorrect = retestRecords.filter((r) => r.isCorrect).length;
  const retestTotal = retestRecords.length;

  // In study-only mode the participant never clicks anything after this
  // screen, so persist completion automatically. Refreshing the page lands
  // them on ThankYouScreen instead of restarting the study.
  useEffect(() => {
    if (STUDY_ONLY_MODE) {
      localStorage.setItem('xade-test-completed', 'true');
    }
  }, []);

  let message: string;
  if (pct >= 80) {
    message = 'Great score. AI-generated deepfakes are getting harder to spot every day.';
  } else if (pct >= 50) {
    message = 'A solid result. As you saw, it is tricky. That is why tools like XADE exist.';
  } else {
    message = 'Most people find this hard. Deepfakes are made to fool the human eye.';
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
            <p className="mt-1 text-sm text-xade-charcoal/50">Your answers have been saved.</p>
          </div>

          <div className="mt-6 text-center">
            <p className="text-5xl font-bold text-xade-blue">
              {correct}/{total}
            </p>
            <p className="mt-1 text-sm text-xade-charcoal/50">{pct}% correct</p>
          </div>

          {retestTotal > 0 && (
            <div className="mt-5 rounded-lg bg-xade-blue/5 px-4 py-3 text-center">
              <p className="text-[11px] font-medium uppercase tracking-widest text-xade-blue/60">
                After our explanations
              </p>
              <p className="mt-1 text-2xl font-bold text-xade-blue">
                {retestCorrect}/{retestTotal}
              </p>
              <p className="mt-0.5 text-xs text-xade-charcoal/50">on 3 new images</p>
            </div>
          )}

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
                    was {r.image.label}, you said {r.answer}
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

          {STUDY_ONLY_MODE ? (
            <p className="mt-8 text-center text-sm text-xade-charcoal/60">
              You are done. You can now close this window.
            </p>
          ) : (
            <button
              onClick={onContinue}
              className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
            >
              Finish
              <ArrowRight className="h-4 w-4" />
            </button>
          )}
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

  // Phase 3 retest state. Shuffled per session so participants don't
  // discuss the order with each other. Only populated when the user had
  // at least one Phase 1 misclassification (see handleExplanationSubmit).
  const [retestImages] = useState<StudyImage[]>(() => shuffleArray(RETEST_IMAGES));
  const [currentRetestIndex, setCurrentRetestIndex] = useState(0);
  const [retestRecords, setRetestRecords] = useState<RetestRecord[]>([]);

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

    // Sample up to PHASE_2_MAX_IMAGES randomly so participants don't see all misclassified images
    const sample = shuffleArray(wrong).slice(0, PHASE_2_MAX_IMAGES);

    const hasPrecomputed = precomputed !== null && precomputed !== 'unavailable';

    if (!hasPrecomputed) {
      // Fallback: live analysis (slow — researcher should run /precompute first)
      setAnalyzeProgress({ done: 0, total: sample.length });
      setPhase('analyzing');
    }

    const allProviders = ['openai'];
    const items: ExplanationItem[] = [];

    for (const record of sample) {
      // Pick one provider at random; updated to best performer after #99 smoke test
      const provider = allProviders[Math.floor(Math.random() * allProviders.length)];

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
        provider,
        mostUsefulComponent: null,
        understandingRating: null,
        mostUsefulComment: null,
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
    mostUsefulComponent: UsefulComponent;
    understandingRating: number;
    mostUsefulComment: string | null;
  }) {
    const updated = explanationItems.map((item, i) =>
      i === currentExplanationIndex ? { ...item, ...answers } : item
    );
    setExplanationItems(updated);

    if (currentExplanationIndex + 1 < explanationItems.length) {
      setCurrentExplanationIndex(currentExplanationIndex + 1);
    } else {
      // Phase 2 done → Phase 3 retest. The retest only runs for
      // participants who had at least one Phase 1 misclassification,
      // which is enforced upstream in startAnalysis (it routes to
      // 'survey' directly when wrong.length === 0, so explanation phase
      // never starts and we never reach this code path).
      setPhase('retest');
    }
  }

  // ---- Retest answer (Phase 3) ----
  function handleRetestAnswer(answer: 'real' | 'fake') {
    const image = retestImages[currentRetestIndex];
    const isCorrect = answer === image.label;
    const newRecords = [...retestRecords, { image, answer, isCorrect }];
    setRetestRecords(newRecords);

    if (currentRetestIndex + 1 < retestImages.length) {
      setCurrentRetestIndex(currentRetestIndex + 1);
    } else {
      setPhase('survey');
    }
  }

  // ---- Survey submit ----
  async function handleSurveySubmit(answers: {
    trustRating: number;
    willingnessToUse: string;
    explanationsHelpedInRetest: number | null;
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
        provider: item.provider,
        most_useful_component: item.mostUsefulComponent,
        understanding_rating: item.understandingRating,
        most_useful_comment: item.mostUsefulComment,
      })),
      retest_answers: retestRecords.map((r) => ({
        image_id: r.image.id,
        image_label: r.image.label,
        user_answer: r.answer,
        is_correct: r.isCorrect,
      })),
      trust_rating: answers.trustRating,
      willingness_to_use: answers.willingnessToUse,
      explanations_helped_in_retest: answers.explanationsHelpedInRetest,
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

  if (phase === 'retest')
    return (
      <RetestScreen
        image={retestImages[currentRetestIndex]}
        current={currentRetestIndex + 1}
        total={retestImages.length}
        onAnswer={handleRetestAnswer}
      />
    );

  if (phase === 'survey')
    return <SurveyScreen didRetest={retestRecords.length > 0} onSubmit={handleSurveySubmit} />;

  return (
    <CompleteScreen
      classificationRecords={classificationRecords}
      retestRecords={retestRecords}
      onContinue={handleComplete}
    />
  );
}
