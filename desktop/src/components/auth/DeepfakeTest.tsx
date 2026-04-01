import { useState } from 'react';
import { ArrowRight, CheckCircle, XCircle } from 'lucide-react';

// ============================================
// Placeholder images — swap these out later
// ============================================
interface TestImage {
  id: number;
  url: string;
  label: 'real' | 'fake';
}

const TEST_IMAGES: TestImage[] = [
  { id: 1, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+1', label: 'real' },
  { id: 2, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+2', label: 'fake' },
  { id: 3, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+3', label: 'real' },
  { id: 4, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+4', label: 'fake' },
  { id: 5, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+5', label: 'real' },
  { id: 6, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+6', label: 'fake' },
  { id: 7, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+7', label: 'real' },
  { id: 8, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+8', label: 'fake' },
  { id: 9, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+9', label: 'real' },
  { id: 10, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+10', label: 'fake' },
  { id: 11, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+11', label: 'real' },
  { id: 12, url: 'https://placehold.co/600x600/e2e8f0/475569?text=Image+12', label: 'fake' },
];

// ============================================
// Intro Screen
// ============================================
function IntroScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="max-w-md text-center">
        <h1 className="text-4xl font-bold tracking-tight text-xade-blue">XADE</h1>
        <p className="mt-1 text-[11px] font-medium uppercase tracking-[0.2em] text-xade-charcoal/40">
          Deepfake Detection
        </p>

        <div className="mt-10 rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          <h2 className="text-xl font-semibold text-xade-charcoal">
            Can you spot a deepfake?
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-xade-charcoal/60">
            You&apos;ll be shown 12 images - some are real and some are AI-generated
            deepfakes. For each one, decide if it&apos;s <strong>Real</strong> or{' '}
            <strong>Fake</strong>. Let&apos;s see how well you do.
          </p>

          <button
            onClick={onStart}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            Take the test
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>

        <button
          onClick={() => {
            localStorage.setItem('xade-test-completed', 'skipped');
            window.location.reload();
          }}
          className="mt-4 text-xs text-xade-charcoal/30 hover:text-xade-charcoal/50"
        >
          Skip for now
        </button>
      </div>
    </div>
  );
}

// ============================================
// Question Screen
// ============================================
interface QuestionScreenProps {
  image: TestImage;
  current: number;
  total: number;
  onAnswer: (answer: 'real' | 'fake') => void;
}

function QuestionScreen({ image, current, total, onAnswer }: QuestionScreenProps) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-lg">
        {/* Progress bar */}
        <div className="mb-2 flex items-center justify-between text-xs text-xade-charcoal/40">
          <span>
            {current} of {total}
          </span>
          <span>{Math.round((current / total) * 100)}%</span>
        </div>
        <div className="mb-8 h-1.5 w-full rounded-full bg-xade-charcoal/10">
          <div
            className="h-1.5 rounded-full bg-xade-blue transition-all duration-300"
            style={{ width: `${(current / total) * 100}%` }}
          />
        </div>

        {/* Image */}
        <div className="overflow-hidden rounded-2xl border border-xade-charcoal/6 bg-white shadow-lg shadow-xade-charcoal/4">
          <img
            src={image.url}
            alt={`Test image ${current}`}
            className="aspect-square w-full object-cover"
          />
        </div>

        {/* Buttons */}
        <div className="mt-6 flex gap-3">
          <button
            onClick={() => onAnswer('real')}
            className="flex-1 rounded-lg border-2 border-green-200 bg-white px-6 py-3.5 text-sm font-semibold text-green-600 transition-colors hover:bg-green-50 hover:border-green-400"
          >
            Real
          </button>
          <button
            onClick={() => onAnswer('fake')}
            className="flex-1 rounded-lg border-2 border-red-200 bg-white px-6 py-3.5 text-sm font-semibold text-red-500 transition-colors hover:bg-red-50 hover:border-red-400"
          >
            Fake
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Results Screen
// ============================================
interface ResultsScreenProps {
  answers: { image: TestImage; answer: 'real' | 'fake' }[];
  onContinue: () => void;
}

function ResultsScreen({ answers, onContinue }: ResultsScreenProps) {
  const correct = answers.filter((a) => a.answer === a.image.label).length;
  const total = answers.length;
  const percentage = Math.round((correct / total) * 100);

  let message: string;
  if (percentage >= 80) {
    message = "Impressive! But AI-generated deepfakes are getting harder to detect every day.";
  } else if (percentage >= 50) {
    message = "Not bad - but as you can see, it's tricky. That's exactly why tools like XADE exist.";
  } else {
    message = "Don't worry - most people struggle with this. Deepfakes are designed to fool the human eye. Let XADE help.";
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-xade-cream p-8">
      <div className="w-full max-w-md">
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          {/* Score */}
          <div className="text-center">
            <p className="text-6xl font-bold text-xade-blue">
              {correct}/{total}
            </p>
            <p className="mt-1 text-sm text-xade-charcoal/50">
              {percentage}% correct
            </p>
          </div>

          <p className="mt-6 text-center text-sm leading-relaxed text-xade-charcoal/60">
            {message}
          </p>

          {/* Answer breakdown */}
          <div className="mt-6 space-y-2">
            {answers.map((a, i) => {
              const isCorrect = a.answer === a.image.label;
              return (
                <div
                  key={a.image.id}
                  className="flex items-center justify-between rounded-lg bg-xade-charcoal/3 px-3 py-2 text-xs"
                >
                  <span className="text-xade-charcoal/50">Image {i + 1}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xade-charcoal/40">
                      was {a.image.label} · you said {a.answer}
                    </span>
                    {isCorrect ? (
                      <CheckCircle className="h-3.5 w-3.5 text-green-500" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5 text-red-400" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Continue */}
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
// Main Test Component
// ============================================
type TestPhase = 'intro' | 'testing' | 'results';

interface DeepfakeTestProps {
  onComplete: () => void;
}

export default function DeepfakeTest({ onComplete }: DeepfakeTestProps) {
  const [phase, setPhase] = useState<TestPhase>('intro');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<{ image: TestImage; answer: 'real' | 'fake' }[]>([]);

  function handleStart() {
    setPhase('testing');
  }

  function handleAnswer(answer: 'real' | 'fake') {
    const newAnswers = [...answers, { image: TEST_IMAGES[currentIndex], answer }];
    setAnswers(newAnswers);

    if (currentIndex + 1 < TEST_IMAGES.length) {
      setCurrentIndex(currentIndex + 1);
    } else {
      setPhase('results');
    }
  }

  function handleContinue() {
    localStorage.setItem('xade-test-completed', 'true');
    onComplete();
  }

  if (phase === 'intro') {
    return <IntroScreen onStart={handleStart} />;
  }

  if (phase === 'testing') {
    return (
      <QuestionScreen
        image={TEST_IMAGES[currentIndex]}
        current={currentIndex + 1}
        total={TEST_IMAGES.length}
        onAnswer={handleAnswer}
      />
    );
  }

  return <ResultsScreen answers={answers} onContinue={handleContinue} />;
}