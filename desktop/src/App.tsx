import { useRef, useState } from 'react';
import {
  BarChart3,
  HelpCircle,
  History,
  Upload,
  User,
  MoreVertical,
  X,
  ChevronLeft,
} from 'lucide-react';
import {
  Button,
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarItem,
  SidebarTrigger,
  SidebarInset,
  useSidebar,
} from '@/components/ui';
import { detectDeepfake, type DetectionResult, type ApiError } from '@/lib/api';

function SidebarLogo() {
  const { isCollapsed } = useSidebar();
  return (
    <div className="flex items-center gap-2">
      {!isCollapsed && <span className="text-lg font-semibold text-xade-blue">XADE</span>}
    </div>
  );
}

function UserProfile() {
  const { isCollapsed } = useSidebar();
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-xade-charcoal/10">
        <User className="h-4 w-4 text-xade-charcoal/70" />
      </div>
      {!isCollapsed && (
        <>
          <span className="flex-1 text-sm font-medium text-xade-charcoal">John Doe</span>
          <button className="text-xade-charcoal/50 hover:text-xade-charcoal">
            <MoreVertical className="h-4 w-4" />
          </button>
        </>
      )}
    </div>
  );
}

function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>
        <SidebarLogo />
        <SidebarTrigger />
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarItem icon={<BarChart3 />}>Statistics</SidebarItem>
          <SidebarItem icon={<HelpCircle />}>Support</SidebarItem>
          <SidebarItem icon={<History />}>History</SidebarItem>
        </SidebarGroup>
        <SidebarGroup label="Recent">
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <UserProfile />
      </SidebarFooter>
    </Sidebar>
  );
}

interface UploadViewProps {
  onResult: (result: DetectionResult, previewUrl: string) => void;
}

function UploadView({ onResult }: UploadViewProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading'>('idle');
  const [error, setError] = useState<ApiError | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
  }

  function handleClear() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setStatus('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  async function handleSubmit() {
    if (!selectedFile || !previewUrl) return;
    setStatus('loading');
    setError(null);
    try {
      const data = await detectDeepfake(selectedFile);
      onResult(data, previewUrl);
    } catch (err) {
      setError(err as ApiError);
      setStatus('idle');
    }
  }

  const errorMessages: Record<ApiError['type'], string> = {
    network: '⚡ Backend offline — start the FastAPI server on port 8000.',
    invalid_file: '📁 Invalid file — please upload a JPG or PNG image.',
    model_unavailable: '🤖 Detection model not loaded — check backend logs.',
    unknown: '❌ Something went wrong. Please try again.',
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="mb-8 text-center">
        <div className="mb-2 h-px w-48 bg-xade-charcoal/20" />
        <h1 className="text-7xl font-bold tracking-tight text-xade-blue">XADE</h1>
        <div className="mt-2 h-px w-48 bg-xade-charcoal/20" />
      </div>

      <div className="w-full max-w-md">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        <div
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-xade-blue/30 bg-white p-12 transition-colors hover:border-xade-blue/50"
        >
          {previewUrl ? (
            <>
              <img
                src={previewUrl}
                alt="Selected"
                className="max-h-48 max-w-full rounded object-contain"
              />
              <p className="mt-3 text-sm text-xade-charcoal/50">{selectedFile?.name}</p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleClear();
                }}
                className="absolute right-3 top-3 text-xade-charcoal/30 hover:text-xade-charcoal/70"
              >
                <X className="h-4 w-4" />
              </button>
            </>
          ) : (
            <>
              <Upload className="mb-4 h-16 w-16 text-xade-blue/50" strokeWidth={1} />
              <p className="mb-1 text-lg font-medium text-xade-charcoal">
                Drag and drop or click here
              </p>
              <p className="text-sm text-xade-charcoal/50">to upload your image (max 2mb)</p>
            </>
          )}
        </div>

        {error && (
          <div className="mt-4 flex items-start justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <span>{errorMessages[error.type]}</span>
            <button
              onClick={() => setError(null)}
              className="ml-3 shrink-0 text-red-400 hover:text-red-600"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="mt-6 flex justify-center">
          <Button
            variant="outline"
            className="min-w-32"
            onClick={handleSubmit}
            disabled={!selectedFile || status === 'loading'}
          >
            {status === 'loading' ? 'Analysing…' : 'Submit'}
          </Button>
        </div>
      </div>
    </div>
  );
}

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={onClose}
    >
      <button onClick={onClose} className="absolute right-6 top-6 text-white/70 hover:text-white">
        <X className="h-6 w-6" />
      </button>
      <img
        src={src}
        alt="Full size"
        className="max-h-[90vh] max-w-[90vw] rounded-xl object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

function TechnicalDetails({ result, isFake }: { result: DetectionResult; isFake: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl bg-white shadow-md">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between px-6 py-4 text-left"
      >
        <h2 className="text-lg font-semibold text-xade-blue">Technical Details</h2>
        <ChevronLeft
          className={`h-4 w-4 text-xade-charcoal/40 transition-transform duration-200 ${
            open ? '-rotate-90' : 'rotate-180'
          }`}
        />
      </button>

      {open && (
        <div className="border-t border-xade-charcoal/10 px-6 py-5">
          <div className="grid grid-cols-3 gap-4 text-sm">
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

          {result.explanation?.technical_notes && (
            <div className="mt-4 border-t border-xade-charcoal/10 pt-4">
              <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">
                Technical Notes
              </p>
              <p className="mt-2 text-sm leading-relaxed text-xade-charcoal/70">
                {result.explanation.technical_notes}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ResultViewProps {
  result: DetectionResult;
  previewUrl: string;
  onBack: () => void;
}

function ResultView({ result, previewUrl, onBack }: ResultViewProps) {
  const isFake = result.prediction === 'fake';
  const confidencePct = Math.round(result.confidence * 100);
  const fakePct = Math.round(result.probabilities.fake * 100);
  const realPct = Math.round(result.probabilities.real * 100);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  const explanation = result.explanation;

  return (
    <div className="min-h-screen bg-white px-24 py-10 max-w-5xl mx-auto">
      {lightboxSrc && <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}

      <button
        onClick={onBack}
        className="mb-6 flex items-center gap-1 text-sm text-xade-charcoal/50 hover:text-xade-charcoal"
      >
        <ChevronLeft className="h-4 w-4" />
        Back
      </button>

      {/* Top row: Confidence + Summary */}
      <div className="mb-6 grid grid-cols-2 gap-4">
        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          <p className={`mb-1 text-6xl font-bold ${isFake ? 'text-red-500' : 'text-green-500'}`}>
            {confidencePct}%
          </p>
          <p className={`text-lg font-semibold ${isFake ? 'text-red-500' : 'text-green-500'}`}>
            {isFake ? 'Deepfake' : 'Authentic'}
          </p>
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-xade-charcoal/40">
              <span>Fake</span>
              <span>Real</span>
            </div>
            <div className="relative h-2 w-full rounded-full bg-xade-charcoal/10">
              <div
                className="absolute left-0 top-0 h-2 rounded-full bg-red-400"
                style={{ width: `${fakePct}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between text-xs text-xade-charcoal/40">
              <span>{fakePct}%</span>
              <span>{realPct}%</span>
            </div>
          </div>
        </div>

        {/* Summary card — VLM explanation summary */}
        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          {explanation ? (
            <>
              <p className="mb-3 text-sm font-medium text-xade-blue">AI Summary</p>
              <p className="text-sm leading-relaxed text-xade-charcoal/70">
                {explanation.summary}
              </p>
              <div className="mt-auto pt-4">
                <p className="text-xs text-xade-charcoal/30">
                  Generated by {explanation.model} in{' '}
                  {(explanation.processing_time_ms / 1000).toFixed(1)}s
                </p>
              </div>
            </>
          ) : (
            <p className="text-sm leading-relaxed text-xade-charcoal/40">
              No explanation available. The VLM service may not be configured.
            </p>
          )}
        </div>
      </div>

      {/* Middle row: Visual Analysis + Detailed Analysis */}
      <div className="mb-6 grid grid-cols-3 items-stretch gap-4">
        <div className="col-span-2 flex flex-col gap-4">
          {/* Visual Analysis */}
          <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
            <h2 className="mb-4 text-lg font-semibold text-xade-blue">Visual Analysis</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="mb-2 text-xs text-xade-charcoal/50">Original Image</p>
                <div
                  className="w-full cursor-zoom-in overflow-hidden rounded-lg"
                  style={{ aspectRatio: '1 / 1' }}
                  onClick={() => setLightboxSrc(previewUrl)}
                >
                  <img
                    src={previewUrl}
                    alt="Original"
                    className="h-full w-full object-cover transition-opacity hover:opacity-80"
                  />
                </div>
              </div>
              <div>
                <p className="mb-2 text-xs text-xade-charcoal/50">Detection Heatmap</p>
                <div
                  className="flex w-full items-center justify-center rounded-lg bg-xade-charcoal/5 text-xs text-xade-charcoal/30"
                  style={{ aspectRatio: '1 / 1' }}
                >
                  GradCAM — coming soon
                </div>
              </div>
            </div>
          </div>

          {/* Why this decision — VLM detailed analysis */}
          <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
            <h2 className="mb-2 text-lg font-semibold text-xade-blue">Why this decision?</h2>
            {explanation ? (
              <p className="text-sm leading-relaxed text-xade-charcoal/70">
                {explanation.detailed_analysis}
              </p>
            ) : (
              <p className="text-sm leading-relaxed text-xade-charcoal/40">
                Detailed analysis is not available. Enable a VLM provider to see AI-generated
                explanations grounded in the detection heatmap.
              </p>
            )}
          </div>
        </div>

        {/* Supporting Evidence — placeholder for future GradCAM regions */}
        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          <h2 className="mb-4 text-lg font-semibold text-xade-blue">Supporting Evidence</h2>
          <div className="flex flex-1 flex-col items-center justify-center text-center">
            <div className="rounded-lg bg-xade-charcoal/5 p-6">
              <p className="text-sm text-xade-charcoal/40">
                Visual evidence regions will appear here once GradCAM heatmap integration is
                complete.
              </p>
            </div>
          </div>
        </div>
      </div>

      <TechnicalDetails result={result} isFake={isFake} />
    </div>
  );
}

function MainContent() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  function handleResult(data: DetectionResult, url: string) {
    setResult(data);
    setPreviewUrl(url);
  }

  function handleBack() {
    setResult(null);
    setPreviewUrl(null);
  }

  return (
    <SidebarInset>
      {result && previewUrl ? (
        <ResultView result={result} previewUrl={previewUrl} onBack={handleBack} />
      ) : (
        <UploadView onResult={handleResult} />
      )}
    </SidebarInset>
  );
}

function App() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <MainContent />
    </SidebarProvider>
  );
}

export default App;
