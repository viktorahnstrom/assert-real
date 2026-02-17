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
      <button
        onClick={onClose}
        className="absolute right-6 top-6 text-white/70 hover:text-white"
      >
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

  const evidenceItems = [
    'Irregular blending at jawline boundary showing pixel discontinuities',
    'Unnatural eye reflection pattern inconsistent with light source',
    'Frequency domain anomaly visualization showing GAN artifacts',
  ];

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

          <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          <p className="text-sm leading-relaxed text-xade-charcoal/70">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
            exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
          </p>
        </div>
      </div>

      <div className="mb-6 grid grid-cols-3 items-stretch gap-4">

        <div className="col-span-2 flex flex-col gap-4">

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
                  className="w-full rounded-lg bg-xade-charcoal/5"
                  style={{ aspectRatio: '1 / 1' }}
                />
              </div>
            </div>
          </div>

          <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
            <h2 className="mb-2 text-lg font-semibold text-xade-blue">Why this decision?</h2>
            <p className="mb-4 text-sm leading-relaxed text-xade-charcoal/70">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
              incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
              exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure
              dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
            </p>
            <p className="mb-3 text-sm font-medium text-xade-charcoal">Key Findings:</p>
            <div className="space-y-2">
              {[
                'Irregular facial boundaries detected in the cheek and jawline region, showing pixel-level discontinuities typical of neural network-based face swapping.',
                "Unnatural lighting inconsistencies between face and background, with shadow angles that don't align with the apparent light source.",
                'Frequency domain anomalies in high-frequency components, showing patterns consistent with upsampling artifacts from generative models.',
              ].map((finding, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-xade-charcoal/5 px-4 py-3 text-sm text-xade-charcoal/70"
                >
                  • {finding}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          <h2 className="mb-4 text-lg font-semibold text-xade-blue">Supporting Evidence</h2>
          <div className="space-y-4">
            {evidenceItems.map((caption, i) => (
              <div key={i}>
                <div
                  className="w-full cursor-pointer overflow-hidden rounded-lg bg-xade-charcoal/5"
                  style={{ aspectRatio: '1 / 1' }}
                />
                <p className="mt-1 text-xs text-xade-charcoal/50">{caption}</p>
              </div>
            ))}
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