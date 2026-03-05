import { useRef, useState } from 'react';
import {
  BarChart3,
  HelpCircle,
  History,
  Upload,
  User,
  X,
  ChevronLeft,
  Settings,
  LogOut,
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
import {
  detectDeepfake,
  analyzeImage,
  type DetectionResult,
  type ApiError,
  type ApiMode,
} from '@/lib/api';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import AuthPage from '@/components/auth/AuthPage';

// ============================================
// Dev toolbar for endpoint / VLM switching
// ============================================

interface DevToolbarProps {
  apiMode: ApiMode;
  onApiModeChange: (mode: ApiMode) => void;
  vlmProvider: string;
  onVlmProviderChange: (provider: string) => void;
}

function DevToolbar({
  apiMode,
  onApiModeChange,
  vlmProvider,
  onVlmProviderChange,
}: DevToolbarProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {open && (
        <div className="mb-2 w-64 rounded-xl border border-xade-charcoal/10 bg-white p-4 shadow-lg">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-xade-charcoal/50">
              Dev Settings
            </p>
            <button
              onClick={() => setOpen(false)}
              className="text-xade-charcoal/30 hover:text-xade-charcoal"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* API Mode */}
          <div className="mb-3">
            <p className="mb-1.5 text-xs font-medium text-xade-charcoal/70">Endpoint</p>
            <div className="flex gap-1.5">
              <button
                onClick={() => onApiModeChange('detect')}
                className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  apiMode === 'detect'
                    ? 'bg-xade-blue text-white'
                    : 'bg-xade-charcoal/5 text-xade-charcoal/60 hover:bg-xade-charcoal/10'
                }`}
              >
                /detect
              </button>
              <button
                onClick={() => onApiModeChange('analyses')}
                className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  apiMode === 'analyses'
                    ? 'bg-xade-blue text-white'
                    : 'bg-xade-charcoal/5 text-xade-charcoal/60 hover:bg-xade-charcoal/10'
                }`}
              >
                /analyses
              </button>
            </div>
            <p className="mt-1 text-[10px] text-xade-charcoal/40">
              {apiMode === 'detect'
                ? 'Direct detection — no database'
                : 'Full flow — saves to Supabase'}
            </p>
          </div>

          {/* VLM Provider */}
          <div>
            <p className="mb-1.5 text-xs font-medium text-xade-charcoal/70">VLM Provider</p>
            <div className="grid grid-cols-2 gap-1.5">
              {(['openai', 'google', 'mock', 'none'] as const).map((provider) => (
                <button
                  key={provider}
                  onClick={() => onVlmProviderChange(provider)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    vlmProvider === provider
                      ? 'bg-xade-blue text-white'
                      : 'bg-xade-charcoal/5 text-xade-charcoal/60 hover:bg-xade-charcoal/10'
                  }`}
                >
                  {provider === 'none'
                    ? 'None'
                    : provider.charAt(0).toUpperCase() + provider.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen((prev) => !prev)}
        className={`flex h-10 w-10 items-center justify-center rounded-full shadow-lg transition-colors ${
          open
            ? 'bg-xade-blue text-white'
            : 'bg-white text-xade-charcoal/50 hover:text-xade-charcoal'
        }`}
      >
        <Settings className="h-4.5 w-4.5" />
      </button>
    </div>
  );
}

// ============================================
// Sidebar
// ============================================

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
  const { user, signOut } = useAuth();

  const displayName =
    user?.user_metadata?.display_name ||
    user?.user_metadata?.full_name ||
    user?.email ||
    'User';

  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-xade-charcoal/10">
        {user?.user_metadata?.avatar_url ? (
          <img
            src={user.user_metadata.avatar_url}
            alt=""
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <User className="h-4 w-4 text-xade-charcoal/70" />
        )}
      </div>
      {!isCollapsed && (
        <>
          <span className="flex-1 truncate text-sm font-medium text-xade-charcoal">
            {displayName}
          </span>
          <button
            onClick={signOut}
            title="Sign out"
            className="text-xade-charcoal/40 hover:text-xade-charcoal"
          >
            <LogOut className="h-4 w-4" />
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

// ============================================
// Upload View
// ============================================

interface UploadViewProps {
  onResult: (result: DetectionResult, previewUrl: string) => void;
  apiMode: ApiMode;
  vlmProvider: string;
}

function UploadView({ onResult, apiMode, vlmProvider }: UploadViewProps) {
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
      let data: DetectionResult;

      if (apiMode === 'analyses') {
        data = await analyzeImage(selectedFile, vlmProvider);
      } else {
        data = await detectDeepfake(selectedFile, vlmProvider);
      }

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
            <span>{error.type in errorMessages ? errorMessages[error.type] : error.message}</span>
            <button
              onClick={() => setError(null)}
              className="ml-3 shrink-0 text-red-400 hover:text-red-600"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="mt-3 flex items-center justify-center gap-2 text-xs text-xade-charcoal/30">
          <span
            className={`rounded px-1.5 py-0.5 ${
              apiMode === 'analyses' ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-600'
            }`}
          >
            {apiMode === 'analyses' ? 'DB Mode' : 'Direct Mode'}
          </span>
          <span>·</span>
          <span>{vlmProvider === 'none' ? 'No VLM' : vlmProvider}</span>
        </div>

        <div className="mt-4 flex justify-center">
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

// ============================================
// Result View
// ============================================

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

        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          {explanation ? (
            <>
              <p className="mb-3 text-sm font-medium text-xade-blue">AI Summary</p>
              <p className="text-sm leading-relaxed text-xade-charcoal/70">{explanation.summary}</p>
              <div className="mt-auto pt-4">
                <p className="text-xs text-xade-charcoal/30">
                  Generated by {explanation.model} in{' '}
                  {(explanation.processing_time_ms / 1000).toFixed(1)}s
                </p>
              </div>
            </>
          ) : (
            <p className="text-sm leading-relaxed text-xade-charcoal/40">
              No explanation available. Select a VLM provider in dev settings to enable AI-generated
              explanations.
            </p>
          )}
        </div>
      </div>

      {/* Middle row: Visual Analysis + Supporting Evidence */}
      <div className="mb-6 grid grid-cols-3 items-stretch gap-4">
        <div className="col-span-2 flex flex-col gap-4">
          {/* Visual Analysis */}
          <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
            <h2 className="mb-4 text-lg font-semibold text-xade-blue">Visual Analysis</h2>
            <div className="grid grid-cols-2 gap-4">
              {/* Original image */}
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

              {/* GradCAM heatmap */}
              <div>
                <p className="mb-2 text-xs text-xade-charcoal/50">
                  Detection Heatmap
                  <span className="ml-1 text-xade-charcoal/30">· red = high focus</span>
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
                      className="h-full w-full object-cover transition-opacity hover:opacity-80"
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
                    GradCAM unavailable
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Why this decision */}
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

        {/* Supporting Evidence */}
        <div className="flex flex-1 flex-col rounded-xl bg-white p-6 shadow-md">
          <h2 className="mb-4 text-lg font-semibold text-xade-blue">Supporting Evidence</h2>
          {result.evidence_regions && result.evidence_regions.length > 0 ? (
            <div className="space-y-4 overflow-y-auto">
              {result.evidence_regions.map((region, i) => (
                <div key={i}>
                  <div
                    className="w-full cursor-zoom-in overflow-hidden rounded-lg"
                    style={{ aspectRatio: '1 / 1' }}
                    onClick={() => setLightboxSrc(region.url)}
                  >
                    <img
                      src={region.url}
                      alt={region.label}
                      className="h-full w-full object-cover transition-opacity hover:opacity-80"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                  <p className="mt-1.5 text-xs font-medium text-xade-charcoal/70">{region.label}</p>
                  <p className="text-xs text-xade-charcoal/30">
                    Activation strength: {Math.round(region.activation_score * 100)}%
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <div className="rounded-lg bg-xade-charcoal/5 p-6">
                <p className="text-sm text-xade-charcoal/40">
                  No high-activation regions detected in this image.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      <TechnicalDetails result={result} isFake={isFake} />
    </div>
  );
}

// ============================================
// Main Content & App
// ============================================

function MainContent() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [apiMode, setApiMode] = useState<ApiMode>('detect');
  const [vlmProvider, setVlmProvider] = useState<string>('openai');

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
        <UploadView onResult={handleResult} apiMode={apiMode} vlmProvider={vlmProvider} />
      )}
      <DevToolbar
        apiMode={apiMode}
        onApiModeChange={setApiMode}
        vlmProvider={vlmProvider}
        onVlmProviderChange={setVlmProvider}
      />
    </SidebarInset>
  );
}

function AuthenticatedApp() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <MainContent />
    </SidebarProvider>
  );
}

function AppRouter() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-xade-cream">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-xade-blue">XADE</h1>
          <p className="mt-2 text-sm text-xade-charcoal/40">Loading…</p>
        </div>
      </div>
    );
  }

  return user ? <AuthenticatedApp /> : <AuthPage />;
}

function App() {
  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  );
}

export default App;