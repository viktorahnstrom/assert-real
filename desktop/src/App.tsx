import { useRef, useState } from 'react';
import { BarChart3, HelpCircle, History, Upload, User, MoreVertical, X } from 'lucide-react';
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

function ResultCard({ result }: { result: DetectionResult }) {
  const isFake = result.prediction === 'fake';
  const confidencePct = Math.round(result.confidence * 100);
  const fakePct = Math.round(result.probabilities.fake * 100);
  const realPct = Math.round(result.probabilities.real * 100);

  return (
    <div
      className={`mt-6 w-full max-w-md rounded-lg border-2 p-6 ${
        isFake ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="text-sm font-medium text-xade-charcoal/60 uppercase tracking-wide">
          Verdict
        </span>
        <span
          className={`text-xl font-bold ${isFake ? 'text-red-600' : 'text-green-600'}`}
        >
          {isFake ? 'DEEPFAKE' : 'REAL'}
        </span>
      </div>

      <div className="mb-4">
        <div className="mb-1 flex justify-between text-xs text-xade-charcoal/60">
          <span>Confidence</span>
          <span>{confidencePct}%</span>
        </div>
        <div className="h-2 w-full rounded-full bg-xade-charcoal/10">
          <div
            className={`h-2 rounded-full transition-all ${isFake ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      <div className="flex justify-between text-sm">
        <div className="text-center">
          <p className="font-semibold text-red-600">{fakePct}%</p>
          <p className="text-xs text-xade-charcoal/50">Fake probability</p>
        </div>
        <div className="text-center">
          <p className="font-semibold text-green-600">{realPct}%</p>
          <p className="text-xs text-xade-charcoal/50">Real probability</p>
        </div>
      </div>

      {/* Model info */}
      <div className="mt-4 border-t border-xade-charcoal/10 pt-3 text-xs text-xade-charcoal/40">
        {result.model} · Trained accuracy {result.accuracy}
      </div>
    </div>
  );
}

function ErrorBanner({ error, onDismiss }: { error: ApiError; onDismiss: () => void }) {
  const messages: Record<ApiError['type'], string> = {
    network: '⚡ Backend offline — start the FastAPI server on port 8000.',
    invalid_file: '📁 Invalid file — please upload a JPG or PNG image.',
    model_unavailable: '🤖 Detection model not loaded — check backend logs.',
    unknown: '❌ Something went wrong. Please try again.',
  };

  return (
    <div className="mt-4 flex w-full max-w-md items-start justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <span>{messages[error.type]}</span>
      <button onClick={onDismiss} className="ml-3 shrink-0 text-red-400 hover:text-red-600">
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

type Status = 'idle' | 'loading' | 'success' | 'error';

function MainContent() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setStatus('idle');
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setStatus('idle');
  }

  function handleClear() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setStatus('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  async function handleSubmit() {
    if (!selectedFile) return;
    setStatus('loading');
    setResult(null);
    setError(null);

    try {
      const data = await detectDeepfake(selectedFile);
      setResult(data);
      setStatus('success');
    } catch (err) {
      setError(err as ApiError);
      setStatus('error');
    }
  }

  return (
    <SidebarInset>
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
                  onClick={(e) => { e.stopPropagation(); handleClear(); }}
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

          {error && <ErrorBanner error={error} onDismiss={() => setError(null)} />}

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

          {result && <ResultCard result={result} />}
        </div>
      </div>
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