import { useEffect, useRef, useState } from 'react';
import {
  BarChart3,
  ChevronLeft,
  History,
  LogOut,
  Plus,
  RotateCcw,
  Settings,
  Trash2,
  Upload,
  User,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Button,
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarItem,
  SidebarTrigger,
  SidebarInset,
  useSidebar,
} from '@/components/ui';
import {
  detectDeepfake,
  analyzeImage,
  fetchUserAnalyses,
  fetchUserImages,
  deleteAnalysis,
  type DetectionResult,
  type AnalysisResult,
  type ImageRecord,
  type ApiError,
  type ApiMode,
  type ExplanationResult,
} from '@/lib/api';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import AuthPage from '@/components/auth/AuthPage';
import DeepfakeTest from '@/components/auth/DeepfakeTest';

type AppView = 'upload' | 'result' | 'history' | 'statistics';

// ============================================
// Utilities
// ============================================

function formatRelativeDate(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

function groupAnalysesByDate(
  analyses: AnalysisResult[]
): { label: string; items: AnalysisResult[] }[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  const buckets: Record<string, AnalysisResult[]> = {
    Today: [],
    Yesterday: [],
    'Last 7 days': [],
    Older: [],
  };

  for (const a of analyses) {
    const d = new Date(a.created_at);
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    if (day >= today) buckets['Today'].push(a);
    else if (day >= yesterday) buckets['Yesterday'].push(a);
    else if (day >= lastWeek) buckets['Last 7 days'].push(a);
    else buckets['Older'].push(a);
  }

  return Object.entries(buckets)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }));
}

function analysisToDetectionResult(analysis: AnalysisResult): DetectionResult {
  const fakeScore = analysis.deepfake_score ?? 0;
  const isFake = analysis.classification === 'fake';
  return {
    prediction: (analysis.classification as 'fake' | 'real') ?? 'real',
    confidence: isFake ? fakeScore : 1 - fakeScore,
    probabilities: { fake: fakeScore, real: 1 - fakeScore },
    model: analysis.model_used ?? 'EfficientNet-B4',
    accuracy: '98.48%',
    gradcam_heatmap_url: analysis.gradcam_heatmap_url ?? null,
    explanation: analysis.explanation ?? null,
    evidence_regions: analysis.evidence_regions ?? [],
  };
}

function getAnalysisImageUrl(
  analysis: AnalysisResult,
  imageMap: Record<string, ImageRecord>
): string | null {
  const record = imageMap[analysis.image_id];
  if (record?.url) return record.url;
  if (analysis.gradcam_heatmap_url) return analysis.gradcam_heatmap_url;
  return null;
}

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

          <div>
            <p className="mb-1.5 text-xs font-medium text-xade-charcoal/70">VLM Provider</p>
            <div className="grid grid-cols-2 gap-1.5">
              {(
                [
                  { id: 'openai', label: 'OpenAI' },
                  { id: 'google', label: 'Google' },
                  { id: 'rule_based', label: 'Rule-Based' },
                  { id: 'mock', label: 'Mock' },
                  { id: 'none', label: 'None' },
                ] as const
              ).map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => onVlmProviderChange(id)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    vlmProvider === id
                      ? 'bg-xade-blue text-white'
                      : 'bg-xade-charcoal/5 text-xade-charcoal/60 hover:bg-xade-charcoal/10'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          {/* Reset Quiz */}
          <div className="mt-3 border-t border-xade-charcoal/10 pt-3">
            <button
              onClick={() => {
                localStorage.removeItem('xade-test-completed');
                window.location.reload();
              }}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-xade-charcoal/5 px-3 py-1.5 text-xs font-medium text-xade-charcoal/60 hover:bg-xade-charcoal/10"
            >
              <RotateCcw className="h-3 w-3" />
              Reset Quiz
            </button>
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
    user?.user_metadata?.display_name || user?.user_metadata?.full_name || user?.email || 'User';

  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-xade-charcoal/10">
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

interface AnalysisSidebarItemProps {
  analysis: AnalysisResult;
  imageUrl: string | null;
  isActive: boolean;
  onClick: () => void;
}

function AnalysisSidebarItem({ analysis, imageUrl, isActive, onClick }: AnalysisSidebarItemProps) {
  const { isCollapsed } = useSidebar();
  const isFake = analysis.classification === 'fake';
  const fakeScore = analysis.deepfake_score ?? 0;
  const confidence =
    analysis.status === 'completed' ? Math.round((isFake ? fakeScore : 1 - fakeScore) * 100) : null;

  return (
    <button
      onClick={onClick}
      title={
        analysis.status === 'completed'
          ? `${isFake ? 'Deepfake' : 'Authentic'} · ${confidence}%`
          : analysis.status
      }
      className={cn(
        'flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors',
        'hover:bg-xade-charcoal/5',
        isActive && 'bg-xade-blue/10',
        isCollapsed && 'justify-center px-2'
      )}
    >
      {/* Thumbnail */}
      <div className="h-9 w-9 shrink-0 overflow-hidden rounded-md">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt=""
            className="h-full w-full object-cover"
            onError={(e) => {
              const el = e.target as HTMLImageElement;
              el.style.display = 'none';
              if (el.parentElement) {
                el.parentElement.classList.add(
                  'flex',
                  'items-center',
                  'justify-center',
                  isFake ? 'bg-red-100' : 'bg-green-100'
                );
              }
            }}
          />
        ) : (
          <div
            className={cn(
              'flex h-full w-full items-center justify-center',
              analysis.status === 'completed'
                ? isFake
                  ? 'bg-red-100'
                  : 'bg-green-100'
                : 'bg-xade-charcoal/5'
            )}
          >
            <div
              className={cn(
                'h-2 w-2 rounded-full',
                analysis.status === 'completed'
                  ? isFake
                    ? 'bg-red-400'
                    : 'bg-green-400'
                  : 'bg-xade-charcoal/20'
              )}
            />
          </div>
        )}
      </div>

      {!isCollapsed && (
        <div className="min-w-0 flex-1">
          <p
            className={cn(
              'truncate text-xs font-medium',
              analysis.status === 'completed'
                ? isFake
                  ? 'text-red-600'
                  : 'text-green-600'
                : 'text-xade-charcoal/50'
            )}
          >
            {analysis.status === 'completed'
              ? `${isFake ? 'Deepfake' : 'Authentic'}${confidence !== null ? ` · ${confidence}%` : ''}`
              : analysis.status}
          </p>
          <p className="truncate text-[10px] text-xade-charcoal/40">
            {formatRelativeDate(analysis.created_at)}
          </p>
        </div>
      )}
    </button>
  );
}

interface AppSidebarProps {
  activeView: AppView;
  analyses: AnalysisResult[];
  imageMap: Record<string, ImageRecord>;
  selectedAnalysisId: string | null;
  onNavigate: (view: AppView) => void;
  onSelectAnalysis: (analysis: AnalysisResult) => void;
  onNewAnalysis: () => void;
}

function AppSidebar({
  activeView,
  analyses,
  imageMap,
  selectedAnalysisId,
  onNavigate,
  onSelectAnalysis,
  onNewAnalysis,
}: AppSidebarProps) {
  const { isCollapsed } = useSidebar();
  const groups = groupAnalysesByDate(analyses);

  return (
    <Sidebar>
      <SidebarHeader>
        <SidebarLogo />
        <SidebarTrigger />
      </SidebarHeader>

      {/* New Analysis button */}
      <div className={cn('px-3 pb-3', isCollapsed && 'px-2')}>
        <button
          onClick={onNewAnalysis}
          className={cn(
            'flex w-full items-center gap-2 rounded-lg bg-xade-blue px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-xade-blue-dark',
            isCollapsed && 'justify-center px-2'
          )}
        >
          <Plus className="h-4 w-4 shrink-0" />
          {!isCollapsed && <span>New Analysis</span>}
        </button>
      </div>

      {/* Scrollable analysis history */}
      <SidebarContent>
        {groups.length === 0
          ? !isCollapsed && (
              <div className="px-3 py-6 text-center">
                <p className="text-xs text-xade-charcoal/30">No analyses yet</p>
                <p className="mt-1 text-[10px] text-xade-charcoal/20">
                  Upload an image to get started
                </p>
              </div>
            )
          : groups.map((group) => (
              <div key={group.label} className="mb-3">
                {!isCollapsed && (
                  <p className="mb-1 px-2 text-[10px] font-medium uppercase tracking-wider text-xade-charcoal/35">
                    {group.label}
                  </p>
                )}
                <div className="space-y-0.5">
                  {group.items.map((analysis) => (
                    <AnalysisSidebarItem
                      key={analysis.id}
                      analysis={analysis}
                      imageUrl={getAnalysisImageUrl(analysis, imageMap)}
                      isActive={selectedAnalysisId === analysis.id}
                      onClick={() => onSelectAnalysis(analysis)}
                    />
                  ))}
                </div>
              </div>
            ))}
      </SidebarContent>

      {/* Bottom navigation */}
      <div className="border-t border-xade-charcoal/10 px-3 py-2">
        <div className="space-y-0.5">
          <SidebarItem
            icon={<History />}
            isActive={activeView === 'history'}
            onClick={() => onNavigate('history')}
          >
            History
          </SidebarItem>
          <SidebarItem
            icon={<BarChart3 />}
            isActive={activeView === 'statistics'}
            onClick={() => onNavigate('statistics')}
          >
            Statistics
          </SidebarItem>
        </div>
      </div>

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
  const { user } = useAuth();
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
        data = await analyzeImage(selectedFile, vlmProvider, user?.id);
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
// History View
// ============================================

interface HistoryCardProps {
  analysis: AnalysisResult;
  imageUrl: string | null;
  onClick: () => void;
  onDelete: () => void;
}

function HistoryCard({ analysis, imageUrl, onClick, onDelete }: HistoryCardProps) {
  const isFake = analysis.classification === 'fake';
  const fakeScore = analysis.deepfake_score ?? 0;
  const confidence =
    analysis.status === 'completed' ? Math.round((isFake ? fakeScore : 1 - fakeScore) * 100) : null;
  const [deleting, setDeleting] = useState(false);

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    setDeleting(true);
    try {
      await deleteAnalysis(analysis.id);
      onDelete();
    } catch {
      setDeleting(false);
    }
  }

  return (
    <div
      onClick={onClick}
      className="group flex w-full cursor-pointer items-center gap-4 rounded-xl bg-white px-4 py-3 shadow-sm transition-shadow hover:shadow-md"
    >
      {/* Thumbnail */}
      <div className="h-12 w-12 shrink-0 overflow-hidden rounded-lg bg-xade-charcoal/5">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt="Analysis"
            className="h-full w-full object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <Upload className="h-4 w-4 text-xade-charcoal/20" strokeWidth={1.5} />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          {analysis.status === 'completed' ? (
            <>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                  isFake ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                }`}
              >
                {isFake ? 'Deepfake' : 'Authentic'}
              </span>
              {confidence !== null && (
                <span className="text-xs font-semibold text-xade-charcoal">
                  {confidence}% confidence
                </span>
              )}
            </>
          ) : (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-600 capitalize">
              {analysis.status}
            </span>
          )}
        </div>
        <p className="mt-0.5 truncate text-xs text-xade-charcoal/40">
          {formatRelativeDate(analysis.created_at)}
          {analysis.model_used && ` · ${analysis.model_used}`}
        </p>
      </div>

      {/* Delete button */}
      <button
        onClick={handleDelete}
        disabled={deleting}
        title="Delete analysis"
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xade-charcoal/30 opacity-0 transition-opacity hover:bg-red-50 hover:text-red-500 group-hover:opacity-100 disabled:opacity-50"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

interface HistoryViewProps {
  analyses: AnalysisResult[];
  imageMap: Record<string, ImageRecord>;
  loading: boolean;
  onNewAnalysis: () => void;
  onSelectAnalysis: (analysis: AnalysisResult) => void;
  onRefresh: () => void;
}

function HistoryView({
  analyses,
  imageMap,
  loading,
  onNewAnalysis,
  onSelectAnalysis,
  onRefresh,
}: HistoryViewProps) {
  return (
    <div className="min-h-screen px-10 py-10">
      <div className="mb-8 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-xade-blue">Analysis History</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onRefresh} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </Button>
          <Button variant="outline" onClick={onNewAnalysis}>
            New Analysis
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-24">
          <p className="text-sm text-xade-charcoal/40">Loading history…</p>
        </div>
      ) : analyses.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <History className="mb-4 h-12 w-12 text-xade-charcoal/15" strokeWidth={1} />
          <p className="text-base font-medium text-xade-charcoal/50">No analyses yet</p>
          <p className="mt-1 text-sm text-xade-charcoal/30">Upload an image to get started</p>
          <Button variant="outline" className="mt-6" onClick={onNewAnalysis}>
            Upload Image
          </Button>
        </div>
      ) : (
        <div className="mx-auto max-w-2xl space-y-6">
          {groupAnalysesByDate(analyses).map(({ label, items }) => (
            <div key={label}>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-xade-charcoal/35">
                {label}
              </p>
              <div className="space-y-1.5">
                {items.map((analysis) => (
                  <HistoryCard
                    key={analysis.id}
                    analysis={analysis}
                    imageUrl={getAnalysisImageUrl(analysis, imageMap)}
                    onClick={() => onSelectAnalysis(analysis)}
                    onDelete={onRefresh}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// Statistics View
// ============================================

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: 'red' | 'green' | 'blue';
}) {
  const valueColor =
    color === 'red' ? 'text-red-500' : color === 'green' ? 'text-green-500' : 'text-xade-blue';

  return (
    <div className="rounded-xl bg-white p-6 shadow-md">
      <p className="text-xs uppercase tracking-wide text-xade-charcoal/40">{label}</p>
      <p className={`mt-2 text-4xl font-bold ${valueColor}`}>{value}</p>
      {sub && <p className="mt-1 text-xs text-xade-charcoal/40">{sub}</p>}
    </div>
  );
}

function StatisticsView({ analyses, loading }: { analyses: AnalysisResult[]; loading: boolean }) {
  const completed = analyses.filter((a) => a.status === 'completed');
  const fakes = completed.filter((a) => a.classification === 'fake');
  const reals = completed.filter((a) => a.classification === 'real');
  const avgConfidence =
    completed.length > 0
      ? completed.reduce((sum, a) => {
          const score = a.deepfake_score ?? 0;
          const isFake = a.classification === 'fake';
          return sum + (isFake ? score : 1 - score);
        }, 0) / completed.length
      : 0;

  return (
    <div className="min-h-screen px-10 py-10">
      <h1 className="mb-8 text-2xl font-bold text-xade-blue">Statistics</h1>

      {loading ? (
        <div className="flex items-center justify-center py-24">
          <p className="text-sm text-xade-charcoal/40">Loading…</p>
        </div>
      ) : completed.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <BarChart3 className="mb-4 h-12 w-12 text-xade-charcoal/15" strokeWidth={1} />
          <p className="text-base font-medium text-xade-charcoal/50">No data yet</p>
          <p className="mt-1 text-sm text-xade-charcoal/30">
            Complete some analyses to see your statistics
          </p>
        </div>
      ) : (
        <div className="grid max-w-2xl grid-cols-2 gap-4">
          <StatCard label="Total Analyses" value={completed.length.toString()} color="blue" />
          <StatCard
            label="Avg. Confidence"
            value={`${Math.round(avgConfidence * 100)}%`}
            color="blue"
          />
          <StatCard
            label="Deepfakes Detected"
            value={fakes.length.toString()}
            sub={
              completed.length > 0
                ? `${Math.round((fakes.length / completed.length) * 100)}% of total`
                : undefined
            }
            color="red"
          />
          <StatCard
            label="Authentic Images"
            value={reals.length.toString()}
            sub={
              completed.length > 0
                ? `${Math.round((reals.length / completed.length) * 100)}% of total`
                : undefined
            }
            color="green"
          />
        </div>
      )}
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

interface ResultViewProps {
  result: DetectionResult;
  previewUrl: string | null;
  onBack: () => void;
}

function ResultView({ result, previewUrl, onBack }: ResultViewProps) {
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
    <div className="mx-auto max-w-4xl px-16 py-10">
      {lightboxSrc && <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}

      <button
        onClick={onBack}
        className="mb-8 flex items-center gap-1 text-sm text-xade-charcoal/40 transition-colors hover:text-xade-charcoal"
      >
        <ChevronLeft className="h-4 w-4" />
        Back
      </button>

      {/* Row 1: Verdict */}
      <div className="mb-4 rounded-2xl border border-black/[0.06] bg-white p-6">
        <div className="flex items-center gap-8">
          <div className="shrink-0">
            <p
              className={`text-[11px] font-semibold uppercase tracking-widest ${isFake ? 'text-red-400' : 'text-emerald-500'}`}
            >
              {isFake ? 'Deepfake detected' : 'Authentic'}
            </p>
            <p
              className={`mt-0.5 text-6xl font-bold tabular-nums leading-none ${isFake ? 'text-red-500' : 'text-emerald-500'}`}
            >
              {confidencePct}%
            </p>
            <p className="mt-1 text-sm text-xade-charcoal/40">confidence</p>
          </div>
          <div className="flex-1">
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
        <div className="grid grid-cols-3 gap-3">
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

      {/* Row 3: Facial Regions — stacked list */}
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
                <div key={idx} className="flex items-start gap-4 py-4 first:pt-0 last:pb-0">
                  {/* Zoomed crop */}
                  <div
                    className="h-24 w-24 shrink-0 cursor-zoom-in overflow-hidden rounded-xl border border-black/[0.06]"
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
    </div>
  );
}

// ============================================
// Authenticated App
// ============================================

function AuthenticatedApp() {
  const { user } = useAuth();
  const [view, setView] = useState<AppView>('upload');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [backToView, setBackToView] = useState<AppView>('upload');
  const [apiMode, setApiMode] = useState<ApiMode>('detect');
  const [vlmProvider, setVlmProvider] = useState<string>('openai');
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [imageMap, setImageMap] = useState<Record<string, ImageRecord>>({});
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);

  function loadHistory(userId: string, showLoading = false) {
    if (showLoading) setHistoryLoading(true);
    Promise.all([
      fetchUserAnalyses(userId).then(setAnalyses),
      fetchUserImages(userId).then((images) => {
        const map: Record<string, ImageRecord> = {};
        images.forEach((img) => {
          map[img.id] = img;
        });
        setImageMap(map);
      }),
    ])
      .catch(() => {})
      .finally(() => setHistoryLoading(false));
  }

  useEffect(() => {
    if (user?.id) loadHistory(user.id, true);
    else setHistoryLoading(false);
  }, [user?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleResult(data: DetectionResult, url: string) {
    setResult(data);
    setPreviewUrl(url);
    setSelectedAnalysisId(null);
    setBackToView('upload');
    setView('result');
    // Refresh history in background without clearing the existing list
    if (user?.id) loadHistory(user.id);
  }

  function handleBack() {
    setResult(null);
    setPreviewUrl(null);
    setSelectedAnalysisId(null);
    setView(backToView);
  }

  function handleSelectAnalysis(analysis: AnalysisResult) {
    const imgUrl = getAnalysisImageUrl(analysis, imageMap);
    setResult(analysisToDetectionResult(analysis));
    setPreviewUrl(imgUrl);
    setSelectedAnalysisId(analysis.id);
    setBackToView(view === 'history' ? 'history' : 'upload');
    setView('result');
  }

  function handleNewAnalysis() {
    setResult(null);
    setPreviewUrl(null);
    setSelectedAnalysisId(null);
    setView('upload');
  }

  return (
    <SidebarProvider>
      <AppSidebar
        activeView={view}
        analyses={analyses}
        imageMap={imageMap}
        selectedAnalysisId={selectedAnalysisId}
        onNavigate={(v) => {
          setSelectedAnalysisId(null);
          setView(v);
        }}
        onSelectAnalysis={handleSelectAnalysis}
        onNewAnalysis={handleNewAnalysis}
      />
      <SidebarInset>
        {view === 'result' && result ? (
          <ResultView result={result} previewUrl={previewUrl} onBack={handleBack} />
        ) : view === 'history' ? (
          <HistoryView
            analyses={analyses}
            imageMap={imageMap}
            loading={historyLoading}
            onNewAnalysis={handleNewAnalysis}
            onSelectAnalysis={handleSelectAnalysis}
            onRefresh={() => user?.id && loadHistory(user.id, true)}
          />
        ) : view === 'statistics' ? (
          <StatisticsView analyses={analyses} loading={historyLoading} />
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
    </SidebarProvider>
  );
}

function AppRouter() {
  const { user, loading } = useAuth();
  const [testCompleted, setTestCompleted] = useState(
    () => localStorage.getItem('xade-test-completed') !== null
  );

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

  if (!testCompleted) {
    return (
      <>
        <DeepfakeTest onComplete={() => setTestCompleted(true)} />
        <DevToolbar
          apiMode="detect"
          onApiModeChange={() => {}}
          vlmProvider="none"
          onVlmProviderChange={() => {}}
        />
      </>
    );
  }

  if (!user) {
    return (
      <>
        <AuthPage />
        <DevToolbar
          apiMode="detect"
          onApiModeChange={() => {}}
          vlmProvider="none"
          onVlmProviderChange={() => {}}
        />
      </>
    );
  }

  return <AuthenticatedApp />;
}

function App() {
  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  );
}

export default App;
