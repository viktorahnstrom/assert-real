const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ============================================
// Types
// ============================================

export interface ExplanationResult {
  summary: string;
  detailed_analysis: string;
  technical_notes: string | null;
  provider: string;
  model: string;
  processing_time_ms: number;
  estimated_cost_usd: number;
}

export interface EvidenceRegion {
  url: string;
  label: string;
  activation_score: number;
  explanation: string | null;
  category_id: string | null;
  category_label: string | null;
  common_artifacts: string[] | null;
  z_scores?: Record<string, number> | null;
  evidence_type?: string | null;
  evidence_ref?: string | null;
  cam_score?: number | null;
  forensic_score?: number | null;
  suspicion_score?: number | null;
  claim_confidence?: number | null;
}

export interface DetectionResult {
  prediction: 'fake' | 'real';
  confidence: number;
  probabilities: {
    fake: number;
    real: number;
  };
  model: string;
  accuracy: string;
  gradcam_heatmap_url: string | null;
  ela_heatmap_url?: string | null;
  explanation: ExplanationResult | null;
  evidence_regions: EvidenceRegion[];
}

export interface AnalysisResult {
  id: string;
  image_id: string;
  user_id: string;
  status: string;
  deepfake_score: number | null;
  classification: string | null;
  model_used: string | null;
  vlm_explanation: string | null;
  vlm_model_used: string | null;
  processing_time_ms: number | null;
  created_at: string;
  completed_at: string | null;
  explanation: ExplanationResult | null;
  gradcam_heatmap_url?: string | null;
  ela_heatmap_url?: string | null;
  evidence_regions?: EvidenceRegion[];
}

export interface ImageRecord {
  id: string;
  filename: string;
  storage_path: string;
  file_size: number;
  mime_type: string;
  uploaded_at: string;
  url?: string; // Signed URL, populated client-side
}

export interface VLMProvider {
  id: string;
  name: string;
  model: string;
  available: boolean;
  latency_estimate_ms: number | null;
  cost_per_1m_input_tokens: number | null;
  cost_per_1m_output_tokens: number | null;
}

export type ApiError =
  | { type: 'network'; message: string }
  | { type: 'invalid_file'; message: string }
  | { type: 'model_unavailable'; message: string }
  | { type: 'unknown'; message: string };

export type ApiMode = 'detect' | 'analyses';

// ============================================
// Direct detect endpoint (development/testing)
// ============================================

import { supabase } from './supabase';

export async function detectDeepfake(file: File, _vlmProvider?: string): Promise<DetectionResult> {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    const error: ApiError = {
      type: 'network',
      message: 'Not authenticated. Please log in.',
    };
    throw error;
  }

  const formData = new FormData();
  formData.append('file', file);

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${session.access_token}`,
      },
      body: formData,
    });
  } catch {
    const error: ApiError = {
      type: 'network',
      message: 'Cannot reach the XADE backend. Make sure it is running on port 8000.',
    };
    throw error;
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Unknown error' }));
    const message = body.detail ?? 'Unknown error';

    if (response.status === 400) throw { type: 'invalid_file', message } as ApiError;
    if (response.status === 401)
      throw { type: 'network', message: 'Session expired. Please log in again.' } as ApiError;
    if (response.status === 503) throw { type: 'model_unavailable', message } as ApiError;

    throw { type: 'unknown', message } as ApiError;
  }

  return response.json() as Promise<DetectionResult>;
}

// ============================================
// Full analyses flow (upload → analyse → save to DB)
// ============================================

async function uploadImage(file: File, userId: string): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE_URL}/api/v1/images/upload?user_id=${encodeURIComponent(userId)}`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw { type: 'unknown', message: body.detail ?? 'Failed to upload image' } as ApiError;
  }

  const data = await response.json();
  return data.id as string;
}

async function createAnalysis(
  imageId: string,
  userId: string,
  vlmProvider: string
): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analyses/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_id: imageId,
      user_id: userId,
      vlm_provider: vlmProvider,
    }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Analysis failed' }));
    throw { type: 'unknown', message: body.detail ?? 'Analysis failed' } as ApiError;
  }

  return response.json() as Promise<AnalysisResult>;
}

export async function analyzeImage(
  file: File,
  vlmProvider: string = 'openai',
  userId?: string
): Promise<DetectionResult> {
  if (!userId) {
    throw { type: 'network', message: 'Not authenticated. Please log in.' } as ApiError;
  }
  let imageId: string;

  try {
    imageId = await uploadImage(file, userId);
  } catch {
    throw {
      type: 'network',
      message: 'Failed to upload image. Make sure backend and Supabase are running.',
    } as ApiError;
  }

  const analysis = await createAnalysis(imageId, userId, vlmProvider);

  const fakeScore = analysis.deepfake_score ?? 0;
  const realScore = 1 - fakeScore;
  const isFake = analysis.classification === 'fake';

  return {
    prediction: (analysis.classification as 'fake' | 'real') ?? 'real',
    confidence: isFake ? fakeScore : realScore,
    probabilities: {
      fake: fakeScore,
      real: realScore,
    },
    model: analysis.model_used ?? 'EfficientNet-B4',
    accuracy: '98.48%',
    gradcam_heatmap_url: analysis.gradcam_heatmap_url ?? null,
    ela_heatmap_url: analysis.ela_heatmap_url ?? null,
    explanation: analysis.explanation ?? null,
    evidence_regions: analysis.evidence_regions ?? [],
  };
}

// ============================================
// User history
// ============================================

export async function fetchUserAnalyses(userId: string): Promise<AnalysisResult[]> {
  try {
    const url = `${API_BASE_URL}/api/v1/analyses/?user_id=${encodeURIComponent(userId)}`;
    console.log('[XADE] fetchUserAnalyses →', url);
    const response = await fetch(url);
    if (!response.ok) {
      console.warn('[XADE] fetchUserAnalyses failed:', response.status, response.statusText);
      return [];
    }
    const data = await response.json();
    console.log('[XADE] fetchUserAnalyses raw response:', data);
    // Backend returns { analyses: [...], count: N }
    const list = (
      Array.isArray(data) ? data : (data.analyses ?? data.items ?? [])
    ) as AnalysisResult[];
    console.log('[XADE] fetchUserAnalyses parsed:', list.length, 'items');
    return list;
  } catch (err) {
    console.error('[XADE] fetchUserAnalyses error:', err);
    return [];
  }
}

export async function fetchUserImages(userId: string): Promise<ImageRecord[]> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/images/?user_id=${encodeURIComponent(userId)}`
    );
    if (!response.ok) return [];
    const data = await response.json();
    // Backend returns { images: [...], count: N }
    const images = (
      Array.isArray(data) ? data : (data.images ?? data.items ?? [])
    ) as ImageRecord[];

    if (images.length === 0) return images;

    // The Supabase images bucket is private — generate signed URLs (1h expiry)
    const storagePaths = images.map((img) => img.storage_path);
    const { data: signedUrls, error } = await supabase.storage
      .from('images')
      .createSignedUrls(storagePaths, 3600);

    if (!error && signedUrls) {
      return images.map((img, i) => ({
        ...img,
        url: signedUrls[i]?.signedUrl ?? undefined,
      }));
    }

    console.warn('[XADE] fetchUserImages: could not generate signed URLs', error);
    return images;
  } catch (err) {
    console.error('[XADE] fetchUserImages error:', err);
    return [];
  }
}

export async function deleteAnalysis(analysisId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analyses/${analysisId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Delete failed' }));
    throw { type: 'unknown', message: body.detail ?? 'Failed to delete analysis' } as ApiError;
  }
}

// ============================================
// User study
// ============================================

export interface StudyExplanation {
  provider: string;
  model: string;
  summary: string;
  detailed_analysis: string;
  technical_notes: string | null;
  processing_time_ms: number;
  error: string | null;
}

export interface StudyAnalysisResult {
  deepfake_score: number;
  classification: string;
  confidence: number;
  gradcam_url: string | null;
  ela_heatmap_url?: string | null;
  evidence_regions?: EvidenceRegion[];
  explanations: Record<string, StudyExplanation>;
}

export interface StudyResultsPayload {
  participant_id: string;
  self_confidence_rating: number;
  baseline_accuracy: number;
  total_images: number;
  correct_count: number;
  incorrect_count: number;
  explanation_answers: object[];
  // Phase 3 — empty array when the participant got 100% on Phase 1 and
  // the retest was skipped. Each entry carries the per-image answer; the
  // #118 timer work will additionally include time_ms and idle_discarded.
  retest_answers: object[];
  trust_rating: number;
  willingness_to_use: string;
  comments: string;
  completed_at: string;
}

export async function studyAnalyzeImage(imageUrl: string): Promise<StudyAnalysisResult> {
  const imgResponse = await fetch(imageUrl);
  if (!imgResponse.ok) throw new Error(`Failed to fetch quiz image: ${imageUrl}`);
  const blob = await imgResponse.blob();
  const filename = imageUrl.split('/').pop() ?? 'image.jpg';
  const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/study/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Study analysis failed' }));
    throw new Error(body.detail ?? 'Study analysis failed');
  }

  return response.json() as Promise<StudyAnalysisResult>;
}

export async function saveStudyResults(payload: StudyResultsPayload): Promise<void> {
  // Store directly in Supabase using the anon key (no login required).
  // The study_results table must have RLS allowing anon inserts — see README.
  try {
    const { error } = await supabase.from('study_results').insert({
      participant_id: payload.participant_id,
      self_confidence_rating: payload.self_confidence_rating,
      baseline_accuracy: payload.baseline_accuracy,
      total_images: payload.total_images,
      correct_count: payload.correct_count,
      incorrect_count: payload.incorrect_count,
      explanation_answers: payload.explanation_answers,
      retest_answers: payload.retest_answers,
      trust_rating: payload.trust_rating,
      willingness_to_use: payload.willingness_to_use,
      comments: payload.comments,
      completed_at: payload.completed_at,
    });
    if (error) console.warn('[XADE study] Supabase insert failed:', error.message);
  } catch (err) {
    console.warn('[XADE study] saveStudyResults failed:', err);
  }
}

// ============================================
// VLM providers
// ============================================

export async function fetchVLMProviders(): Promise<VLMProvider[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/vlm-providers`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.providers as VLMProvider[];
  } catch {
    return [];
  }
}
