const API_BASE_URL = 'http://localhost:8000';

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
  explanation: ExplanationResult | null;
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

export async function detectDeepfake(
  file: File,
  vlmProvider: string = 'openai',
): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({
    vlm_provider: vlmProvider,
    explain: 'true',
    include_gradcam: 'true',
  });

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/detect?${params}`, {
      method: 'POST',
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

    if (response.status === 400) {
      const error: ApiError = { type: 'invalid_file', message };
      throw error;
    }
    if (response.status === 503) {
      const error: ApiError = { type: 'model_unavailable', message };
      throw error;
    }

    const error: ApiError = { type: 'unknown', message };
    throw error;
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
    },
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
  vlmProvider: string,
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
  userId: string = '00000000-0000-0000-0000-000000000000',
): Promise<DetectionResult> {
  let imageId: string;

  try {
    imageId = await uploadImage(file, userId);
  } catch {
    const error: ApiError = {
      type: 'network',
      message: 'Failed to upload image. Make sure backend and Supabase are running.',
    };
    throw error;
  }

  try {
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
      gradcam_heatmap_url: null,
      explanation: analysis.explanation ?? null,
    };
  } catch (err) {
    throw err as ApiError;
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