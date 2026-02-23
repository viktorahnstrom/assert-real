const API_BASE_URL = 'http://localhost:8000';

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

export async function detectDeepfake(
  file: File,
  vlmProvider: string = 'openai',
): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({
    vlm_provider: vlmProvider,
    explain: 'true',
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

export async function fetchVLMProviders(): Promise<VLMProvider[]> {
  const response = await fetch(`${API_BASE_URL}/api/vlm-providers`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.providers as VLMProvider[];
}