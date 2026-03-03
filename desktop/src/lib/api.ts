const API_BASE_URL = 'http://localhost:8000';

export interface DetectionResult {
  prediction: 'fake' | 'real';
  confidence: number;
  probabilities: {
    fake: number;
    real: number;
  };
  model: string;
  accuracy: string;
}

export type ApiError =
  | { type: 'network'; message: string }
  | { type: 'invalid_file'; message: string }
  | { type: 'model_unavailable'; message: string }
  | { type: 'unknown'; message: string };

export async function detectDeepfake(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('file', file);

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/detect`, {
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
