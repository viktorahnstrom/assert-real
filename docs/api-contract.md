# XADE API Contract

**Version:** 0.1.0
**Base URL:** `http://localhost:8000/api/v1``

## Authentication
Currently none (development). Production will require API keys.

---

## Endpoints

### 1. Health Check
**GET** `/health`

**Response 200:**

json:

{
    "status": "healthy",
    "timestamp: "2026-02-05T10:30:00Z"
}

---

### 2. Analyze Image from Deepfake
**POST** `/analyze`

**Requests:**
- Content-Type: `multipart/form-data`
- Body:
    - `image` : File (JPEG/PNG, max 10MB)
    - `explanation_detail`: String (optional) - "simple" | "detailed" | "technical"
    - `vlm_provider`: String (optional) - "gpt4v" | "gemini" | "llava" | "claude"

**Response 200:**

json:

{
    "analysis_id": "uuid-string",
    "timestamp": "2026-02-05T10:30:00Z",
    "detection" : {
        "is_deepfake": true,
        "confidence": 0.87,
        "model_used": "clip-large"
    },
    "explanation:" {
        "summary: "This image shows signs of AI manipulation...",
        "detailed_findings": [
            "Unnatural facial boundaries near the jawline",
            "Inconsistent lightning patterns on the left cheek",
            "Frequency domain anomalies in high-frequency components"
        ],
        "heatmap-url": "/resulst/uuid-string/heatmap.png",
        "vlm_provider": "gpt4v"
    },
    "metadata": {
        "image_dimensions": [1920, 1080],
        "processing_time_ms": 2340
    }
}

**Response 400:** Invalid image format
**Response 413:** File too large
**Response 500:** Processing error

---

### 3. Get Analysis History
**GET** `/history`

**Query Parameters:**
- `limit` Integer (default: 20, max: 100)
- `offset` Integer (default: 0)

**Response 200:**

json:

{
    "total": 45,
    "results: [
        {
            "analysis_id": "uuid-string",
            "timestamp": "2026-02-05T10:30:00Z",
            "thumbnail_url": "/results/uuid-string/thumbnail.jpg",
            "is_deepfake": true,
            "conficence": 0.87
        }
    ]
}

---

### 4. Get Specific Analysis
**GET** `/analysis/{analysis_id}`

**Response 200:** Same structure as POST `analyze` response

**Reponse 404:** Analysis not found

---

### 5. List Available VLM Providers
**GET** `/vlm-provider`

json:

{
    "providers": [
        {
            "id": "gpt4v",
            "name": "GPT-4 Vision",
            "available": true,
            "latency-estimate_ms": 2000
        },
        {
            "id": "gemini",
            "name": "Gemini Vision",
            "available": true,
            "latency-estimate_ms": 1500
        },
        {
            "id": "llava",
            "name": "LLaVA 1.6",
            "available": false,
            "latency-estimate_ms": null
        },
        {
            "id": "claude",
            "name": "Claude Vision",
            "available": true,
            "latency-estimate_ms": 1800
        }
    ]
}

---

## Error Response Format

json:

{
    "error": {
        "code": "INVALID_IMAGE_FORMAT",
        "message: "Only JPEG and PNG images are supported",
        "details": {}
    }
}


---

## Rate Limits
- Development: No limits
- Production: 100 requests/hour per IP (to be implemented)

## WebSocket Support (Future)
**WS** `/ws/analyze`

For real-time progress updates during long-running analyses.