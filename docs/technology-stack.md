# XADE Technology Stack

**Version:** 0.1.0
Last Updated: February 5, 2026

---

## Overview

XADE uses a modern, cross-platform technology stack optimized for machine learning interference, real-time explanations, and seamless user experience across desktop and mobile devices.

---

### Frontend Technologies

## Desktop Application

Technology          Version         Purpose

Electron            40.1.0          Cross-platform  desktop framework
React               19.x            UI library
TypeScript          5.4.0+          Type-safe Javascript
Vite                7.x             Build tool and dev server
Tailwind CSS        4.x             Utility-first CSS framework
shadcn/ui           latest          Component Library (Radix UI + Tailwind)
React Router        6.x             Client-side routing

**Desktop-Specific Libraries:**
- electron-store
- electron-updater
-@electron/remote

## Mobile Application

Technology          Version         Purpose

Expo                SDK 52          Rect Native development platform
React Native        0.76.x          Cross-platform mobile framework
TypeScript          5.4.0+          Type-safe Javascript
Expo Router         4.x             File-based navigation
NativeWind          4.x             Tailwind CSS for React Native

**Mobile-Specific Libraries:**
- expo-camera
- expo-image-picker
- expo-file-system
- expo-image-manipulator
- react-native-svg
- @react-native-async-storage/async-storage

---

### Backend Technologies

## Core Framework

Technology          Version         Purpose

Python              3.11+           Backend programming language
FastAPI             0.109.0+        Web framework for APIs
Uvicorn             0.27.0+         ASGI Server
Pydantic            2.5.0+          Data validation and settings
pyhton-multipart    0.0.6+          File upload handling

## Database & Storage

Technology          Version         Purpose

Supabase            latest          PostgreSQL hosting + Auth + Storage
PostgreSQL          15+             Relational database
SQLAlchemy          2.0+            ORM (Object-Relation Mapping)

**Supabase Services Used:**
- Database: Managed PostgreSQL with row-level security
- Storage: Image uploads and heatmap storage
- Auth: (Future) User authentication and authorization
- Realtime: (Future) Live analysis updates

---

### Machine Learning & AI

## Detection Models

Technology          Version         Purpose

PyTorch             2.1+            Deep learning framework
torchvision         0.16+           Computer vision utilities
transformers        4.36+           CLIP and other transformers
opencv-python       4.9+            Image preprocessing
Pillow              10.2.0+         Image handling

**Detection Models:**
- CLIP (openai/clip-vit-large-patch14) - Primary detection
- Efficientnet-B4 - Alternative detection model
- GradCAM - Heatmap generation
//TBC


## Vision-Language Models (VLMs)

Provider        Version                 SDK/API                 Purpose

OpenAI          GPT-4V                  openai (1.10+)          Premium explanations
Google          Gemini 1.5 Pro Vision   google-generativeai     Fast Explanations
Anthropic       Claude 3.5              anthropic (0.12+)       Detailed explanations
Open source     LLaVA 1.6 (34B)         transformers            Local interference
//TBC

**VLM Integration:**
- httpx (0.26x) - Async HTTP client for API calls


---

### Development Tools

## Code Quality

Tool            Purpose                         Config File

ESLint          JavaScript/Typescript linting   .eslintrc.cjs
Prettier        Code formatting                 .prettierrc
Ruff            Python linting & formatting     backend/pyproject.toml
TypeScript      Type checking                   tsconfig.json


## Testing

Technology              Scope           Purpose

pytest                  Backend         Unit & integration tests
pytest-asyncio          Backned         Async test support
httpx                   Backend         Test client for FastAPI
Vitest                  Frontend        Unit & component tests
React Testing Library   Frontend        Component testing
Playwright              E2E             End-to-end testing


## CI/CD

Technology              Purpose

GitHub Actions          Continuous integration
Docker                  Containerization (production)
Docker Compose          Local development orchestration

**CI Pipeline:**
- Lint checking (ESLint, Ruff)
- Format checking (Prettier, Ruff)
- Type checking (TypeScript)
- Unit tests (pytest, Vitest)
- Build validation


---

### API Communication

## REST API

- Protocol: HTTPS (production), HTTP (development)
- Format: JSON
- File Uploads: multipart/form-data
- Authentication: (Future) JWT tokens via Supabase Auth


## WebSocket (Future)

- Protocol: WSS
- Use Case: Real-time analysis progress update
- Library: webscokets (Python), native WebSocket API (JS)



