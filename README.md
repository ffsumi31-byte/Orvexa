# Verdant 3D — ready-to-deploy v23

This package keeps the existing multi-provider API, classic Roblox-brick builder, background build jobs, revision endpoint, and shared library while improving the Depth 3D quality rules and preview framing.

## Local setup

1. Install Python 3.11+ (3.12 is also fine).
2. Create and activate a virtual environment.
3. Install requirements: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add at least one provider key in your shell or host environment. The app reads environment variables; it does not automatically load `.env`, so use `export GEMINI_API_KEY=...` or a dotenv runner if needed.
5. Start: `uvicorn server:app --host 0.0.0.0 --port 8000`
6. Open `http://127.0.0.1:8000`. API status is at `/health`; standalone low-poly editor is at `/workspace`.

## Render deployment

Push the contents of this folder to a Git repository and create a Render Blueprint from `render.yaml`, or create a Python web service manually with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Health-check path: `/health`

Set `GEMINI_API_KEY` (and optionally `SERPER_API_KEY`) in the host dashboard. You can also configure the OpenAI, Anthropic, and Groq providers. A persistent disk is needed if the shared SQLite library must survive host restarts; set `MEMORY_DB` to its mount path. Without a persistent disk the service still runs, but the host may reset the saved library.

## What changed

- Reworked the Depth 3D system prompt to demand recognizable silhouettes, full anatomy, named accessories, attached parts and purposeful colors.
- Raised Depth 3D part/point budgets, especially at Mid and High effort.
- Added subject checks for humanoid anatomy and explicitly requested swords/shields, which feed the existing auto-revision loop.
- Corrects very thin humanoids with a depth-aware normalization pass while preserving intentionally thin sword, shield, cape, wing and panel geometry.
- Versioned Depth 3D memory cache keys: old v22 builds remain visible in the library but are not automatically returned as exact matches, so a new request can use the new sculpting rules.
- Larger, wider responsive UI and a closer auto-framing camera in the 3D workspace.
- Added a standalone `workspace.html`, deployment blueprint, requirements and environment template.

## Notes

- API provider model IDs and availability depend on the key/provider you configure. Check the chosen provider's current model list if a model ID fails.
- Pricing cards in the front-end are sample UI labels and do not implement payment processing.
