# Belief Extraction API

A FastAPI service that analyses StoryBot conversation transcripts to surface the **evolving beliefs** a user holds about themselves and different topics, powered by **Gemma 4 (`gemma4:e4b`) via Ollama**.

The API is designed to serve three downstream teams:

| Team | Use of output |
|---|---|
| **Conversation Scoring** | `conversation_value_score` (0–10) |
| **StoryBot** | `dominant_theme` + `suggested_follow_up` |
| **Content Recommendation** | `content_tags` |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Any recent 3.x for local dev |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager |
| [Ollama](https://ollama.com/download) | LLM runtime (run on host machine) |
| Docker Desktop (optional) | Required only for Docker deployment |

---

## Quick Start — Local Development

### 1. Install Ollama & pull the model

```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull gemma4:e4b
```

### 2. Install Python dependencies with uv

```bash
# From the repo root
pip install uv   # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install runtime deps
uv pip install -r build/app/requirements.txt
```

### 3. Start Ollama (if not already running)

```bash
ollama serve   # runs on http://localhost:11434 by default
```

### 4. Run the API

> If you have Anaconda installed, always use the venv uvicorn (not the system one) to avoid NumPy version conflicts.

```powershell
# Option A — activate the venv first, then run normally
..\.venv\Scripts\activate          # from inside build/
uvicorn app.main:app --reload --port 8000

# Option B — run directly without activating
# (from the repo root)
.venv\Scripts\uvicorn app.main:app --app-dir build --reload --port 8000

# Option B — or from the build/ directory
..\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

The API will be available at:
- **Interactive docs (Swagger UI)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## Endpoints

### `POST /api/v1/analyze`

Accepts a StoryBot conversation and returns extracted beliefs with downstream scores.

**Request body** — matches the `conversations.json` format:

```json
{
  "messages_list": [
    {
      "ref_conversation_id": 98696,
      "ref_user_id": 782,
      "transaction_datetime_utc": "2023-10-01T10:15:00Z",
      "screen_name": "ChattyPenguin",
      "message": "I'd like to read the daily stories! They used to lift my spirits."
    }
  ],
  "ref_conversation_id": 98696,
  "ref_user_id": 782
}
```

**Query parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `debug` | bool | `false` | Include raw Gemma output in response |

**Example response**:

```json
{
  "ref_conversation_id": 98696,
  "ref_user_id": 782,
  "analyzed_at": "2024-06-01T12:00:00",
  "message_count": 20,
  "user_message_count": 10,
  "beliefs": [
    {
      "topic": "autonomy",
      "belief": "The user strongly values remaining independent in their home.",
      "sentiment": "positive",
      "confidence": 0.92,
      "evidence": "Yes, please! I want to stay here for as long as I can."
    }
  ],
  "downstream_scores": {
    "conversation_value_score": 8.1,
    "dominant_theme": "autonomy",
    "suggested_follow_up": "How are you feeling about your safety at home this week?",
    "content_tags": ["aging-in-place", "family-support", "independence", "self-care"]
  },
  "raw_llm_output": null
}
```

### `GET /health`

Returns API version and liveness status.

---

## Testing the endpoint with curl

Using the first conversation from the sample data:

```bash
# Extract the first conversation and POST it
python -c "
import json, sys
data = json.load(open('data/conversations.json'))
print(json.dumps(data[0]))
" | curl -s -X POST http://localhost:8000/api/v1/analyze \
    -H 'Content-Type: application/json' \
    -d @- | python -m json.tool
```

Or with **httpx** (Python):

```python
import httpx, json

payload = json.load(open("data/conversations.json"))[0]
resp = httpx.post("http://localhost:8000/api/v1/analyze", json=payload)
print(resp.json())
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:e4b` | Ollama model tag |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `DEBUG_RESPONSES` | `false` | Always include raw LLM output |

---

## Docker Deployment

> Requires Docker Desktop (or Docker Engine) + Ollama running on the **host machine**.

```bash
# 1. Pull the model on the host
ollama pull gemma4:e4b

# 2. Copy and (optionally) edit the env file
cp build/deployment/.env.example build/deployment/.env

# 3. Build and start the container
cd build/deployment
docker compose up --build

# API is now available at http://localhost:8000
```

**Linux users**: edit `.env` and set `OLLAMA_BASE_URL=http://172.17.0.1:11434` (default Docker bridge gateway) instead of `host.docker.internal`.

---

## Running Tests

```bash
# 1. Create an isolated virtual environment (only needed once)
uv venv .venv --seed

# 2. Install all deps into the venv
uv pip install --python .venv/Scripts/python -r build/app/requirements.txt -r tests/requirements-test.txt

# 3. Run the test suite (no live Ollama required)
.venv/Scripts/pytest -v
```

Tests are organised as follows:

| File | Coverage |
|---|---|
| `tests/test_schemas.py` | Pydantic model validation (no network) |
| `tests/test_extractor.py` | Feature extraction with Ollama mocked |
| `tests/test_api.py` | Full endpoint integration with mocked services |

No live Ollama instance is needed to run the test suite — all external calls are mocked via `pytest-httpx`.

---

## Project Structure

```
muh-loret/
├── build/
│   ├── Dockerfile                  # Multi-stage uv build
│   ├── app/
│   │   ├── main.py                 # FastAPI entry-point
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── requirements.txt        # Runtime dependencies
│   │   ├── routers/
│   │   │   └── analyze.py          # POST /api/v1/analyze
│   │   ├── services/
│   │   │   ├── extractor.py        # Gemma 4 belief extraction
│   │   │   └── ml_models.py        # TF-IDF scoring & tag enrichment
│   │   └── utils/
│   │       └── __init__.py         # Ollama helpers, JSON parsing
│   └── deployment/
│       ├── docker-compose.yml      # Docker Compose (single service)
│       └── .env.example            # Environment variable template
├── data/
│   ├── conversations.json          # Sample StoryBot conversation data
│   └── README.md                   # Dataset documentation
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── requirements-test.txt       # Test-only deps
│   ├── test_schemas.py
│   ├── test_extractor.py
│   └── test_api.py
└── README.md                       # ← you are here
```

---

## How it Works

1. **Conversation ingestion** — The API validates incoming JSON against the `conversations.json` schema.
2. **Belief extraction** — User-only messages are assembled into a transcript and sent to `gemma4:e4b` via Ollama with a structured JSON prompt asking for beliefs, sentiment, and confidence.
3. **ML post-processing** — Extracted beliefs are clustered using TF-IDF cosine similarity (scikit-learn) and scored across five heuristic factors (topic coverage, LLM confidence, sentiment richness, conversation length, personal disclosure).
4. **Structured output** — Results are returned with per-belief entries and three downstream score fields ready for consumption.

## Follow up work

1. **Give options for CPU only** - The ollama instance and model chosen for this exercise may be intensive for CPU only workstations. Needs scoping to lighterweight models matching performance.
2. **Refine ML to illustrate belief and opinion growth** - The current ML pipeline is basic and could be improved to better illustrate belief and opinion growth over time and between conversations.
3. **Enrich conversation data more** - The current dataset is small and may not be representative of all possible conversations. Needs to be expanded to include more data.
4. **Add more API functions** - The current API is limited and could be expanded to include more functions and endpoints. For example, a function to get the most recent conversation for a user, or a function to get the most recent conversation for a user.
5. **Add k8s deployment and CI/CD** - The current deployment is local only and could be improved to include k8s deployment and CI/CD. 
6. **Add iterative liveness checks** - The current liveness check is basic and could be improved to include more information about the API's health. For example, a function to get the most recent conversation for a user, or a function to get the most recent conversation for a user.
7. **Enhance scoring models** - Current tests yielded questionable analysis and the scoring models need to be improved to better reflect the conversation data. Currently the model is not deterministic and yields different results for the same input.
8. **Add frontend UI** - Enhance the user experience for interacting with the API. This would help with testing and demonstrating the API, but also scope for other features like conversation flow and showcasing multiple conversations from the same user. 