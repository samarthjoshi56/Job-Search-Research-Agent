# Job Search Research Agent

A Python + LangGraph-based research agent that evaluates job descriptions, parses candidate resumes, searches the web for company insights, and stores all research in a persistent SQLite database so interrupted workflows can be resumed without losing work.

> **Current Phase: Phase 3** – Persistent Memory & State Management.

---

## 🏗️ Architecture

```text
                   ┌── Web Search (Tavily API)
                   │
Job Description    ├── Resume Parser (LLM Extractor)
       ↓           │
    Planner ───────┤── Job Matcher (Requirements vs Resume)
       ↓           │
   Tool Agent ─────┤── Research Collector (State Organizer)
       ↓           │
Tool Execution ────┘
       ↓
Response Generator
       ↓
  Final Analysis
       ↓
  SQLite DB ← persists research, analysis & final response at every step
```

### Phase 3 Persistence Flow

1. **Start / Resume** – on `new`, a UUID `job_id` is generated and the job is saved to SQLite.  On `resume`, previously saved research and candidate analysis are loaded back into the agent state, and the LLM is told what was already researched so it avoids duplicate tool calls.
2. **Tool Execution** – after every tool round, `research`, `resume_data`, `job_match`, and `tool_results` are written to the database.
3. **Interrupt** – if the process is killed mid-run (Ctrl+C), the status is set to `interrupted`.  Re-running `python main.py resume <JOB_ID>` picks up where execution stopped.
4. **Completion** – when the final response is generated it is persisted, and the job status becomes `complete`.

---

## 📁 Project Structure

```text
Job-Search-Research-Agent/
│
├── agent/
│   ├── __init__.py       # Package export
│   ├── state.py          # AgentState TypedDict (includes job_id – Phase 3)
│   ├── planner.py        # Planner node (LLM-based extraction)
│   ├── generator.py      # Final response generator node
│   └── graph.py          # LangGraph workflow with SQLite persistence hooks
│
├── tools/
│   ├── __init__.py       # Tools package export
│   ├── web_search.py     # Web search tool (Tavily)
│   ├── resume_parser.py  # Resume parsing tool
│   ├── job_matcher.py    # Job matching tool
│   └── research_store.py # State research collector tool
│
├── storage/
│   ├── __init__.py       # Storage package export
│   └── database.py       # SQLite persistence layer (Phase 3)
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py     # Phase 1 + 2 agent tests (7 tests)
│   ├── test_tools.py     # Tool unit tests (8 tests)
│   └── test_storage.py   # Phase 3 database tests (29 tests)
│
├── data/
│   └── .gitkeep          # Ensures data/ is tracked; agent.db lives here
│
├── .env.example          # Environment variable placeholders
├── .gitignore
├── requirements.txt
├── main.py               # Phase 3 CLI (new / resume / list)
└── README.md
```

---

## 🗄️ Database Schema

All tables use SQLite via Python's built-in `sqlite3` module. No ORM.

### `jobs`
| Column | Type | Description |
|---|---|---|
| `job_id` | TEXT PK | UUID assigned at creation |
| `title` | TEXT | Optional job title |
| `company` | TEXT | Optional company name |
| `job_description` | TEXT | Full job description text |
| `resume_text` | TEXT | Optional candidate resume |
| `status` | TEXT | `new` → `running` → `complete` / `interrupted` / `error` |
| `created_at` | TEXT | ISO-8601 UTC timestamp |
| `updated_at` | TEXT | ISO-8601 UTC timestamp |

### `research`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `job_id` | TEXT FK → jobs | Parent job session |
| `key` | TEXT | Research category (e.g. `company_culture`) |
| `value` | TEXT | JSON-encoded research data |
| `source` | TEXT | Optional source attribution |
| `created_at` | TEXT | ISO-8601 UTC |
| UNIQUE | `(job_id, key)` | Upsert semantics |

### `candidate_analysis`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `job_id` | TEXT FK → jobs | Parent job session |
| `resume_data` | TEXT | JSON-encoded parsed resume |
| `job_match` | TEXT | JSON-encoded matching result |
| `tool_results` | TEXT | JSON-encoded web search results |
| `created_at / updated_at` | TEXT | ISO-8601 UTC |

### `final_responses`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `job_id` | TEXT FK → jobs | Parent job session |
| `response_json` | TEXT | Full JSON of `FinalResponse` |
| `created_at / updated_at` | TEXT | ISO-8601 UTC |

---

## 🧰 Available Tools

| Tool | File | Description |
|---|---|---|
| `search_web` | `tools/web_search.py` | Searches via Tavily API; returns `title`, `url`, `snippet`. |
| `parse_resume` | `tools/resume_parser.py` | Extracts `skills`, `frameworks`, `experience`, `education`, `projects`. |
| `match_job` | `tools/job_matcher.py` | Identifies matching skills, missing skills, and potential gaps. |
| `store_research` | `tools/research_store.py` | Saves key research findings to agent state (and DB in Phase 3). |

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **LangGraph** – Workflow & tool-calling orchestration
- **LangChain / LangChain OpenAI** – LLM integration and structured outputs
- **Tavily Python** – Web search API
- **SQLite (built-in)** – Phase 3 persistent memory (no ORM)
- **Pydantic** – Data schema validation
- **python-dotenv** – Environment configuration
- **pytest** – Test suite (44 tests, 0 failures)

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/samarthjoshi56/Job-Search-Research-Agent.git
cd Job-Search-Research-Agent
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here      # optional
JOB_AGENT_DB_PATH=data/agent.db              # optional, this is the default
```

---

## 💻 CLI Usage (Phase 3)

### Start a new job research session
```bash
python main.py new
# or simply:
python main.py
```
Paste the job description when prompted, optionally add a resume, then press `Ctrl+D`.  
A `job_id` UUID is printed and saved to the database.

### Resume an interrupted session
```bash
python main.py resume <JOB_ID>
```
Loads previously saved research from SQLite and resumes the workflow, skipping already-completed tool calls.

### List all job sessions
```bash
python main.py list
```
Displays all sessions with their ID, status, and creation time.

---

## 🧪 Running Tests

```bash
pytest -v
```

**44 tests across 3 modules:**
- `test_agent.py` – 7 Phase 1/2 agent tests (mocked LLM)
- `test_tools.py` – 8 tool unit tests (mocked Tavily + LLM)
- `test_storage.py` – 29 Phase 3 database tests (isolated tmp SQLite)

All database tests use pytest's `tmp_path` fixture – no real database is touched.

---

## 🔮 Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Core LangGraph architecture |
| Phase 2 | ✅ Complete | Tool integration (web search, resume parsing, job matching) |
| Phase 3 | ✅ Complete | SQLite persistence, resume interrupted workflows, CLI |
| Phase 4 | 🔜 Planned | Automated evaluation harness & scoring |
| Phase 5 | 🔜 Planned | FastAPI REST endpoints & Docker containerisation |
