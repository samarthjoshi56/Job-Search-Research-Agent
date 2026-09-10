# Job Search Research Agent

A Python + LangGraph-based research agent designed to help users evaluate, analyze, and process job descriptions and candidate resumes efficiently.

> **Note:** This repository currently represents **Phase 2** of the project, focusing on tool integration, structured candidate matching, web research, and source grounding. Persistent database storage (SQLite), FastAPI endpoints, Docker support, and automated evaluation harnesses will be introduced in Phase 3+.

---

## 🏗️ Phase 2 Architecture

In Phase 2, the agent is upgraded with a dynamic tool-calling architecture. The LLM determines which tools are necessary based on the job description and candidate resume input:

```text
               ┌── Web Search (Tavily API)
               │
Job Description│── Resume Parser (LLM Extractor)
       ↓       │
    Planner ───┤── Job Matcher (Requirements vs Resume)
       ↓       │
  Tool Agent ──┤── Research Collector (State Organizer)
       ↓
Response Generator
       ↓
  Final Analysis
```

### Graph Execution Workflow
1. **Planner Node (`agent/planner.py`)**: Analyzes the raw job description and extracts structured evaluation criteria.
2. **Tool Agent Node (`agent/graph.py`)**: Evaluates state and decides dynamically which tools to invoke. Bounded by a maximum iteration limit (`MAX_ITERATIONS = 5`) to prevent infinite tool-calling loops.
3. **Tool Execution Node (`agent/graph.py`)**: Executes requested tool calls and updates `AgentState` (`tool_results`, `resume_data`, `job_match`, `research`).
4. **Response Generator Node (`agent/generator.py`)**: Synthesizes all gathered information, grounding company details in source links and highlighting candidate skill matches/gaps.

---

## 🧰 Available Tools

| Tool | File | Description |
| :--- | :--- | :--- |
| `search_web` | `tools/web_search.py` | Searches company & role information via Tavily API; returns normalized `title`, `url`, and `snippet`. |
| `parse_resume` | `tools/resume_parser.py` | Extracts structured candidate details (`skills`, `frameworks`, `experience`, `education`, `projects`) from raw resume text. |
| `match_job` | `tools/job_matcher.py` | Compares candidate skills and experience against job requirements to identify matching skills, missing skills, and potential gaps. |
| `store_research` | `tools/research_store.py` | Organizes key research findings into the agent state. |

---

## 🛠️ Tech Stack
- **Python**: 3.11+
- **LangGraph**: Workflow & tool-calling orchestration
- **LangChain / LangChain OpenAI**: LLM integration and structured outputs
- **Tavily Python**: Web search API integration
- **Pydantic**: Data schema definition and validation
- **python-dotenv**: Environment configuration
- **pytest**: Test suite framework

---

## 📁 Project Structure

```text
Job-Search-Research-Agent/
│
├── agent/
│   ├── __init__.py       # Package initialization
│   ├── state.py          # Extended AgentState (TypedDict)
│   ├── planner.py        # Planner node
│   ├── generator.py      # Final response generator node
│   └── graph.py          # LangGraph workflow with tool-calling loop
│
├── tools/
│   ├── __init__.py       # Tools package export
│   ├── web_search.py     # Web search tool (Tavily)
│   ├── resume_parser.py  # Resume parsing tool
│   ├── job_matcher.py    # Job matching tool
│   └── research_store.py # State research collector tool
│
├── tests/
│   ├── __init__.py       # Test package initialization
│   ├── test_agent.py     # Agent state, planner, generator, and graph tests
│   └── test_tools.py     # Unit tests for all tools (100% mocked APIs)
│
├── .env.example          # Environment variable placeholders
├── .gitignore             # Git ignore configuration
├── requirements.txt      # Python dependencies
├── main.py               # Updated CLI entry point
└── README.md             # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/samarthjoshi56/Job-Search-Research-Agent.git
cd Job-Search-Research-Agent
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to supply your credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 💻 How to Run

Run the CLI interface:
```bash
python main.py
```

1. **Job Description**: Paste job description text and press `Ctrl+D` (or `Ctrl+Z` on Windows) on a new line.
2. **Resume Text** (Optional): Enter candidate resume text or press Enter to skip.

---

## 🧪 Running Tests

Unit tests mock external LLM and Tavily API calls to ensure zero network dependencies during testing:

```bash
pytest
```

---

## 🔮 Roadmap / Future Phases
- **Phase 3**: Vector DB (FAISS/Chromadb) & Persistent Memory (SQLite Checkpointing)
- **Phase 4**: PDF Resume Parser & Multi-format Document Processing
- **Phase 5**: FastAPI REST Endpoints & Docker Containerization
