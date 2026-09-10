# Job Search Research Agent

A Python + LangGraph-based research agent designed to help users evaluate, analyze, and process job descriptions efficiently.

> **Note:** This repository currently represents **Phase 1** of the project, focusing exclusively on the core agent architecture. Web search, tool usage, database persistence, resume parsing, FastAPI endpoints, Docker support, and automated evaluation will be introduced in subsequent phases.

---

## 🏗️ Architecture

Phase 1 establishes the fundamental flow of the research agent using **LangGraph**:

```text
User Job Description
        │
        ▼
     Planner (LLM extracts structured criteria)
        │
        ▼
    Agent State (TypedDict containing job_description, plan, etc.)
        │
        ▼
 Response Generator (LLM synthesizes summary, requirements, and assessment)
        │
        ▼
   Final Output
```

### Graph Components
1. **Agent State (`agent/state.py`)**: A `TypedDict` structure holding `job_description`, `messages`, `plan`, and `final_response`.
2. **Planner Node (`agent/planner.py`)**: Uses OpenAI LLM structured output (`Pydantic`) to extract required skills, experience, education, location, responsibilities, and preferred skills.
3. **Response Generator Node (`agent/generator.py`)**: Uses the structured plan and job description to produce a concise summary, key requirements list, and an initial job assessment.
4. **Graph Execution (`agent/graph.py`)**: Configures the state transition: `planner` ➔ `response_generator` ➔ `END`.

---

## 🛠️ Tech Stack
- **Python**: 3.11+
- **LangGraph**: Workflow orchestration
- **LangChain / LangChain OpenAI**: LLM integration and structured outputs
- **Pydantic**: Data schema definition and validation
- **python-dotenv**: Environment configuration
- **pytest**: Test suite framework

---

## 📁 Project Structure

```text
Job-Search-Research-Agent/
│
├── agent/
│   ├── __init__.py      # Package initialization
│   ├── state.py         # TypedDict state definition
│   ├── planner.py       # Planner node implementation
│   ├── graph.py         # LangGraph workflow compiler
│   └── generator.py     # Response generator node implementation
│
├── tests/
│   ├── __init__.py      # Test package initialization
│   └── test_agent.py    # Unit tests for state, planner, generator, and graph
│
├── .env.example         # Template for environment variables
├── .gitignore            # Git ignore configuration
├── requirements.txt     # Python dependencies
├── main.py              # CLI entry point
└── README.md            # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/samarthjoshi56/Job-Search-Research-Agent.git
cd Job-Search-Research-Agent
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your OpenAI API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 💻 How to Run

Run the simple CLI entry point:
```bash
python main.py
```
You will be prompted to paste a job description. Once done, press `Ctrl+D` (or `Ctrl+Z` on Windows) on a new line to process.

---

## 🧪 Running Tests

Unit tests mock external LLM calls to verify graph structure, state validation, and output parsing without requiring an active API key:

```bash
pytest
```

---

## 🔮 Roadmap / Future Phases
- **Phase 2**: Web Search Integration (Tavily/Google Search) & External Tools
- **Phase 3**: Resume Parsing & Matching Analysis
- **Phase 4**: Vector DB (FAISS/Chromadb) & Persistence (SQLite)
- **Phase 5**: FastAPI REST API & Dockerization
