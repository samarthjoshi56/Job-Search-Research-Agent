"""
main.py – Phase 3 CLI entry point.

Commands
--------
  python main.py new         – Start a new job research session.
  python main.py resume JOB_ID – Resume an interrupted session.
  python main.py list        – List all saved job sessions.
  python main.py             – Defaults to 'new' (backwards compatible).

Environment
-----------
  OPENAI_API_KEY   (required)
  TAVILY_API_KEY   (optional, enables web search)
  JOB_AGENT_DB_PATH (optional, default: data/agent.db)
"""

import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure data/ directory exists before anything else
# ---------------------------------------------------------------------------
_data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_data_dir, exist_ok=True)


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("Create a .env file based on .env.example.", file=sys.stderr)
        sys.exit(1)


def _read_multiline(prompt: str) -> str:
    print(prompt)
    print("(Press Ctrl+D / Ctrl+Z on a new line when done)")
    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def _read_line(prompt: str) -> str:
    try:
        sys.stdin = open("/dev/tty")
    except Exception:
        pass
    print(prompt, end=" ", flush=True)
    try:
        return sys.stdin.readline().strip()
    except Exception:
        return ""


def _print_result(final_response: dict) -> None:
    print("=" * 60)
    print("FINAL JOB SEARCH & CANDIDATE RESEARCH ANALYSIS")
    print("=" * 60)

    print("\n[SUMMARY]")
    print(final_response.get("summary", "N/A"))

    print("\n[REQUIREMENTS]")
    for req in final_response.get("requirements", []):
        print(f"  - {req}")

    print("\n[INITIAL ASSESSMENT]")
    print(final_response.get("initial_assessment", "N/A"))

    if final_response.get("company_research"):
        print("\n[COMPANY RESEARCH]")
        print(final_response["company_research"])

    if final_response.get("candidate_match"):
        print("\n[CANDIDATE MATCH ANALYSIS]")
        match = final_response["candidate_match"]
        if isinstance(match, dict):
            print(f"  Matching Skills : {', '.join(match.get('matching_skills', [])) or 'None'}")
            print(f"  Missing Skills  : {', '.join(match.get('missing_skills', [])) or 'None'}")
            print(f"  Relevant Exp    : {', '.join(match.get('relevant_experience', [])) or 'None'}")
            print(f"  Potential Gaps  : {', '.join(match.get('potential_gaps', [])) or 'None'}")
        else:
            print(str(match))

    if final_response.get("sources"):
        print("\n[SOURCES / REFERENCES]")
        for src in final_response["sources"]:
            print(f"  - {src}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_new() -> None:
    """Start a completely new job research session."""
    from agent.graph import create_agent_graph
    from storage.database import create_job, init_db

    job_description = _read_multiline("\nPaste the job description below:")
    if not job_description:
        print("Error: No job description provided.", file=sys.stderr)
        sys.exit(1)

    resume_text = _read_line("\nOptional – paste resume text (or press Enter to skip):")

    job_id = str(uuid.uuid4())
    init_db()
    create_job(
        job_id=job_id,
        job_description=job_description,
        resume_text=resume_text,
    )
    print(f"\nJob session created: {job_id}")

    _run_graph(job_id, job_description, resume_text)


def cmd_resume(job_id: str) -> None:
    """Resume an interrupted job research session."""
    from agent.graph import create_agent_graph
    from storage.database import get_job, load_research, load_candidate_analysis

    job = get_job(job_id)
    if not job:
        print(f"Error: No job found with ID '{job_id}'.", file=sys.stderr)
        sys.exit(1)

    print(f"\nResuming job: {job_id}")
    print(f"  Status: {job['status']}")
    print(f"  Created: {job['created_at']}")

    prior_research = load_research(job_id)
    prior_analysis = load_candidate_analysis(job_id) or {}

    if prior_research:
        print(f"\nLoaded {len(prior_research)} prior research fragments from DB.")
    if prior_analysis.get("resume_data"):
        print("Loaded prior resume analysis from DB.")

    _run_graph(
        job_id=job_id,
        job_description=job["job_description"],
        resume_text=job.get("resume_text") or "",
        prior_research=prior_research,
        prior_analysis=prior_analysis,
    )


def cmd_list() -> None:
    """List all saved job sessions."""
    from storage.database import list_jobs

    jobs = list_jobs()
    if not jobs:
        print("No job sessions found.")
        return

    print(f"\n{'ID':<38} {'STATUS':<12} {'CREATED':<28}")
    print("-" * 80)
    for j in jobs:
        created = j.get("created_at", "")[:19].replace("T", " ")
        print(f"{j['job_id']:<38} {j['status']:<12} {created}")


def _run_graph(
    job_id: str,
    job_description: str,
    resume_text: str = "",
    prior_research: dict | None = None,
    prior_analysis: dict | None = None,
) -> None:
    """Build the graph and invoke it, printing the final result."""
    from agent.graph import create_agent_graph
    from storage.database import update_job_status

    app = create_agent_graph()

    initial_state = {
        "job_id": job_id,
        "job_description": job_description,
        "resume_text": resume_text or None,
        "messages": [],
        "plan": {},
        "tool_results": (prior_analysis or {}).get("tool_results") or [],
        "research": prior_research or {},
        "resume_data": (prior_analysis or {}).get("resume_data") or {},
        "job_match": (prior_analysis or {}).get("job_match") or {},
        "final_response": {},
        "iteration_count": 0,
    }

    print("\nRunning Research & Analysis...\n")
    try:
        update_job_status(job_id=job_id, status="running")
        result_state = app.invoke(initial_state)
        final_response = result_state.get("final_response", {})
        _print_result(final_response)
        print(f"\nSession saved. job_id = {job_id}")
    except KeyboardInterrupt:
        update_job_status(job_id=job_id, status="interrupted")
        print(f"\nInterrupted. Resume with: python main.py resume {job_id}")
        sys.exit(0)
    except Exception as e:
        update_job_status(job_id=job_id, status="error")
        print(f"Error during graph execution: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    _require_api_key()

    args = sys.argv[1:]
    command = args[0].lower() if args else "new"

    print("=== Job Search Research Agent (Phase 3) ===")

    if command == "list":
        cmd_list()
    elif command == "resume":
        if len(args) < 2:
            print("Usage: python main.py resume <JOB_ID>", file=sys.stderr)
            sys.exit(1)
        cmd_resume(args[1])
    else:
        # 'new' or no argument – backwards compatible default
        cmd_new()


if __name__ == "__main__":
    main()
