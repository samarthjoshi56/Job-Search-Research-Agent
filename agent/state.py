from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """
    Represents the state of our job-search research agent (Phase 3).

    New in Phase 3:
        job_id – UUID that ties this workflow run to a persisted DB record.
                 When provided, the agent loads previous research from SQLite
                 before executing and saves incremental updates back.
    """
    # Core inputs
    job_id: str               # Phase 3: unique identifier for this job session
    job_description: str
    resume_text: Optional[str]

    # Workflow internals
    messages: List[Any]
    plan: Dict[str, Any]
    iteration_count: int

    # Tool outputs
    tool_results: List[Dict[str, Any]]
    research: Dict[str, Any]
    resume_data: Dict[str, Any]
    job_match: Dict[str, Any]

    # Final output
    final_response: Dict[str, Any]
