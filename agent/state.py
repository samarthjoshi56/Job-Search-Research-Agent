from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """
    Represents the state of our job-search research agent (Phase 2).
    """
    job_description: str
    resume_text: Optional[str]
    messages: List[Any]
    plan: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    research: Dict[str, Any]
    resume_data: Dict[str, Any]
    job_match: Dict[str, Any]
    final_response: Dict[str, Any]
    iteration_count: int
