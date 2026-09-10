from typing import TypedDict, List, Any

class AgentState(TypedDict):
    """
    Represents the state of our job-search research agent.
    """
    job_description: str
    messages: List[Any]
    plan: dict
    final_response: dict
