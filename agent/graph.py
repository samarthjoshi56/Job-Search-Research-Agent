"""
agent/graph.py – Phase 3 LangGraph workflow with SQLite persistence.

Changes from Phase 2:
  * planner_node  – loads previously saved research/candidate analysis from DB
                    before running the LLM, so interrupted workflows pick up
                    where they left off without duplicating tool calls.
  * tool_execution_node – after updating in-memory state, persists research
                           and candidate analysis to SQLite.
  * response_generator_node wrapper – persists the final response and marks
                                      the job as 'complete'.
  * create_agent_graph() – accepts an optional checkpointer for LangGraph's
                            native checkpointing on top of our SQLite layer.
"""

import json
import os
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.planner import planner
from agent.generator import response_generator
from agent.state import AgentState
from tools import ALL_TOOLS, match_job, parse_resume, search_web, store_research

MAX_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Persistence helpers (imported lazily to avoid breaking tests that don't use DB)
# ---------------------------------------------------------------------------

def _db_available() -> bool:
    """Return True when the storage package can be imported."""
    try:
        import storage  # noqa: F401
        return True
    except ImportError:
        return False


def _load_previous_state(job_id: str) -> dict:
    """
    Load previously saved research, candidate analysis, and tool results
    for the given job_id.  Returns an empty dict if nothing is found.
    """
    if not job_id or not _db_available():
        return {}
    try:
        from storage.database import load_research, load_candidate_analysis
        research = load_research(job_id)
        analysis = load_candidate_analysis(job_id) or {}
        return {
            "research": research,
            "resume_data": analysis.get("resume_data") or {},
            "job_match": analysis.get("job_match") or {},
            "tool_results": analysis.get("tool_results") or [],
        }
    except Exception:
        return {}


def _persist_research(job_id: str, research: dict) -> None:
    if not job_id or not _db_available():
        return
    try:
        from storage.database import save_research
        for key, value in research.items():
            save_research(job_id=job_id, key=key, value=value)
    except Exception:
        pass


def _persist_candidate_analysis(
    job_id: str,
    resume_data: dict,
    job_match: dict,
    tool_results: list,
) -> None:
    if not job_id or not _db_available():
        return
    try:
        from storage.database import save_candidate_analysis
        save_candidate_analysis(
            job_id=job_id,
            resume_data=resume_data,
            job_match=job_match,
            tool_results=tool_results,
        )
    except Exception:
        pass


def _persist_final_response(job_id: str, response: dict) -> None:
    if not job_id or not _db_available():
        return
    try:
        from storage.database import save_final_response, update_job_status
        save_final_response(job_id=job_id, response=response)
        update_job_status(job_id=job_id, status="complete")
    except Exception:
        pass


def _mark_running(job_id: str) -> None:
    if not job_id or not _db_available():
        return
    try:
        from storage.database import update_job_status
        update_job_status(job_id=job_id, status="running")
    except Exception:
        pass


def _mark_interrupted(job_id: str) -> None:
    if not job_id or not _db_available():
        return
    try:
        from storage.database import update_job_status
        update_job_status(job_id=job_id, status="interrupted")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    Phase 3: Load any previously saved research from DB before planning.
    This prevents duplicate tool calls when resuming an interrupted workflow.
    """
    job_id = state.get("job_id", "")

    # --- Phase 3: restore prior state from DB ---
    prior = _load_previous_state(job_id)
    existing_research = prior.get("research", {})
    existing_resume_data = prior.get("resume_data", {})
    existing_job_match = prior.get("job_match", {})
    existing_tool_results = prior.get("tool_results", [])

    # Merge with anything already in memory (in-memory takes precedence)
    merged_research = {**existing_research, **state.get("research", {})}
    merged_resume_data = {**existing_resume_data, **state.get("resume_data", {})}
    merged_job_match = {**existing_job_match, **state.get("job_match", {})}
    merged_tool_results = existing_tool_results or state.get("tool_results", [])

    _mark_running(job_id)

    result = planner(state)

    job_desc = state.get("job_description", "")
    resume_txt = state.get("resume_text", "")

    system_prompt = (
        "You are an intelligent job research agent. Your goal is to analyze the job description and "
        "candidate profile using available tools.\n"
        "Available tools:\n"
        "- search_web: Search for company or role details.\n"
        "- parse_resume: Parse candidate resume text into structured fields.\n"
        "- match_job: Compare job requirements against candidate skills and experience.\n"
        "- store_research: Organize key research notes into state.\n\n"
        "Instructions:\n"
        "1. If a resume is provided, parse it and compare it with the job requirements.\n"
        "2. If company background or specific details are needed, use search_web.\n"
        "3. Store key findings using store_research if helpful.\n"
        "4. Once enough information has been gathered, respond with a final text summary and stop tool calling."
    )

    # If we have prior research, hint to the agent so it avoids duplicate work.
    if merged_research:
        prior_summary = json.dumps(merged_research, indent=2)
        system_prompt += (
            f"\n\nNote: The following research was already collected in a previous session. "
            f"Do NOT re-fetch this information:\n{prior_summary}"
        )

    user_content = f"Job Description:\n{job_desc}"
    if resume_txt and resume_txt.strip():
        user_content += f"\n\nCandidate Resume:\n{resume_txt}"

    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    return {
        "plan": result.get("plan", {}),
        "messages": initial_messages,
        "tool_results": merged_tool_results,
        "research": merged_research,
        "resume_data": merged_resume_data,
        "job_match": merged_job_match,
        "iteration_count": 0,
    }


def tool_agent_node(state: AgentState) -> dict:
    """Agent node that decides whether to invoke tools or finish."""
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0) + 1

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    response = llm_with_tools.invoke(messages)

    return {
        "messages": messages + [response],
        "iteration_count": iteration_count,
    }


def tool_execution_node(state: AgentState) -> dict:
    """
    Executes requested tool calls, updates in-memory state structures,
    and persists research + candidate analysis to SQLite (Phase 3).
    """
    job_id = state.get("job_id", "")
    messages = state.get("messages", [])
    last_message = messages[-1]

    tool_results = list(state.get("tool_results", []))
    research = dict(state.get("research", {}))
    resume_data = dict(state.get("resume_data", {}))
    job_match = dict(state.get("job_match", {}))

    new_messages = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            tool_output = None
            if name == "search_web":
                tool_output = search_web.invoke(args)
                tool_results.append(tool_output)
            elif name == "parse_resume":
                tool_output = parse_resume.invoke(args)
                if isinstance(tool_output, dict):
                    resume_data.update(tool_output)
            elif name == "match_job":
                tool_output = match_job.invoke(args)
                if isinstance(tool_output, dict):
                    job_match.update(tool_output)
            elif name == "store_research":
                tool_output = store_research.invoke(args)
                if isinstance(tool_output, dict) and "key" in tool_output:
                    research[tool_output["key"]] = tool_output["data"]
            else:
                tool_output = {"error": f"Unknown tool {name}"}

            new_messages.append(
                ToolMessage(
                    content=json.dumps(tool_output, default=str),
                    tool_call_id=call_id,
                    name=name,
                )
            )

    # --- Phase 3: persist after every tool round ---
    _persist_research(job_id, research)
    _persist_candidate_analysis(job_id, resume_data, job_match, tool_results)

    return {
        "messages": messages + new_messages,
        "tool_results": tool_results,
        "research": research,
        "resume_data": resume_data,
        "job_match": job_match,
    }


def _response_generator_node(state: AgentState) -> dict:
    """
    Thin wrapper around response_generator that persists the final result.
    """
    result = response_generator(state)
    job_id = state.get("job_id", "")
    final = result.get("final_response", {})
    _persist_final_response(job_id, final)
    return result


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(
    state: AgentState,
) -> Literal["tool_execution", "response_generator"]:
    """Determines whether to execute tools or proceed to final response generator."""
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0)

    if iteration_count >= MAX_ITERATIONS:
        return "response_generator"

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_execution"

    return "response_generator"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def create_agent_graph(checkpointer=None):
    """
    Constructs and compiles the Phase 3 tool-using agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for native graph-level
                      state persistence (e.g. MemorySaver for tests).
                      Our SQLite layer operates independently of this.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_agent", tool_agent_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("response_generator", _response_generator_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "tool_agent")

    workflow.add_conditional_edges(
        "tool_agent",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "response_generator": "response_generator",
        },
    )

    workflow.add_edge("tool_execution", "tool_agent")
    workflow.add_edge("response_generator", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
