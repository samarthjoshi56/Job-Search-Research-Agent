import json
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.planner import planner
from agent.generator import response_generator
from tools import ALL_TOOLS, search_web, parse_resume, match_job, store_research

MAX_ITERATIONS = 5

def planner_node(state: AgentState) -> dict:
    """Runs the planner and sets up initial agent state."""
    result = planner(state)
    
    # Initialize messages and counter if not already present
    messages = state.get("messages", [])
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
    
    user_content = f"Job Description:\n{job_desc}"
    if resume_txt and resume_txt.strip():
        user_content += f"\n\nCandidate Resume:\n{resume_txt}"
        
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    return {
        "plan": result.get("plan", {}),
        "messages": initial_messages,
        "tool_results": state.get("tool_results", []),
        "research": state.get("research", {}),
        "resume_data": state.get("resume_data", {}),
        "job_match": state.get("job_match", {}),
        "iteration_count": 0
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
        "iteration_count": iteration_count
    }

def tool_execution_node(state: AgentState) -> dict:
    """Executes requested tool calls and updates state structures."""
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
                
            new_messages.append(ToolMessage(
                content=json.dumps(tool_output, default=str),
                tool_call_id=call_id,
                name=name
            ))
            
    return {
        "messages": messages + new_messages,
        "tool_results": tool_results,
        "research": research,
        "resume_data": resume_data,
        "job_match": job_match
    }

def should_continue(state: AgentState) -> Literal["tool_execution", "response_generator"]:
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

def create_agent_graph():
    """
    Constructs and compiles the Phase 2 tool-using agent graph.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_agent", tool_agent_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("response_generator", response_generator)
    
    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "tool_agent")
    
    workflow.add_conditional_edges(
        "tool_agent",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "response_generator": "response_generator"
        }
    )
    
    workflow.add_edge("tool_execution", "tool_agent")
    workflow.add_edge("response_generator", END)
    
    return workflow.compile()
