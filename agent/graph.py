from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.planner import planner
from agent.generator import response_generator

def create_agent_graph():
    """
    Constructs and returns the LangGraph workflow.
    """
    # Initialize the graph with our custom state
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("planner", planner)
    workflow.add_node("response_generator", response_generator)

    # Set the entry point
    workflow.set_entry_point("planner")

    # Define the edges
    workflow.add_edge("planner", "response_generator")
    workflow.add_edge("response_generator", END)

    # Compile the graph
    app = workflow.compile()
    
    return app
