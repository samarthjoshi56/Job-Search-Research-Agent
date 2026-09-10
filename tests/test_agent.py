import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from agent.state import AgentState
from agent.planner import ExtractionPlan, planner
from agent.generator import FinalResponse, response_generator
from agent.graph import create_agent_graph, should_continue, tool_execution_node, MAX_ITERATIONS

# Test state structure
def test_agent_state_creation():
    state: AgentState = {
        "job_description": "Test JD",
        "resume_text": "Test Resume",
        "messages": [],
        "plan": {},
        "tool_results": [],
        "research": {},
        "resume_data": {},
        "job_match": {},
        "final_response": {},
        "iteration_count": 0
    }
    assert state["job_description"] == "Test JD"
    assert state["resume_text"] == "Test Resume"
    assert isinstance(state["plan"], dict)
    assert isinstance(state["tool_results"], list)

# Test Planner Output Structure (Mocking LLM)
@patch("agent.planner.ChatOpenAI")
def test_planner_output_structure(mock_chat):
    mock_llm_instance = MagicMock()
    mock_chain = MagicMock()
    
    mock_plan = ExtractionPlan(
        required_technical_skills=["Python", "Pytest"],
        years_of_experience="3+",
        education_requirements="BS CS",
        location="Remote",
        important_responsibilities=["Testing"],
        preferred_skills=["Docker"]
    )
    
    mock_chain.invoke.return_value = mock_plan
    mock_llm_instance.with_structured_output.return_value = mock_chain
    mock_chat.return_value = mock_llm_instance

    with patch("agent.planner.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        state = {"job_description": "We need a Python tester.", "messages": [], "plan": {}, "final_response": {}}
        result = planner(state)
        assert "plan" in result
        assert result["plan"]["required_technical_skills"] == ["Python", "Pytest"]
        assert result["plan"]["location"] == "Remote"

# Test Generator Output Structure (Mocking LLM)
@patch("agent.generator.ChatOpenAI")
def test_generator_output_structure(mock_chat):
    mock_chain = MagicMock()
    mock_response = FinalResponse(
        summary="Test Summary",
        requirements=["Python"],
        initial_assessment="Looks good.",
        company_research="Innovative startup",
        candidate_match={"matching_skills": ["Python"]},
        sources=["Source: https://example.com"]
    )
    mock_chain.invoke.return_value = mock_response

    with patch("agent.generator.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        state = {
            "job_description": "JD",
            "messages": [],
            "plan": {"required_technical_skills": ["Python"]},
            "tool_results": [{"title": "Source", "url": "https://example.com"}],
            "final_response": {}
        }
        result = response_generator(state)
        assert "final_response" in result
        assert result["final_response"]["summary"] == "Test Summary"
        assert result["final_response"]["sources"] == ["Source: https://example.com"]

# Test Graph Creation
def test_graph_creation():
    app = create_agent_graph()
    assert app is not None

# Test Missing API Key / Configuration Handling
def test_missing_api_key_handling():
    with patch.dict(os.environ, {}, clear=True):
        assert os.getenv("OPENAI_API_KEY") is None
        from main import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

# Test Conditional Logic & Termination
def test_should_continue_logic():
    # Case 1: Max iterations reached -> terminate to generator
    state_max: AgentState = {"messages": [AIMessage(content="test")], "iteration_count": MAX_ITERATIONS}
    assert should_continue(state_max) == "response_generator"

    # Case 2: Tool call requested -> continue to tool execution
    tool_call_msg = AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "test"}, "id": "call_1"}])
    state_tool: AgentState = {"messages": [tool_call_msg], "iteration_count": 1}
    assert should_continue(state_tool) == "tool_execution"

    # Case 3: Plain response without tool calls -> terminate to generator
    state_plain: AgentState = {"messages": [AIMessage(content="Done research.")], "iteration_count": 1}
    assert should_continue(state_plain) == "response_generator"

# Test Tool Execution Node State Update
@patch("agent.graph.search_web")
def test_tool_execution_node_updates_state(mock_search):
    mock_search.invoke.return_value = [{"title": "Info", "url": "https://test.com", "snippet": "Details"}]
    
    tool_msg = AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "Acme"}, "id": "call_abc"}])
    state: AgentState = {
        "messages": [tool_msg],
        "tool_results": [],
        "research": {},
        "resume_data": {},
        "job_match": {}
    }
    
    updated = tool_execution_node(state)
    assert len(updated["tool_results"]) == 1
    assert updated["tool_results"][0][0]["title"] == "Info"
    assert len(updated["messages"]) == 2

