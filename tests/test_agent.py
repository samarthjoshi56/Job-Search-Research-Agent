import pytest
import os
from unittest.mock import patch, MagicMock
from agent.state import AgentState
from agent.planner import ExtractionPlan, planner
from agent.generator import FinalResponse, response_generator
from agent.graph import create_agent_graph

# Test state structure
def test_agent_state_creation():
    state: AgentState = {
        "job_description": "Test JD",
        "messages": [],
        "plan": {},
        "final_response": {}
    }
    assert state["job_description"] == "Test JD"
    assert isinstance(state["plan"], dict)

# Test Planner Output Structure (Mocking LLM)
@patch('agent.planner.ChatOpenAI')
def test_planner_output_structure(mock_chat):
    # Setup mock
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

    # To mock the pipe operator properly, we actually mock the prompt | llm line or invoke directly
    with patch('agent.planner.ChatPromptTemplate.from_messages') as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        
        state = {"job_description": "We need a Python tester.", "messages": [], "plan": {}, "final_response": {}}
        result = planner(state)
        
        assert "plan" in result
        assert result["plan"]["required_technical_skills"] == ["Python", "Pytest"]
        assert result["plan"]["location"] == "Remote"

# Test Generator Output Structure (Mocking LLM)
@patch('agent.generator.ChatOpenAI')
def test_generator_output_structure(mock_chat):
    mock_chain = MagicMock()
    
    mock_response = FinalResponse(
        summary="Test Summary",
        requirements=["Python"],
        initial_assessment="Looks good."
    )
    mock_chain.invoke.return_value = mock_response

    with patch('agent.generator.ChatPromptTemplate.from_messages') as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        
        state = {
            "job_description": "JD",
            "messages": [],
            "plan": {"required_technical_skills": ["Python"]},
            "final_response": {}
        }
        
        result = response_generator(state)
        
        assert "final_response" in result
        assert result["final_response"]["summary"] == "Test Summary"
        assert result["final_response"]["requirements"] == ["Python"]

# Test Graph Execution Compilation
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

