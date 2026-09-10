import pytest
from unittest.mock import patch, MagicMock
from tools.web_search import search_web
from tools.resume_parser import parse_resume, ParsedResume
from tools.job_matcher import match_job, JobMatchResult
from tools.research_store import store_research

# 1. Web Search Tool Tests
def test_web_search_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        res = search_web.invoke({"query": "Python developer"})
        assert len(res) == 1
        assert res[0]["title"] == "Tavily API Key Missing"

@patch("tavily.TavilyClient")
def test_web_search_success(mock_tavily_client):
    mock_instance = MagicMock()
    mock_instance.search.return_value = {
        "results": [
            {"title": "Company Info", "url": "https://example.com", "content": "Tech startup"}
        ]
    }
    mock_tavily_client.return_value = mock_instance

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        res = search_web.invoke({"query": "Acme Corp"})
        assert len(res) == 1
        assert res[0]["title"] == "Company Info"
        assert res[0]["url"] == "https://example.com"
        assert res[0]["snippet"] == "Tech startup"

@patch("tavily.TavilyClient")
def test_web_search_error_handling(mock_tavily_client):
    mock_instance = MagicMock()
    mock_instance.search.side_effect = Exception("API rate limit")
    mock_tavily_client.return_value = mock_instance

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        res = search_web.invoke({"query": "Acme Corp"})
        assert len(res) == 1
        assert res[0]["title"] == "Web Search Error"
        assert "API rate limit" in res[0]["snippet"]

# 2. Resume Parser Tool Tests
def test_parse_resume_empty_input():
    res = parse_resume.invoke({"resume_text": ""})
    assert res["education"] == "Not provided"
    assert res["skills"] == []

@patch("tools.resume_parser.ChatOpenAI")
def test_parse_resume_success(mock_chat):
    mock_chain = MagicMock()
    mock_parsed = ParsedResume(
        skills=["Python", "FastAPI"],
        programming_languages=["Python"],
        frameworks=["LangChain"],
        experience=["Senior Developer at Tech Corp"],
        education="BS Computer Science",
        projects=["AI Assistant"]
    )
    mock_chain.invoke.return_value = mock_parsed

    with patch("tools.resume_parser.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        res = parse_resume.invoke({"resume_text": "Experienced Python dev."})
        assert "Python" in res["skills"]
        assert res["education"] == "BS Computer Science"

# 3. Job Matcher Tool Tests
def test_job_matcher_empty_requirements():
    res = match_job.invoke({
        "job_requirements": [],
        "candidate_skills": ["Python"],
        "experience_summary": "3 years experience"
    })
    assert res["matching_skills"] == ["Python"]
    assert "No specific job requirements" in res["potential_gaps"][0]

@patch("tools.job_matcher.ChatOpenAI")
def test_job_matcher_success(mock_chat):
    mock_chain = MagicMock()
    mock_match = JobMatchResult(
        matching_skills=["Python"],
        missing_skills=["Kubernetes"],
        relevant_experience=["Built microservices"],
        potential_gaps=["No DevOps experience"]
    )
    mock_chain.invoke.return_value = mock_match

    with patch("tools.job_matcher.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        res = match_job.invoke({
            "job_requirements": ["Python", "Kubernetes"],
            "candidate_skills": ["Python"],
            "experience_summary": "Backend dev"
        })
        assert res["matching_skills"] == ["Python"]
        assert res["missing_skills"] == ["Kubernetes"]

# 4. Research Store Tool Tests
def test_store_research():
    res = store_research.invoke({"key": "company_culture", "data": "Remote-first workforce"})
    assert res["status"] == "success"
    assert res["key"] == "company_culture"
    assert res["data"] == "Remote-first workforce"
