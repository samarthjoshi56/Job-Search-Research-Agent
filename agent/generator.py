from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState

class FinalResponse(BaseModel):
    """The final structured analysis of the job description and candidate match."""
    summary: str = Field(description="A concise summary of the job role and company")
    requirements: List[str] = Field(description="A list of key requirements (skills, experience, etc.)")
    initial_assessment: str = Field(description="A preliminary assessment of the job's main challenges and appeal")
    company_research: Optional[str] = Field(default=None, description="Researched insights about the company or industry")
    candidate_match: Optional[Dict[str, Any]] = Field(default=None, description="Match analysis comparing job requirements with resume")
    sources: List[str] = Field(default_factory=list, description="Source URLs or references grounding the research")

def response_generator(state: AgentState) -> dict:
    """
    Takes the structured plan, tool results, research, and candidate match data to generate a final grounded analysis.
    """
    plan = state.get("plan", {})
    job_description = state.get("job_description", "")
    resume_text = state.get("resume_text", "")
    tool_results = state.get("tool_results", [])
    research = state.get("research", {})
    job_match = state.get("job_match", {})
    resume_data = state.get("resume_data", {})

    plan_text = "\n".join(f"{k}: {v}" for k, v in plan.items())
    tool_text = "\n".join(str(res) for res in tool_results) if tool_results else "None"
    research_text = "\n".join(f"{k}: {v}" for k, v in research.items()) if research else "None"
    match_text = str(job_match) if job_match else "None"

    # Extract source URLs from tool_results
    sources = []
    for res in tool_results:
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and item.get("url") and item["url"] != "N/A":
                    sources.append(f"{item.get('title', 'Source')}: {item['url']}")
        elif isinstance(res, dict) and res.get("url") and res["url"] != "N/A":
            sources.append(f"{res.get('title', 'Source')}: {res['url']}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter and career strategist. "
                   "Analyze the job description, extracted plan, tool research results, "
                   "and candidate matching details. Produce a comprehensive structured analysis. "
                   "Ground company insights in the provided research data. Do not invent sources."),
        ("human", "Job Description:\n{job_description}\n\n"
                  "Candidate Resume:\n{resume_text}\n\n"
                  "Plan:\n{plan_text}\n\n"
                  "Tool Results:\n{tool_text}\n\n"
                  "Research Collected:\n{research_text}\n\n"
                  "Job Match Analysis:\n{match_text}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm.with_structured_output(FinalResponse)
    
    final_result = chain.invoke({
        "job_description": job_description,
        "resume_text": resume_text if resume_text else "Not provided",
        "plan_text": plan_text,
        "tool_text": tool_text,
        "research_text": research_text,
        "match_text": match_text
    })

    result_dict = final_result.model_dump()
    # Ensure sources from web search tools are included if missing
    if sources and not result_dict.get("sources"):
        result_dict["sources"] = list(set(sources))

    return {"final_response": result_dict}
