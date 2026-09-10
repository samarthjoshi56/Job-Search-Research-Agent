from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState

class ExtractionPlan(BaseModel):
    """The structured plan of information to extract from a job description."""
    required_technical_skills: List[str] = Field(description="List of required technical skills")
    years_of_experience: Optional[str] = Field(description="Required years of experience")
    education_requirements: Optional[str] = Field(description="Required education or degrees")
    location: Optional[str] = Field(description="Job location or remote status")
    important_responsibilities: List[str] = Field(description="Key responsibilities of the role")
    preferred_skills: List[str] = Field(description="Preferred but not strictly required skills")

def planner(state: AgentState) -> dict:
    """
    Analyzes the job description and outputs a structured extraction plan.
    """
    job_description = state["job_description"]

    # We use a system prompt to instruct the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter and job analyst. "
                   "Analyze the provided job description and extract the key information "
                   "such as required skills, experience, education, location, responsibilities, "
                   "and preferred skills. If any information is missing, leave it empty or null."),
        ("human", "{job_description}")
    ])

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # We use with_structured_output to ensure we get our Pydantic model back
    chain = prompt | llm.with_structured_output(ExtractionPlan)
    
    plan_result = chain.invoke({"job_description": job_description})

    # Convert Pydantic model to dict for the state
    return {"plan": plan_result.model_dump()}
