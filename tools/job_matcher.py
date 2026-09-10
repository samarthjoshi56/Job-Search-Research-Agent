from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class JobMatchResult(BaseModel):
    """Structured job matching output."""
    matching_skills: List[str] = Field(description="Skills required by the job that the candidate possesses")
    missing_skills: List[str] = Field(description="Skills required by the job that appear missing from the candidate's profile")
    relevant_experience: List[str] = Field(description="Candidate experiences that directly align with key job responsibilities")
    potential_gaps: List[str] = Field(description="Identified gaps in experience, education, or core qualifications")

@tool
def match_job(job_requirements: List[str], candidate_skills: List[str], experience_summary: str) -> dict:
    """
    Compares job requirements against candidate skills and experience summary,
    returning structured insights on matching skills, missing skills, relevant experience, and potential gaps.
    """
    if not job_requirements:
        return {
            "matching_skills": candidate_skills,
            "missing_skills": [],
            "relevant_experience": [experience_summary] if experience_summary else [],
            "potential_gaps": ["No specific job requirements provided to compare against."]
        }

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR analyst. Compare the provided job requirements "
                       "with the candidate's skills and experience. Identify matching skills, "
                       "missing skills, relevant experience, and potential gaps."),
            ("human", "Job Requirements:\n{job_requirements}\n\nCandidate Skills:\n{candidate_skills}\n\nExperience Summary:\n{experience_summary}")
        ])

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        chain = prompt | llm.with_structured_output(JobMatchResult)
        result = chain.invoke({
            "job_requirements": "\n".join(f"- {r}" for r in job_requirements),
            "candidate_skills": ", ".join(candidate_skills) if candidate_skills else "None listed",
            "experience_summary": experience_summary if experience_summary else "None listed"
        })
        return result.model_dump()
    except Exception as e:
        return {
            "error": f"Job matching failed: {str(e)}",
            "matching_skills": [],
            "missing_skills": job_requirements,
            "relevant_experience": [],
            "potential_gaps": ["Error during matching analysis."]
        }
