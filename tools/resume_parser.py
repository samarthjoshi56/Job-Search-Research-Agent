from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ParsedResume(BaseModel):
    """Structured candidate resume details."""
    skills: List[str] = Field(description="General technical and soft skills")
    programming_languages: List[str] = Field(description="Programming languages mastered")
    frameworks: List[str] = Field(description="Frameworks and libraries used")
    experience: List[str] = Field(description="Summary of work experience and positions held")
    education: Optional[str] = Field(description="Highest degree or educational background")
    projects: List[str] = Field(description="Key projects mentioned in the resume")

@tool
def parse_resume(resume_text: str) -> dict:
    """
    Parses raw resume text and extracts structured candidate information
    including skills, programming languages, frameworks, experience, education, and projects.
    """
    if not resume_text or not resume_text.strip():
        return {
            "skills": [],
            "programming_languages": [],
            "frameworks": [],
            "experience": [],
            "education": "Not provided",
            "projects": []
        }

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR resume parser. Extract structured details "
                       "from the provided resume text. If a category is absent, leave it empty."),
            ("human", "{resume_text}")
        ])

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        chain = prompt | llm.with_structured_output(ParsedResume)
        result = chain.invoke({"resume_text": resume_text})
        return result.model_dump()
    except Exception as e:
        return {
            "error": f"Failed to parse resume: {str(e)}",
            "skills": [],
            "programming_languages": [],
            "frameworks": [],
            "experience": [],
            "education": "Parsing error",
            "projects": []
        }
