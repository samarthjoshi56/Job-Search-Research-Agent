from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState

class FinalResponse(BaseModel):
    """The final structured analysis of the job description."""
    summary: str = Field(description="A concise summary of the job role and company")
    requirements: List[str] = Field(description="A list of key requirements (skills, experience, etc.)")
    initial_assessment: str = Field(description="A preliminary assessment of the job's main challenges and appeal")

def response_generator(state: AgentState) -> dict:
    """
    Takes the structured plan and generates a final concise preliminary analysis.
    """
    plan = state.get("plan", {})
    job_description = state.get("job_description", "")

    # Convert the plan dict back to a readable string for the prompt
    plan_text = "\n".join(f"{k}: {v}" for k, v in plan.items())

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter. Based on the original job description "
                   "and the extracted key information (the plan), generate a final structured "
                   "analysis consisting of a summary, a list of combined key requirements, and "
                   "an initial assessment."),
        ("human", "Original Job Description:\n{job_description}\n\nExtracted Plan:\n{plan_text}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm.with_structured_output(FinalResponse)
    
    final_result = chain.invoke({
        "job_description": job_description,
        "plan_text": plan_text
    })

    return {"final_response": final_result.model_dump()}
