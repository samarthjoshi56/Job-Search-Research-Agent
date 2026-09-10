import os
import sys
from dotenv import load_dotenv
from agent.graph import create_agent_graph

def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please create a .env file based on .env.example and add your API key.", file=sys.stderr)
        sys.exit(1)

    print("=== Job Search Research Agent (Phase 2) ===")
    print("Please paste the job description below (press Ctrl+D or Ctrl+Z on a new line when done):")
    
    try:
        job_description = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)

    if not job_description:
        print("Error: No job description provided.", file=sys.stderr)
        sys.exit(1)

    # Re-open stdin for resume prompt if available
    try:
        sys.stdin = open('/dev/tty')
    except Exception:
        pass

    print("\nOptional: Enter candidate resume text (or press Enter to skip):")
    try:
        resume_text = sys.stdin.readline().strip()
    except Exception:
        resume_text = ""

    print("\nInitializing Tool Agent Graph...")
    app = create_agent_graph()

    print("Running Research & Analysis...\n")
    initial_state = {
        "job_description": job_description,
        "resume_text": resume_text if resume_text else None,
        "messages": [],
        "plan": {},
        "tool_results": [],
        "research": {},
        "resume_data": {},
        "job_match": {},
        "final_response": {},
        "iteration_count": 0
    }

    try:
        result_state = app.invoke(initial_state)
        final_response = result_state.get("final_response", {})
        
        print("=" * 60)
        print("FINAL JOB SEARCH & CANDIDATE RESEARCH ANALYSIS")
        print("=" * 60)
        
        print("\n[SUMMARY]")
        print(final_response.get("summary", "N/A"))
        
        print("\n[REQUIREMENTS]")
        for req in final_response.get("requirements", []):
            print(f"- {req}")
            
        print("\n[INITIAL ASSESSMENT]")
        print(final_response.get("initial_assessment", "N/A"))
        
        if final_response.get("company_research"):
            print("\n[COMPANY RESEARCH]")
            print(final_response.get("company_research"))
            
        if final_response.get("candidate_match"):
            print("\n[CANDIDATE MATCH ANALYSIS]")
            match_data = final_response.get("candidate_match", {})
            if isinstance(match_data, dict):
                print(f"Matching Skills: {', '.join(match_data.get('matching_skills', [])) or 'None'}")
                print(f"Missing Skills: {', '.join(match_data.get('missing_skills', [])) or 'None'}")
                print(f"Relevant Experience: {', '.join(match_data.get('relevant_experience', [])) or 'None'}")
                print(f"Potential Gaps: {', '.join(match_data.get('potential_gaps', [])) or 'None'}")
            else:
                print(str(match_data))

        if final_response.get("sources"):
            print("\n[SOURCES / REFERENCES]")
            for src in final_response.get("sources", []):
                print(f"- {src}")

        print("=" * 60)
        
    except Exception as e:
        print(f"Error during graph execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
