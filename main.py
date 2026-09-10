import os
import sys
import json
from dotenv import load_dotenv
from agent.graph import create_agent_graph

def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please create a .env file based on .env.example and add your API key.", file=sys.stderr)
        sys.exit(1)

    print("=== Job Search Research Agent (Phase 1) ===")
    print("Please paste the job description below (press Ctrl+D on a new line to submit):")
    
    try:
        job_description = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)

    if not job_description:
        print("Error: No job description provided.", file=sys.stderr)
        sys.exit(1)

    print("\nInitializing Agent Graph...")
    app = create_agent_graph()

    print("\nRunning Analysis...\n")
    initial_state = {
        "job_description": job_description,
        "messages": [],
        "plan": {},
        "final_response": {}
    }

    try:
        # Run the graph
        result_state = app.invoke(initial_state)
        
        # Display results
        final_response = result_state.get("final_response", {})
        
        print("=" * 50)
        print("FINAL ANALYSIS")
        print("=" * 50)
        
        print("\n[SUMMARY]")
        print(final_response.get("summary", "N/A"))
        
        print("\n[REQUIREMENTS]")
        for req in final_response.get("requirements", []):
            print(f"- {req}")
            
        print("\n[INITIAL ASSESSMENT]")
        print(final_response.get("initial_assessment", "N/A"))
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during graph execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
