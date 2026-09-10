import os
from typing import List, Dict, Any
from langchain_core.tools import tool

@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """
    Search the web for information about a company, job role, or industry.
    Returns a list of structured search results with title, url, and snippet.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{
            "title": "Tavily API Key Missing",
            "url": "N/A",
            "snippet": "Search skipped because TAVILY_API_KEY environment variable is not configured."
        }]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=5)
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", item.get("snippet", ""))
            })
        return results
    except Exception as e:
        return [{
            "title": "Web Search Error",
            "url": "N/A",
            "snippet": f"An error occurred while performing web search: {str(e)}"
        }]
