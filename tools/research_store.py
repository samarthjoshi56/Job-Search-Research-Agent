from typing import Dict, Any
from langchain_core.tools import tool

@tool
def store_research(key: str, data: str) -> Dict[str, Any]:
    """
    Stores key research notes or collected information into the research state.
    """
    return {
        "status": "success",
        "key": key.strip(),
        "data": data.strip()
    }
