"""
storage package – Phase 3 SQLite persistence layer.
"""
from storage.database import (
    init_db,
    create_job,
    get_job,
    list_jobs,
    update_job_status,
    save_research,
    load_research,
    save_candidate_analysis,
    load_candidate_analysis,
    save_final_response,
    load_final_response,
)

__all__ = [
    "init_db",
    "create_job",
    "get_job",
    "list_jobs",
    "update_job_status",
    "save_research",
    "load_research",
    "save_candidate_analysis",
    "load_candidate_analysis",
    "save_final_response",
    "load_final_response",
]
