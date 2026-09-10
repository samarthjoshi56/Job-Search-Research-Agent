"""
tests/test_storage.py – Phase 3 SQLite persistence tests.

All tests use a temporary database file created via pytest's tmp_path fixture
so they never touch the real data/agent.db and are fully isolated from each other.

Key scenario tested (end-to-end workflow):
  1. create_job        – a job is persisted
  2. save_research     – research fragments are stored mid-workflow
  3. Simulate stop     – update status to 'interrupted'
  4. load_research     – fragments survive a process restart (new connection)
  5. save_candidate_analysis + load – candidate data round-trips cleanly
  6. Resume workflow   – the loaded state equals what was saved
  7. save_final_response + load – final output is persisted
  8. update_job_status to 'complete'
  9. list_jobs         – job appears in the listing
"""

import json
import os
import sqlite3
import pytest

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOB_ID = "test-job-001"
JOB_DESC = "We are looking for a Senior Python Engineer with 5+ years experience."


def _db(tmp_path) -> str:
    """Return a path to a fresh, isolated SQLite file."""
    return str(tmp_path / "test_agent.db")


# ---------------------------------------------------------------------------
# Schema / init tests
# ---------------------------------------------------------------------------

class TestInitDB:
    def test_creates_file(self, tmp_path):
        db = _db(tmp_path)
        init_db(db)
        assert os.path.exists(db)

    def test_idempotent(self, tmp_path):
        db = _db(tmp_path)
        init_db(db)
        init_db(db)  # should not raise
        assert os.path.exists(db)

    def test_all_tables_created(self, tmp_path):
        db = _db(tmp_path)
        init_db(db)
        conn = sqlite3.connect(db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert {"jobs", "research", "candidate_analysis", "final_responses"}.issubset(tables)


# ---------------------------------------------------------------------------
# Jobs table tests
# ---------------------------------------------------------------------------

class TestJobs:
    def test_create_and_get_job(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        job = get_job(JOB_ID, db_path=db)
        assert job is not None
        assert job["job_id"] == JOB_ID
        assert job["job_description"] == JOB_DESC
        assert job["status"] == "new"

    def test_create_job_idempotent(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        create_job(JOB_ID, "Different description", db_path=db)  # should not overwrite
        job = get_job(JOB_ID, db_path=db)
        assert job["job_description"] == JOB_DESC  # original preserved

    def test_get_nonexistent_job_returns_none(self, tmp_path):
        db = _db(tmp_path)
        init_db(db)
        result = get_job("does-not-exist", db_path=db)
        assert result is None

    def test_update_job_status(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        update_job_status(JOB_ID, "running", db_path=db)
        assert get_job(JOB_ID, db_path=db)["status"] == "running"

        update_job_status(JOB_ID, "interrupted", db_path=db)
        assert get_job(JOB_ID, db_path=db)["status"] == "interrupted"

        update_job_status(JOB_ID, "complete", db_path=db)
        assert get_job(JOB_ID, db_path=db)["status"] == "complete"

    def test_list_jobs_empty(self, tmp_path):
        db = _db(tmp_path)
        init_db(db)
        assert list_jobs(db_path=db) == []

    def test_list_jobs_returns_all(self, tmp_path):
        db = _db(tmp_path)
        create_job("job-a", "JD A", db_path=db)
        create_job("job-b", "JD B", db_path=db)
        jobs = list_jobs(db_path=db)
        assert len(jobs) == 2
        ids = {j["job_id"] for j in jobs}
        assert ids == {"job-a", "job-b"}

    def test_create_job_with_resume(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, resume_text="5 years Python.", db_path=db)
        job = get_job(JOB_ID, db_path=db)
        assert job["resume_text"] == "5 years Python."


# ---------------------------------------------------------------------------
# Research table tests
# ---------------------------------------------------------------------------

class TestResearch:
    def test_save_and_load_research(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_research(JOB_ID, "company_culture", "Remote-first", db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert result["company_culture"] == "Remote-first"

    def test_save_research_json_value(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        data = {"skills": ["Python", "LangGraph"], "level": "senior"}
        save_research(JOB_ID, "parsed_skills", data, db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert result["parsed_skills"] == data

    def test_save_research_upsert(self, tmp_path):
        """Re-saving same key overwrites old value."""
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_research(JOB_ID, "key1", "old_value", db_path=db)
        save_research(JOB_ID, "key1", "new_value", db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert result["key1"] == "new_value"

    def test_load_research_empty_returns_empty_dict(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert result == {}

    def test_multiple_research_keys(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_research(JOB_ID, "company", "Acme Corp", db_path=db)
        save_research(JOB_ID, "role_level", "Senior", db_path=db)
        save_research(JOB_ID, "salary", "$150k", db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert len(result) == 3
        assert result["company"] == "Acme Corp"
        assert result["role_level"] == "Senior"
        assert result["salary"] == "$150k"


# ---------------------------------------------------------------------------
# Candidate analysis tests
# ---------------------------------------------------------------------------

class TestCandidateAnalysis:
    def test_save_and_load_candidate_analysis(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        resume_data = {"skills": ["Python", "FastAPI"], "education": "BS CS"}
        job_match = {"matching_skills": ["Python"], "missing_skills": ["Kubernetes"]}
        tool_results = [{"title": "Company Info", "url": "https://example.com"}]

        save_candidate_analysis(
            JOB_ID,
            resume_data=resume_data,
            job_match=job_match,
            tool_results=tool_results,
            db_path=db,
        )

        result = load_candidate_analysis(JOB_ID, db_path=db)
        assert result is not None
        assert result["resume_data"] == resume_data
        assert result["job_match"] == job_match
        assert result["tool_results"] == tool_results

    def test_candidate_analysis_upsert(self, tmp_path):
        """Second save overwrites first."""
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_candidate_analysis(JOB_ID, resume_data={"skills": ["Python"]}, db_path=db)
        save_candidate_analysis(JOB_ID, resume_data={"skills": ["Python", "Go"]}, db_path=db)
        result = load_candidate_analysis(JOB_ID, db_path=db)
        assert "Go" in result["resume_data"]["skills"]

    def test_load_candidate_analysis_none_if_missing(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        result = load_candidate_analysis(JOB_ID, db_path=db)
        assert result is None


# ---------------------------------------------------------------------------
# Final response tests
# ---------------------------------------------------------------------------

class TestFinalResponse:
    def test_save_and_load_final_response(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        response = {
            "summary": "Great Python role at a startup.",
            "requirements": ["Python", "LangGraph"],
            "initial_assessment": "Good fit.",
            "sources": ["https://example.com"],
        }
        save_final_response(JOB_ID, response, db_path=db)
        loaded = load_final_response(JOB_ID, db_path=db)
        assert loaded == response

    def test_final_response_upsert(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_final_response(JOB_ID, {"summary": "v1"}, db_path=db)
        save_final_response(JOB_ID, {"summary": "v2"}, db_path=db)
        loaded = load_final_response(JOB_ID, db_path=db)
        assert loaded["summary"] == "v2"

    def test_load_final_response_none_if_missing(self, tmp_path):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        assert load_final_response(JOB_ID, db_path=db) is None


# ---------------------------------------------------------------------------
# End-to-end workflow scenario
# ---------------------------------------------------------------------------

class TestWorkflowLifecycle:
    """
    Demonstrates the full Phase 3 stop→resume scenario:
      create → research saved → interrupted → load → research intact → resume → complete
    """

    def test_stop_and_resume_workflow(self, tmp_path):
        db = _db(tmp_path)

        # Step 1: create a new job
        job_id = "lifecycle-job-001"
        create_job(job_id, JOB_DESC, resume_text="5 years Python.", db_path=db)
        job = get_job(job_id, db_path=db)
        assert job["status"] == "new"

        # Step 2: workflow starts, saves some research
        update_job_status(job_id, "running", db_path=db)
        save_research(job_id, "company_culture", "Remote-first", db_path=db)
        save_research(job_id, "tech_stack", {"languages": ["Python", "Go"]}, db_path=db)
        save_candidate_analysis(
            job_id,
            resume_data={"skills": ["Python", "FastAPI"]},
            job_match={"matching_skills": ["Python"]},
            tool_results=[{"title": "Info", "url": "https://acme.com"}],
            db_path=db,
        )

        # Step 3: simulate interruption (process killed)
        update_job_status(job_id, "interrupted", db_path=db)
        assert get_job(job_id, db_path=db)["status"] == "interrupted"

        # Step 4: "new process" loads state from DB (simulated by fresh loads)
        loaded_research = load_research(job_id, db_path=db)
        loaded_analysis = load_candidate_analysis(job_id, db_path=db)

        # Step 5: verify previously saved research is intact
        assert loaded_research["company_culture"] == "Remote-first"
        assert loaded_research["tech_stack"] == {"languages": ["Python", "Go"]}
        assert loaded_analysis["resume_data"]["skills"] == ["Python", "FastAPI"]
        assert loaded_analysis["job_match"]["matching_skills"] == ["Python"]
        assert loaded_analysis["tool_results"][0]["url"] == "https://acme.com"

        # Step 6: resume – additional research saved, status updated
        update_job_status(job_id, "running", db_path=db)
        save_research(job_id, "salary_range", "$150k-$180k", db_path=db)
        all_research = load_research(job_id, db_path=db)
        assert len(all_research) == 3  # original 2 + 1 new

        # Step 7: final response saved and job marked complete
        final_response = {
            "summary": "Excellent match.",
            "requirements": ["Python", "FastAPI"],
            "initial_assessment": "Strong candidate.",
            "sources": ["https://acme.com"],
        }
        save_final_response(job_id, final_response, db_path=db)
        update_job_status(job_id, "complete", db_path=db)

        # Step 8: verify final state
        assert load_final_response(job_id, db_path=db) == final_response
        assert get_job(job_id, db_path=db)["status"] == "complete"

        # Step 9: job appears in listing
        jobs = list_jobs(db_path=db)
        job_ids = [j["job_id"] for j in jobs]
        assert job_id in job_ids

    @pytest.mark.parametrize("status", ["new", "running", "interrupted", "complete", "error"])
    def test_all_valid_statuses(self, tmp_path, status):
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        update_job_status(JOB_ID, status, db_path=db)
        assert get_job(JOB_ID, db_path=db)["status"] == status

    def test_duplicate_research_key_upsert_not_duplicate(self, tmp_path):
        """Saving the same key twice must not create two rows."""
        db = _db(tmp_path)
        create_job(JOB_ID, JOB_DESC, db_path=db)
        save_research(JOB_ID, "tech_stack", "Python", db_path=db)
        save_research(JOB_ID, "tech_stack", "Python, Go", db_path=db)
        result = load_research(JOB_ID, db_path=db)
        assert len(result) == 1
        assert result["tech_stack"] == "Python, Go"

    def test_parameterised_sql_injection_safe(self, tmp_path):
        """Confirm parameterised queries protect against SQL injection."""
        db = _db(tmp_path)
        malicious_id = "'; DROP TABLE jobs; --"
        # Should not raise and should not corrupt the DB
        result = get_job(malicious_id, db_path=db)
        assert result is None
        # jobs table must still exist
        conn = sqlite3.connect(db)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "jobs" in tables
