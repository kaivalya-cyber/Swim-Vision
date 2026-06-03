"""Browser-based interface for running the SwimVision analysis pipeline."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, request, send_file, url_for
from werkzeug.utils import secure_filename

from src.run_pipeline import run_pipeline
from src.analytics.trend import analyze_trends
from src.storage.session_manager import (
    init_db,
    record_session,
    record_metrics,
    record_risk,
    record_symmetry,
    upsert_swimmer,
    get_swimmer_history,
    get_all_swimmers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = WEBSITE_ROOT / "frontend" / "dist"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"
PUBLIC_DIR = WEBSITE_ROOT / "public"
UPLOADS_DIR = PROJECT_ROOT / "web_uploads"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mpg", ".mpeg", ".webm"}
MAX_RECENT_JOBS = 8
IS_VERCEL = bool(os.environ.get("VERCEL"))


@dataclass
class JobRecord:
    """In-memory representation of a pipeline run."""

    id: str
    clip_id: str
    original_filename: str
    input_path: str
    crop: list[int] | None
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    current_step: str = "Waiting to start"
    step_index: int = 0
    total_steps: int = 6
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    analysis_mode: str = "dive"
    stroke_start_frame: int = 0
    swimmer_id: str = ""

    def touch(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")


app = Flask(__name__, static_folder=str(FRONTEND_DIST_DIR / "assets"), static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


def _is_allowed_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _parse_crop(form: Any) -> list[int] | None:
    raw_values = [form.get("crop_x", "").strip(), form.get("crop_y", "").strip(), form.get("crop_w", "").strip(), form.get("crop_h", "").strip()]
    if not any(raw_values):
        return None
    if not all(raw_values):
        raise ValueError("Fill all four crop fields or leave all of them blank.")
    crop = [int(value) for value in raw_values]
    if crop[2] <= 0 or crop[3] <= 0:
        raise ValueError("Crop width and height must be positive integers.")
    return crop


def _build_summary(clip_id: str, outputs: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase_boundaries": {},
        "overall_severity": None,
        "flagged_metrics": [],
    }

    deviations_key = "deviations_json"
    if deviations_key not in outputs:
        deviations_key = "stroke_deviations_json"
    
    deviations_path_str = outputs.get(deviations_key)
    if deviations_path_str:
        deviations_path = Path(deviations_path_str)
        if deviations_path.exists():
            with open(deviations_path, "r", encoding="utf-8") as handle:
                deviations = json.load(handle)
            summary["phase_boundaries"] = deviations.get("phase_boundaries", {})
            summary["overall_severity"] = deviations.get("overall_severity")
            flagged_rows: list[dict[str, Any]] = []
            for phase_name in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle"):
                for row in deviations.get(phase_name, []):
                    if isinstance(row, dict) and row.get("flag") in {"MINOR", "SIGNIFICANT", "CRITICAL"}:
                        flagged_rows.append(
                            {
                                "phase": phase_name.replace("_phase", "").replace("_cycle", ""),
                                "metric": row.get("metric"),
                                "measured": row.get("measured"),
                                "flag": row.get("flag"),
                            }
                        )
            flagged_rows.sort(key=lambda row: ("OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL").index(row["flag"]))
            summary["flagged_metrics"] = flagged_rows

    report_key = "report_json"
    report_path_str = outputs.get(report_key)
    if report_path_str:
        report_path = Path(report_path_str)
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            summary["reaction_time_ms"] = report.get("reaction_time_ms")
            summary["annotated_video_path"] = report.get("annotated_video_path")
            summary["num_cycles"] = report.get("num_cycles")
            summary["analysis_mode"] = report.get("analysis_mode")

    # Load optional advanced analysis results
    for key, json_path in [("velocity_acceleration", outputs.get("vel_accel_json")), ("dynamic_estimates", outputs.get("dynamic_json")), ("symmetry_analysis", outputs.get("symmetry_json")), ("injury_risk", outputs.get("risk_json"))]:
        if json_path:
            try:
                path = Path(json_path)
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        summary[key] = json.load(f)
            except Exception:
                pass

    summary["clip_id"] = clip_id
    return summary


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "running"
        job.current_step = "Preparing analysis"
        job.touch()

    def _progress_callback(event: dict[str, object]) -> None:
        with _jobs_lock:
            active_job = _jobs[job_id]
            active_job.total_steps = int(event.get("total_steps", active_job.total_steps))
            if event.get("event") == "step_started":
                active_job.step_index = int(event.get("step_index", active_job.step_index))
                active_job.current_step = str(event.get("label", active_job.current_step))
            elif event.get("event") == "step_completed":
                active_job.step_index = int(event.get("step_index", active_job.step_index))
                active_job.current_step = f"Finished {event.get('label', '')}".strip()
            active_job.touch()

    try:
        outputs = run_pipeline(
            input_path=job.input_path,
            clip_id=job.clip_id,
            crop=job.crop,
            progress_callback=_progress_callback,
            analysis_mode=job.analysis_mode,
            stroke_start_frame=job.stroke_start_frame,
            preset="standard",
        )
        outputs_payload = {name: str(path) for name, path in outputs.items()}
        summary = _build_summary(job.clip_id, outputs_payload)

        # Persist completed session to SQLite
        if job.swimmer_id:
            try:
                storage_conn = init_db()
                record_session(
                    storage_conn,
                    session_id=job_id,
                    clip_id=job.clip_id,
                    swimmer_id=job.swimmer_id,
                    analysis_mode=job.analysis_mode,
                    original_filename=job.original_filename,
                    overall_severity=summary.get("overall_severity"),
                    reaction_time_ms=summary.get("reaction_time_ms"),
                    num_cycles=summary.get("num_cycles", 0),
                    status="completed",
                )
                # Record metrics
                deviations_data = summary.get("deviations", summary)
                record_metrics(storage_conn, job_id, deviations_data)
                # Record symmetry if available
                symmetry = summary.get("symmetry_analysis")
                if symmetry:
                    record_symmetry(storage_conn, job_id, symmetry)
                # Record risk if available
                risk = summary.get("injury_risk")
                if risk:
                    record_risk(storage_conn, job_id, risk)
                storage_conn.close()
            except Exception:
                pass

        with _jobs_lock:
            active_job = _jobs[job_id]
            active_job.status = "completed"
            active_job.step_index = active_job.total_steps
            active_job.current_step = "Analysis complete"
            active_job.outputs = outputs_payload
            active_job.summary = summary
            active_job.touch()
    except Exception as exc:
        with _jobs_lock:
            active_job = _jobs[job_id]
            active_job.status = "failed"
            active_job.error = str(exc)
            active_job.current_step = "Pipeline failed"
            active_job.touch()


def _get_job_or_404(job_id: str) -> JobRecord:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            abort(404)
        return job


def _job_payload(job: JobRecord) -> dict[str, Any]:
    payload = asdict(job)
    payload["artifact_urls"] = {
        name: url_for("download_artifact_api", job_id=job.id, artifact_name=name)
        for name in job.outputs
    }
    return payload


def _frontend_index_path() -> Path | None:
    if FRONTEND_INDEX_PATH.exists():
        return FRONTEND_INDEX_PATH
    public_index = PUBLIC_DIR / "index.html"
    if public_index.exists():
        return public_index
    return None


@app.get("/")
def index() -> Any:
    return _serve_frontend()


@app.post("/api/jobs")
def create_job() -> Any:
    if IS_VERCEL:
        return (
            jsonify(
                {
                    "error": "Video pipeline jobs are disabled in this Vercel deployment.",
                    "hint": "Use this UI for report browsing and run heavy SwimVision processing on a dedicated backend/local machine.",
                }
            ),
            501,
        )

    uploaded_file = request.files.get("video")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Choose a race or training video to upload."}), 400
    if not _is_allowed_upload(uploaded_file.filename):
        return jsonify({"error": "Upload an MP4, MOV, AVI, MPEG, or WEBM file."}), 400

    clip_id = request.form.get("clip_id", "").strip() or f"swim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    clip_id = secure_filename(clip_id) or f"swim_{uuid.uuid4().hex[:8]}"

    swimmer_id = request.form.get("swimmer_id", "").strip() or ""

    analysis_mode = request.form.get("analysis_mode", "dive").strip()
    if analysis_mode not in ("dive", "stroke"):
        analysis_mode = "dive"

    stroke_start_frame = 0
    if analysis_mode == "stroke":
        try:
            stroke_start_frame = int(request.form.get("stroke_start_frame", "0"))
        except (ValueError, TypeError):
            stroke_start_frame = 0

    try:
        crop = _parse_crop(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = uuid.uuid4().hex
    upload_dir = UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_filename = secure_filename(uploaded_file.filename) or f"{clip_id}.mov"
    input_path = upload_dir / input_filename
    uploaded_file.save(input_path)

    job = JobRecord(
        id=job_id,
        clip_id=clip_id,
        original_filename=uploaded_file.filename,
        input_path=str(input_path),
        crop=crop,
        analysis_mode=analysis_mode,
        stroke_start_frame=stroke_start_frame,
        swimmer_id=swimmer_id,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    # Persist to SQLite if swimmer_id is provided
    if swimmer_id:
        try:
            storage_conn = init_db()
            upsert_swimmer(storage_conn, swimmer_id, name=swimmer_id)
            record_session(
                storage_conn,
                session_id=job_id,
                clip_id=clip_id,
                swimmer_id=swimmer_id,
                analysis_mode=analysis_mode,
                original_filename=uploaded_file.filename,
                crop=crop,
                status="queued",
            )
            storage_conn.close()
        except Exception:
            pass

    worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "job_url": url_for("job_detail", job_id=job_id)}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str) -> Any:
    job = _get_job_or_404(job_id)
    return jsonify(_job_payload(job))


@app.get("/jobs/<job_id>")
def job_detail(job_id: str) -> Any:
    _get_job_or_404(job_id)
    return _serve_frontend()


@app.get("/trends")
def trends_page() -> Any:
    """Serve the trends dashboard SPA page."""
    return _serve_frontend()


def _download_artifact(job_id: str, artifact_name: str) -> Any:
    job = _get_job_or_404(job_id)
    artifact_path = job.outputs.get(artifact_name)
    if artifact_path is None:
        abort(404)
    path = Path(artifact_path)
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=False)


@app.get("/api/jobs/<job_id>/download/<artifact_name>")
def download_artifact_api(job_id: str, artifact_name: str) -> Any:
    return _download_artifact(job_id, artifact_name)


@app.get("/jobs/<job_id>/download/<artifact_name>")
def download_artifact(job_id: str, artifact_name: str) -> Any:
    return _download_artifact(job_id, artifact_name)


@app.post("/jobs/<job_id>/rerun")
def rerun_job(job_id: str) -> Any:
    job = _get_job_or_404(job_id)
    if job.status == "running":
        return jsonify({"error": "This analysis is already running."}), 409

    with _jobs_lock:
        active_job = _jobs[job_id]
        active_job.status = "queued"
        active_job.error = None
        active_job.outputs = {}
        active_job.summary = {}
        active_job.step_index = 0
        active_job.current_step = "Waiting to restart"
        active_job.touch()

    worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    worker.start()
    return redirect(url_for("job_detail", job_id=job_id))


@app.get("/api/trends")
def trends_analysis() -> Any:
    """Run longitudinal trend analysis across completed jobs.

    Accepts ?job_ids=id1,id2,... or scans all completed jobs for report JSONs.
    Accepts ?swimmer_id=... to filter by a specific swimmer.
    Accepts ?analysis_mode=dive|stroke to filter by mode.
    Accepts ?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD for date range.
    Accepts ?aggregation=week|month to group sessions by time period.
    Accepts ?primary_metric=stroke_rate to set the primary metric.
    """

    requested_ids = request.args.get("job_ids", "")
    swimmer_filter = request.args.get("swimmer_id", "").strip()
    analysis_mode_filter = request.args.get("analysis_mode", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    with _jobs_lock:
        if requested_ids:
            report_paths = []
            for jid in requested_ids.split(","):
                jid = jid.strip()
                job = _jobs.get(jid)
                if job and job.status == "completed" and "report_json" in job.outputs:
                    if not swimmer_filter or job.swimmer_id == swimmer_filter:
                        if not analysis_mode_filter or job.analysis_mode == analysis_mode_filter:
                            report_paths.append(job.outputs["report_json"])
        else:
            report_paths = [
                job.outputs["report_json"]
                for job in _jobs.values()
                if job.status == "completed" and "report_json" in job.outputs
                and (not swimmer_filter or job.swimmer_id == swimmer_filter)
                and (not analysis_mode_filter or job.analysis_mode == analysis_mode_filter)
            ]

    if not report_paths:
        return jsonify({"error": "No completed reports found for trend analysis."}), 404

    primary_metric = request.args.get("primary_metric", "stroke_rate")
    aggregation = request.args.get("aggregation", "").strip()
    try:
        result = analyze_trends(
            report_paths,
            primary_metric=primary_metric,
            analysis_mode=analysis_mode_filter,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
        )
    except Exception as exc:
        return jsonify({"error": f"Trend analysis failed: {exc}"}), 500

    # Include available swimmer IDs for the frontend filter dropdown
    with _jobs_lock:
        swimmer_ids = sorted(
            {job.swimmer_id for job in _jobs.values() if job.swimmer_id and job.status == "completed"}
        )
        available_modes = sorted(
            {job.analysis_mode for job in _jobs.values() if job.status == "completed"}
        )
    result["available_swimmer_ids"] = list(swimmer_ids)
    result["available_analysis_modes"] = list(available_modes)
    result["active_swimmer_id"] = swimmer_filter or ""
    result["active_analysis_mode"] = analysis_mode_filter or ""

    return jsonify(result)


@app.get("/api/trends/compare")
def trends_compare() -> Any:
    """Compare trends between two swimmers side-by-side.

    Accepts ?swimmer_a=id1&swimmer_b=id2&primary_metric=stroke_rate
    &analysis_mode=dive|stroke&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD.
    """

    swimmer_a = request.args.get("swimmer_a", "").strip()
    swimmer_b = request.args.get("swimmer_b", "").strip()
    if not swimmer_a or not swimmer_b:
        return jsonify({"error": "Provide swimmer_a and swimmer_b query parameters."}), 400
    if swimmer_a == swimmer_b:
        return jsonify({"error": "Select two different swimmers to compare."}), 400

    primary_metric = request.args.get("primary_metric", "stroke_rate")
    analysis_mode_filter = request.args.get("analysis_mode", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()

    with _jobs_lock:
        paths_a = [
            job.outputs["report_json"]
            for job in _jobs.values()
            if job.status == "completed" and "report_json" in job.outputs
            and job.swimmer_id == swimmer_a
        ]
        paths_b = [
            job.outputs["report_json"]
            for job in _jobs.values()
            if job.status == "completed" and "report_json" in job.outputs
            and job.swimmer_id == swimmer_b
        ]

    result = {"swimmer_a": {}, "swimmer_b": {}, "primary_metric": primary_metric}

    if paths_a:
        try:
            result["swimmer_a"] = analyze_trends(
                paths_a,
                primary_metric=primary_metric,
                analysis_mode=analysis_mode_filter,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            result["swimmer_a"] = {"error": str(exc)}
    else:
        result["swimmer_a"] = {"error": f"No completed sessions for {swimmer_a}"}

    if paths_b:
        try:
            result["swimmer_b"] = analyze_trends(
                paths_b,
                primary_metric=primary_metric,
                analysis_mode=analysis_mode_filter,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            result["swimmer_b"] = {"error": str(exc)}
    else:
        result["swimmer_b"] = {"error": f"No completed sessions for {swimmer_b}"}

    # Build comparison summary
    trend_a = result["swimmer_a"].get("trend_summary", {}).get("primary_trend", {})
    trend_b = result["swimmer_b"].get("trend_summary", {}).get("primary_trend", {})

    if trend_a and trend_b:
        result["comparison"] = {
            "swimmer_a_name": swimmer_a,
            "swimmer_b_name": swimmer_b,
            "swimmer_a_mean": trend_a.get("mean"),
            "swimmer_b_mean": trend_b.get("mean"),
            "swimmer_a_direction": trend_a.get("direction"),
            "swimmer_b_direction": trend_b.get("direction"),
            "diff": (trend_a.get("mean", 0) or 0) - (trend_b.get("mean", 0) or 0),
        }

    return jsonify(result)


def _serve_frontend() -> Any:
    index_path = _frontend_index_path()
    if index_path is None:
        return (
            jsonify(
                {
                    "error": "Frontend build not found.",
                    "hint": "Run 'npm install' and 'npm run build' inside the frontend directory.",
                }
            ),
            503,
        )
    return send_file(index_path)


@app.get("/<path:path>")
def frontend_fallback(path: str) -> Any:
    if path.startswith("api/") or path.startswith("jobs/"):
        abort(404)
    return _serve_frontend()


if __name__ == "__main__":
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True)
