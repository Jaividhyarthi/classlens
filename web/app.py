from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import dashboard_data, store

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="ClassLens")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")

DEFAULT_DEPARTMENT = "Mechanical Engineering"


def current_faculty_id(request: Request) -> str:
    fid = request.cookies.get("faculty_id")
    roster = store.load_faculty()
    if fid and any(f["id"] == fid for f in roster):
        return fid
    return roster[0]["id"] if roster else ""


def current_role(request: Request) -> str:
    return request.cookies.get("role", "staff")


@app.get("/")
def root(request: Request):
    if not request.cookies.get("role"):
        return RedirectResponse("/login")
    role = current_role(request)
    return RedirectResponse("/hod/dashboard" if role == "hod" else "/staff/dashboard")


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@app.post("/login")
def login_submit(request: Request, role: str = Form("staff"), username: str = Form("")):
    roster = store.load_faculty()
    matched = next((f for f in roster if f["email"].lower() == username.strip().lower()), None)
    faculty_id = matched["id"] if matched else (roster[0]["id"] if roster else "")
    dest = "/hod/dashboard" if role == "hod" else "/staff/dashboard"
    resp = RedirectResponse(dest, status_code=303)
    resp.set_cookie("role", role, httponly=True, samesite="lax")
    resp.set_cookie("faculty_id", faculty_id, httponly=True, samesite="lax")
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("role")
    resp.delete_cookie("faculty_id")
    return resp


@app.get("/upload")
def upload_page(request: Request):
    faculty = store.load_faculty()
    syllabus = store.load_syllabus()
    current_faculty = store.get_faculty(current_faculty_id(request))
    subjects = list(syllabus["subjects"].keys())
    first_subject = subjects[0] if subjects else None
    units = syllabus["subjects"].get(first_subject, {}).get("units", []) if first_subject else []
    unit = units[0] if units else None

    unit_progress_note = "No prior sessions analyzed for this unit yet."
    if unit and current_faculty:
        past = [
            r for r in dashboard_data.reports_for_faculty(current_faculty["id"])
            if r.get("subject") == first_subject and (r.get("syllabus") or {}).get("unit_id") == unit["id"]
        ]
        if past:
            avg_pct = sum(r["syllabus"]["pct_covered"] for r in past) / len(past)
            unit_progress_note = f"Module syllabus progress (avg of {len(past)} analyzed session(s)): {avg_pct:.0f}% delivered"

    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "role": current_role(request),
            "faculty_list": faculty,
            "current_faculty": current_faculty,
            "subjects": subjects,
            "unit": unit,
            "unit_progress_note": unit_progress_note,
        },
    )


@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    subject: str = Form(...),
    year: int = Form(3),
    unit: str = Form(""),
    hall: str = Form(""),
    faculty_id: str = Form(""),
):
    session_id = store.new_session_id()
    session_dir = store.UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest_path = session_dir / f"original{suffix}"
    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    if not faculty_id:
        faculty_id = current_faculty_id(request)
    faculty = store.get_faculty(faculty_id)

    store.create_session({
        "session_id": session_id,
        "faculty_id": faculty_id,
        "faculty_name": faculty["name"] if faculty else faculty_id,
        "subject": subject,
        "year": year,
        "unit": unit,
        "hall": hall or "Lecture Hall",
        "original_filename": file.filename,
        "upload_path": str(dest_path),
    })

    from pipeline.runner import run_pipeline
    background_tasks.add_task(run_pipeline, session_id)

    return {"session_id": session_id, "status_url": f"/api/analyze/{session_id}/status", "report_url": f"/report/{session_id}"}


@app.get("/api/analyze/{session_id}/status")
def api_analyze_status(session_id: str):
    session = store.get_session(session_id)
    if session is None:
        return {"error": "not_found"}
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "stage": session.get("stage"),
        "error": session.get("error"),
    }


@app.get("/report/{session_id}")
def report_page(request: Request, session_id: str):
    session = store.get_session(session_id)
    report = store.get_report(session_id)
    duration_label = "00:00"
    if report and report.get("duration_sec"):
        total = int(report["duration_sec"])
        duration_label = f"{total // 60:02d}:{total % 60:02d}"
    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "session": session,
            "report": report,
            "duration_label": duration_label,
        },
    )


@app.get("/staff/dashboard")
def staff_dashboard(request: Request):
    faculty_id = current_faculty_id(request)
    ctx = dashboard_data.staff_dashboard_context(faculty_id)
    return templates.TemplateResponse(request, "staff_dashboard.html", ctx)


@app.get("/hod/dashboard")
def hod_dashboard(request: Request):
    ctx = dashboard_data.hod_dashboard_context(DEFAULT_DEPARTMENT)
    return templates.TemplateResponse(request, "hod_dashboard.html", ctx)


@app.get("/faculty-directory")
def faculty_directory(request: Request):
    ctx = dashboard_data.faculty_directory_context(DEFAULT_DEPARTMENT)
    return templates.TemplateResponse(request, "faculty_directory.html", ctx)


@app.get("/analytics")
def analytics(request: Request, subject: str = ""):
    ctx = dashboard_data.analytics_context(subject or None)
    return templates.TemplateResponse(request, "analytics.html", ctx)


@app.get("/settings")
def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {"role": current_role(request)})
