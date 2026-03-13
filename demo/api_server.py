"""
PEP BALANCE - Verfügbarkeit API Server

REST API that receives employee/schedule data from PEP Balance
and returns availability pattern analysis.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from normalizer import normalize_request
from availability_checker import (
    detect_availability_patterns,
    detect_all_patterns,
    detect_alternating_patterns,
)

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class EmployeeInput(BaseModel):
    id: int
    name: str = ""
    availability: dict[str, list[int]] = Field(
        default_factory=dict,
        description='Day → [start_hour, end_hour], e.g. {"Mon": [6, 14]}',
    )


class ScheduleEntry(BaseModel):
    employee_id: int
    date: str = Field(description="ISO date, e.g. 2025-03-10")
    day: str = Field(description="Weekday abbreviation: Mon, Tue, Wed, Thu, Fri, Sat, Sun")
    scheduled: bool


class AbsenceEntry(BaseModel):
    employee_id: int
    date: str = Field(description="ISO date, e.g. 2025-03-12")
    type: str = Field(
        default="other",
        description="Absence type: vacation, sick, day_off, holiday",
    )


class ConfigInput(BaseModel):
    min_deviations: int = Field(default=6, description="Minimum deviations to flag (default 6)")
    confidence_threshold: float = Field(default=50.0, description="Minimum confidence % to include (default 50)")
    months: int = Field(default=6, description="Lookback period in months (default 6)")


class VerfuegbarkeitRequest(BaseModel):
    employees: list[EmployeeInput]
    schedule: list[ScheduleEntry]
    absences: list[AbsenceEntry] = Field(default_factory=list)
    config: Optional[ConfigInput] = None


# ─── Response Models ─────────────────────────────────────────────────────────

class Finding(BaseModel):
    employee_id: int
    employee_name: str
    day: str
    slot: str
    total_available: int
    times_scheduled: int
    deviations: int
    confidence: float
    recommendation: str
    action: str


class AlternatingPattern(BaseModel):
    employee_id: int
    employee_name: str
    day: str
    pattern: str
    even_week_rate: float
    odd_week_rate: float


class VerfuegbarkeitResponse(BaseModel):
    status: str = "ok"
    analyzed_employees: int
    total_findings: int
    findings: list[Finding]
    alternating_patterns: list[AlternatingPattern]


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PEP BALANCE - Verfügbarkeit API",
    description="Analyzes employee availability patterns and returns recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_analysis(employees, schedule_df, absences_df, config):
    """Run availability analysis and return structured results."""
    confidence_threshold = config.get("confidence_threshold", 50)

    # Detect unused availability patterns
    raw_findings = detect_all_patterns(
        schedule_df, absences_df,
        months=config.get("months", 6),
        employees=employees,
    )

    # Filter by confidence threshold and add action field
    findings = []
    for f in raw_findings:
        if f["confidence"] < confidence_threshold:
            continue
        findings.append({
            **f,
            "action": "REMOVE" if f["confidence"] >= 80 else "REVIEW",
        })

    # Detect alternating-week patterns for all employees
    alternating = []
    for emp in employees:
        patterns = detect_alternating_patterns(
            emp["id"], schedule_df, absences_df,
            employees=employees,
        )
        alternating.extend(patterns)

    return findings, alternating


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/verfuegbarkeit", response_model=VerfuegbarkeitResponse)
def analyze_verfuegbarkeit(request: VerfuegbarkeitRequest):
    """Analyze availability patterns for all employees."""
    try:
        data = request.model_dump()
        employees, schedule_df, absences_df, config = normalize_request(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Data normalization error: {e}")

    if not employees:
        raise HTTPException(status_code=400, detail="No employees provided")
    if schedule_df.empty:
        raise HTTPException(status_code=400, detail="No schedule data provided")

    findings, alternating = _run_analysis(employees, schedule_df, absences_df, config)

    return VerfuegbarkeitResponse(
        analyzed_employees=len(employees),
        total_findings=len(findings),
        findings=findings,
        alternating_patterns=alternating,
    )


@app.post("/api/verfuegbarkeit/{employee_id}", response_model=VerfuegbarkeitResponse)
def analyze_verfuegbarkeit_single(employee_id: int, request: VerfuegbarkeitRequest):
    """Analyze availability patterns for a single employee."""
    try:
        data = request.model_dump()
        employees, schedule_df, absences_df, config = normalize_request(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Data normalization error: {e}")

    # Filter to requested employee
    emp = next((e for e in employees if e["id"] == employee_id), None)
    if emp is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    # Run analysis with all employees context but filter results
    findings, alternating = _run_analysis(employees, schedule_df, absences_df, config)

    findings = [f for f in findings if f["employee_id"] == employee_id]
    alternating = [a for a in alternating if a["employee_id"] == employee_id]

    return VerfuegbarkeitResponse(
        analyzed_employees=1,
        total_findings=len(findings),
        findings=findings,
        alternating_patterns=alternating,
    )
