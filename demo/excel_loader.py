"""
PEP BALANCE AI Demo - Excel Data Loader

Parses the real PEP BALANCE Excel file into DataFrames matching
the format expected by availability_checker.py and task_scorer.py.

Excel structure:
  Sheet "MA": Daily schedule per employee (planned vs actual shifts)
  Sheet "Aufg nach Monat": Monthly task hours per employee
  Sheet "Aufg-01-05": Summary of task hours across full period
"""

from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd


EXCEL_PATH = Path(__file__).parent / "KI-Analysedaten MA 25-06-02 (1).xlsx"

# German → English day mapping
DAY_MAP_DE_EN = {
    "Mo": "Mon", "Di": "Tue", "Mi": "Wed",
    "Do": "Thu", "Fr": "Fri", "Sa": "Sat", "So": "Sun",
}

# Task column pairs in the "Aufg nach Monat" / "Aufg-01-05" sheets
# Each task has (hours_col, days_col)
TASK_COLUMNS = {
    "Besprechung":        ("Besprechung", "Unnamed: 6"),
    "Etiketten stecken":  ("Etiketten stecken", "Unnamed: 8"),
    "Kassenaufsicht":     ("Kassenaufsicht", "Unnamed: 10"),
    "Marktleitung":       ("Marktleitung", "Unnamed: 12"),
    "Qualitätssicherung": ("Qualitätssicherung", "Unnamed: 14"),
    "Sonderaufgabe":      ("Sonderaufgabe", "Unnamed: 16"),
    "Warenannahme":       ("Warenannahme", "Unnamed: 18"),
}

# Absence types in the Beschreibung column
ABSENCE_TYPES = {
    "Urlaub", "Krankheit mit Lohnfortzahlung", "Kind krank mit Lohnfortzahlung",
    "Frei", "Frei Überstundenabbau",
}
HOLIDAY_TYPES = {
    "Neujahrstag", "Karfreitag", "Ostermontag",
    "Tag der Arbeit", "Christi Himmelfahrt",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _time_to_hour(t):
    """Convert a datetime.time to float hour, or return None."""
    if isinstance(t, dt_time):
        return t.hour + t.minute / 60
    return None


def _time_to_int_hour(t):
    """Convert a datetime.time to integer hour, or return None."""
    if isinstance(t, dt_time):
        return t.hour
    return None


def _hours_between(start, end):
    """Hours between two datetime.time objects."""
    s = _time_to_hour(start)
    e = _time_to_hour(end)
    if s is not None and e is not None and e > s:
        return round(e - s, 2)
    return None


# ─── Load Raw Sheets ─────────────────────────────────────────────────────────

def _load_ma_sheet(path=None):
    """Load and clean the MA (daily schedule) sheet."""
    path = path or EXCEL_PATH
    df = pd.read_excel(path, sheet_name="MA")

    # The real header is in row index 2; data starts at row 3
    col_names = [
        "_drop", "employee_name", "day_de", "date",
        "plan_start", "plan_end", "plan_break", "plan_hours",
        "actual_start", "actual_end", "actual_break", "actual_hours",
        "description", "absence_hours", "time_account", "total_hours",
        "settlement_account",
    ]
    df.columns = col_names[:len(df.columns)]
    # Drop header rows and empty rows
    df = df.iloc[3:].copy()  # skip the 3 header/title rows
    df = df[df["employee_name"].notna()].copy()
    df = df[df["employee_name"].str.startswith("MA-", na=False)].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["date_only"] = df["date"].dt.date

    # Map German days to English
    df["day"] = df["day_de"].map(DAY_MAP_DE_EN)

    # Extract employee numeric ID
    df["employee_id"] = df["employee_name"].str.extract(r"MA-(\d+)").astype(int)

    return df


def _load_task_sheet(path=None):
    """Load and clean the 'Aufg nach Monat' (monthly task hours) sheet."""
    path = path or EXCEL_PATH
    df = pd.read_excel(path, sheet_name="Aufg nach Monat")

    # Row 0 is sub-headers (Std/Tage), data starts at row 1
    df = df.iloc[1:].copy()
    df = df[df["Name - Mitarbeiter"].notna()].copy()
    df = df[df["Name - Mitarbeiter"].str.startswith("MA-", na=False)].copy()

    df["employee_id"] = df["Name - Mitarbeiter"].str.extract(r"MA-(\d+)").astype(int)

    # Parse period dates
    df["month_start"] = pd.to_datetime(df["Von"])
    df["month_end"] = pd.to_datetime(df["Bis"])

    # Convert task columns to float
    for task_name, (hrs_col, days_col) in TASK_COLUMNS.items():
        df[hrs_col] = pd.to_numeric(df[hrs_col], errors="coerce").fillna(0)
        df[days_col] = pd.to_numeric(df[days_col], errors="coerce").fillna(0)

    return df


def _load_summary_sheet(path=None):
    """Load the 'Aufg-01-05' (full-period summary) sheet."""
    path = path or EXCEL_PATH
    df = pd.read_excel(path, sheet_name="Aufg-01-05")

    df = df.iloc[1:].copy()
    df = df[df["Name"].notna()].copy()
    df = df[df["Name"].str.startswith("MA-", na=False)].copy()

    df["employee_id"] = df["Name"].str.extract(r"MA-(\d+)").astype(int)

    for task_name, (hrs_col, days_col) in TASK_COLUMNS.items():
        if hrs_col in df.columns:
            df[hrs_col] = pd.to_numeric(df[hrs_col], errors="coerce").fillna(0)
        if days_col in df.columns:
            df[days_col] = pd.to_numeric(df[days_col], errors="coerce").fillna(0)

    return df


# ─── Build Employee List ─────────────────────────────────────────────────────

def load_employees(path=None):
    """
    Build employee list with inferred skills and availability.

    Returns list of dicts matching the format used by the synthetic data:
        {"id": int, "name": str, "skills": [...], "assigned_tasks": [...],
         "availability": {"Mon": (start_h, end_h), ...}}
    """
    ma_df = _load_ma_sheet(path)
    summary_df = _load_summary_sheet(path)

    employees = []
    for eid in sorted(ma_df["employee_id"].unique()):
        name = f"MA-{eid:02d}"

        # Infer skills: any task with > 0 total hours in the summary
        skills = []
        assigned_tasks = []
        emp_summary = summary_df[summary_df["employee_id"] == eid]
        if len(emp_summary) > 0:
            row = emp_summary.iloc[0]
            for task_name, (hrs_col, _) in TASK_COLUMNS.items():
                if hrs_col in row.index and row[hrs_col] > 0:
                    skills.append(task_name)
                    # "Assigned" = tasks with significant hours (>10h total)
                    if row[hrs_col] > 10:
                        assigned_tasks.append(task_name)

        # Infer availability: for each weekday, find the most common
        # planned start/end times
        emp_ma = ma_df[ma_df["employee_id"] == eid]
        availability = {}
        for day_en in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            day_rows = emp_ma[
                (emp_ma["day"] == day_en) & (emp_ma["plan_start"].notna())
            ]
            if len(day_rows) < 2:
                continue
            # Most common start/end hour
            starts = day_rows["plan_start"].apply(_time_to_int_hour).dropna()
            ends = day_rows["plan_end"].apply(_time_to_int_hour).dropna()
            if len(starts) > 0 and len(ends) > 0:
                availability[day_en] = (
                    int(starts.mode().iloc[0]),
                    int(ends.mode().iloc[0]),
                )

        employees.append({
            "id": eid,
            "name": name,
            "skills": skills,
            "assigned_tasks": assigned_tasks,
            "availability": availability,
        })

    return employees


# ─── Build Schedule DataFrame ────────────────────────────────────────────────

def load_schedule(path=None):
    """
    Convert MA sheet into schedule DataFrame matching synthetic format.

    Columns: employee_id, date, day, available_start, available_end,
             scheduled, task_type, shift_start, shift_end
    """
    ma_df = _load_ma_sheet(path)
    task_df = _load_task_sheet(path)
    employees = load_employees(path)
    emp_avail = {e["id"]: e["availability"] for e in employees}

    records = []
    for _, row in ma_df.iterrows():
        eid = row["employee_id"]
        day = row["day"]
        date = row["date_only"]
        desc = row.get("description", None)

        avail = emp_avail.get(eid, {})
        avail_start = avail.get(day, (None, None))[0] if day in avail else None
        avail_end = avail.get(day, (None, None))[1] if day in avail else None

        has_plan = row["plan_start"] is not None and not pd.isna(row.get("plan_start"))
        has_actual = row["actual_start"] is not None and not pd.isna(row.get("actual_start"))

        # Determine if scheduled (actually worked)
        scheduled = has_actual

        # Determine task type from monthly distribution
        task_type = None
        if scheduled:
            task_type = _assign_task_for_day(eid, row["date"], task_df)

        shift_start = _time_to_int_hour(row.get("actual_start")) if has_actual else None
        shift_end = _time_to_int_hour(row.get("actual_end")) if has_actual else None

        # Use plan times for availability if no general availability
        if avail_start is None and has_plan:
            avail_start = _time_to_int_hour(row["plan_start"])
            avail_end = _time_to_int_hour(row["plan_end"])

        records.append({
            "employee_id": eid,
            "date": date,
            "day": day,
            "available_start": avail_start,
            "available_end": avail_end,
            "scheduled": scheduled,
            "task_type": task_type,
            "shift_start": shift_start,
            "shift_end": shift_end,
        })

    return pd.DataFrame(records)


def _assign_task_for_day(employee_id, date, task_df):
    """
    Assign a task type for a given day based on the monthly task distribution.

    Uses weighted random selection based on hours per task that month.
    """
    dt = pd.Timestamp(date)
    month_rows = task_df[
        (task_df["employee_id"] == employee_id)
        & (task_df["month_start"] <= dt)
        & (task_df["month_end"] >= dt)
    ]
    if len(month_rows) == 0:
        return None

    row = month_rows.iloc[0]
    weights = {}
    for task_name, (hrs_col, _) in TASK_COLUMNS.items():
        hrs = row.get(hrs_col, 0)
        if isinstance(hrs, (int, float)) and hrs > 0:
            weights[task_name] = hrs

    if not weights:
        return None

    tasks = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs = probs / probs.sum()

    rng = np.random.RandomState(hash((employee_id, str(date))) % (2**31))
    return rng.choice(tasks, p=probs)


# ─── Build Absences DataFrame ────────────────────────────────────────────────

def load_absences(path=None):
    """
    Extract absence records from the MA sheet.

    Returns DataFrame with: employee_id, date, type
    """
    ma_df = _load_ma_sheet(path)
    absences = []

    for _, row in ma_df.iterrows():
        desc = row.get("description")
        if not isinstance(desc, str):
            continue

        # Classify absence type
        if desc in ABSENCE_TYPES:
            if "Urlaub" in desc:
                abs_type = "vacation"
            elif "krank" in desc.lower() or "Krankheit" in desc:
                abs_type = "sick"
            else:
                abs_type = "day_off"
        elif desc in HOLIDAY_TYPES:
            abs_type = "holiday"
        else:
            continue  # Not an absence (e.g., "Home-Office", "Seminar")

        absences.append({
            "employee_id": row["employee_id"],
            "date": row["date_only"],
            "type": abs_type,
        })

    return pd.DataFrame(absences)


# ─── Build Completions DataFrame ─────────────────────────────────────────────

def load_completions(path=None):
    """
    Build task completion records with success based on hours deviation.

    Success = actual hours within 15% of planned hours.
    """
    ma_df = _load_ma_sheet(path)
    task_df = _load_task_sheet(path)

    records = []
    for _, row in ma_df.iterrows():
        # Need both planned and actual shifts
        if pd.isna(row.get("plan_start")) or pd.isna(row.get("actual_start")):
            continue
        if pd.isna(row.get("plan_hours")) or pd.isna(row.get("actual_hours")):
            continue

        planned_h = _time_to_hour(row["plan_hours"])
        actual_h = _time_to_hour(row["actual_hours"])
        if planned_h is None or actual_h is None or planned_h == 0:
            continue

        # Success = actual within 15% of planned
        deviation = abs(actual_h - planned_h) / planned_h
        success = deviation <= 0.15

        task_type = _assign_task_for_day(
            row["employee_id"], row["date"], task_df
        )
        if task_type is None:
            continue

        shift_start = _time_to_int_hour(row["actual_start"])

        records.append({
            "employee_id": row["employee_id"],
            "task_type": task_type,
            "date": row["date_only"],
            "shift_start": shift_start,
            "shift_end": _time_to_int_hour(row["actual_end"]),
            "success": success,
            "duration_minutes": int((actual_h or 0) * 60),
            "time_of_day": shift_start if shift_start else 8,
            "planned_hours": planned_h,
            "actual_hours": actual_h,
        })

    return pd.DataFrame(records)


# ─── Public API ───────────────────────────────────────────────────────────────

def load_real_data(path=None):
    """
    Load all data from the Excel file, returning the same dict format
    as data_generator.generate_all_data().
    """
    path = path or EXCEL_PATH
    employees = load_employees(path)
    absences = load_absences(path)
    schedule = load_schedule(path)
    completions = load_completions(path)

    return {
        "employees": pd.DataFrame(employees),
        "employees_list": employees,
        "absences": absences,
        "schedule": schedule,
        "completions": completions,
    }


if __name__ == "__main__":
    data = load_real_data()
    print(f"Employees:   {len(data['employees_list'])}")
    for e in data["employees_list"]:
        print(f"  {e['name']}: skills={e['skills']}, "
              f"assigned={e['assigned_tasks']}, "
              f"avail_days={list(e['availability'].keys())}")
    print(f"\nAbsences:    {len(data['absences'])} records")
    print(f"Schedule:    {len(data['schedule'])} records")
    print(f"Completions: {len(data['completions'])} records")

    if len(data["completions"]) > 0:
        sr = data["completions"]["success"].mean() * 100
        print(f"Overall success rate: {sr:.1f}%")
