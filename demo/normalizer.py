"""
PEP BALANCE - Input Normalizer

Converts JSON input from PEP Balance into internal DataFrames
expected by availability_checker.py.

If PEP Balance changes their data format, only this file needs updating.
"""

import pandas as pd


def normalize_employees(raw_employees):
    """
    Convert JSON employee list into the internal format.

    Input (from PEP Balance):
        [{"id": 1, "name": "MA-01", "availability": {"Mon": [6, 14], ...}}]

    Output:
        [{"id": 1, "name": "MA-01", "availability": {"Mon": (6, 14), ...}}]
    """
    employees = []
    for emp in raw_employees:
        availability = {}
        for day, hours in emp.get("availability", {}).items():
            if isinstance(hours, list) and len(hours) == 2:
                availability[day] = (hours[0], hours[1])
            elif isinstance(hours, dict):
                availability[day] = (hours.get("start", 0), hours.get("end", 0))

        employees.append({
            "id": emp["id"],
            "name": emp.get("name", f"MA-{emp['id']:02d}"),
            "availability": availability,
        })
    return employees


def normalize_schedule(raw_schedule):
    """
    Convert JSON schedule entries into a pandas DataFrame.

    Input (from PEP Balance):
        [{"employee_id": 1, "date": "2025-03-10", "day": "Mon", "scheduled": true}]

    Output columns:
        employee_id, date, day, scheduled
    """
    if not raw_schedule:
        return pd.DataFrame(columns=[
            "employee_id", "date", "day", "scheduled",
        ])

    df = pd.DataFrame(raw_schedule)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["scheduled"] = df["scheduled"].astype(bool)

    # Ensure required columns exist
    for col in ["employee_id", "date", "day", "scheduled"]:
        if col not in df.columns:
            raise ValueError(f"Missing required schedule column: {col}")

    return df


def normalize_absences(raw_absences):
    """
    Convert JSON absence entries into a pandas DataFrame.

    Input (from PEP Balance):
        [{"employee_id": 1, "date": "2025-03-12", "type": "vacation"}]

    Output columns:
        employee_id, date, type
    """
    if not raw_absences:
        return pd.DataFrame(columns=["employee_id", "date", "type"])

    df = pd.DataFrame(raw_absences)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def normalize_request(data):
    """
    Full normalization: takes the raw JSON request body and returns
    (employees_list, schedule_df, absences_df, config).
    """
    employees = normalize_employees(data.get("employees", []))
    schedule = normalize_schedule(data.get("schedule", []))
    absences = normalize_absences(data.get("absences", []))

    config = data.get("config") or {}
    config.setdefault("min_deviations", 6)
    config.setdefault("confidence_threshold", 50)
    config.setdefault("months", 6)

    return employees, schedule, absences, config
