#!/usr/bin/env python3
"""
PEP BALANCE - API Test Script

Sends sample requests to the Verfügbarkeit API and validates responses.

Usage:
    1. Start the server:  uvicorn api_server:app --port 8000
    2. Run this script:   python test_api.py
"""

import json
import sys

import requests

BASE_URL = "http://localhost:8000"

# ─── Sample Data ─────────────────────────────────────────────────────────────
# Simulates what PEP Balance would send.
# MA-01: available Mon-Fri but only scheduled Mon/Wed/Fri (never Tue/Thu)
# MA-02: available Mon-Fri, scheduled normally
# MA-03: available Mon-Wed, alternating week pattern on Wed

SAMPLE_PAYLOAD = {
    "employees": [
        {
            "id": 1,
            "name": "MA-01",
            "availability": {
                "Mon": [6, 14],
                "Tue": [6, 14],
                "Wed": [6, 14],
                "Thu": [6, 14],
                "Fri": [6, 14],
            },
        },
        {
            "id": 2,
            "name": "MA-02",
            "availability": {
                "Mon": [8, 16],
                "Tue": [8, 16],
                "Wed": [8, 16],
                "Thu": [8, 16],
                "Fri": [8, 16],
            },
        },
        {
            "id": 3,
            "name": "MA-03",
            "availability": {
                "Mon": [7, 15],
                "Tue": [7, 15],
                "Wed": [7, 15],
            },
        },
    ],
    "schedule": [],
    "absences": [],
}


def _generate_schedule():
    """Generate 6 months of schedule data for testing."""
    import pandas as pd

    records = []
    dates = pd.date_range("2024-10-01", "2025-03-31", freq="D")

    for d in dates:
        day_abbr = d.strftime("%a")[:3]

        # MA-01: scheduled Mon/Wed/Fri, never Tue/Thu
        if day_abbr in ("Mon", "Tue", "Wed", "Thu", "Fri"):
            scheduled = day_abbr in ("Mon", "Wed", "Fri")
            records.append({
                "employee_id": 1,
                "date": d.strftime("%Y-%m-%d"),
                "day": day_abbr,
                "scheduled": scheduled,
            })

        # MA-02: scheduled every weekday (normal)
        if day_abbr in ("Mon", "Tue", "Wed", "Thu", "Fri"):
            records.append({
                "employee_id": 2,
                "date": d.strftime("%Y-%m-%d"),
                "day": day_abbr,
                "scheduled": True,
            })

        # MA-03: Mon/Tue always scheduled, Wed only even weeks
        if day_abbr in ("Mon", "Tue", "Wed"):
            week_num = d.isocalendar()[1]
            scheduled = True if day_abbr != "Wed" else (week_num % 2 == 0)
            records.append({
                "employee_id": 3,
                "date": d.strftime("%Y-%m-%d"),
                "day": day_abbr,
                "scheduled": scheduled,
            })

    return records


def _add_absences():
    """Add a few absence records."""
    return [
        {"employee_id": 1, "date": "2025-01-06", "type": "vacation"},
        {"employee_id": 1, "date": "2025-01-07", "type": "vacation"},
        {"employee_id": 3, "date": "2025-02-10", "type": "sick"},
    ]


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_health():
    print("TEST: GET /api/health")
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json()["status"] == "ok"
    print("  PASSED\n")


def test_bulk_analysis():
    print("TEST: POST /api/verfuegbarkeit (all employees)")
    payload = {**SAMPLE_PAYLOAD}
    payload["schedule"] = _generate_schedule()
    payload["absences"] = _add_absences()

    r = requests.post(f"{BASE_URL}/api/verfuegbarkeit", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Analyzed: {data['analyzed_employees']} employees")
    print(f"  Findings: {data['total_findings']}")

    for f in data["findings"]:
        print(f"    -> {f['employee_name']} | {f['day']} | "
              f"confidence: {f['confidence']}% | action: {f['action']}")
        print(f"       {f['recommendation']}")

    if data["alternating_patterns"]:
        print(f"  Alternating patterns:")
        for p in data["alternating_patterns"]:
            print(f"    -> {p['employee_name']} | {p['day']} | {p['pattern']}")

    # MA-01 should be flagged for Tue and Thu (never scheduled)
    flagged_days = {f["day"] for f in data["findings"] if f["employee_id"] == 1}
    assert "Tue" in flagged_days, "MA-01 Tue should be flagged"
    assert "Thu" in flagged_days, "MA-01 Thu should be flagged"

    # MA-02 should NOT be flagged (always scheduled)
    ma02_findings = [f for f in data["findings"] if f["employee_id"] == 2]
    assert len(ma02_findings) == 0, "MA-02 should have no findings"

    print("  PASSED\n")
    return payload


def test_single_employee(payload):
    print("TEST: POST /api/verfuegbarkeit/1 (single employee)")
    r = requests.post(f"{BASE_URL}/api/verfuegbarkeit/1", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    data = r.json()
    assert data["analyzed_employees"] == 1
    for f in data["findings"]:
        assert f["employee_id"] == 1, "Should only contain MA-01 findings"

    print(f"  Findings for MA-01: {data['total_findings']}")
    print("  PASSED\n")


def test_employee_not_found(payload):
    print("TEST: POST /api/verfuegbarkeit/999 (not found)")
    r = requests.post(f"{BASE_URL}/api/verfuegbarkeit/999", json=payload)
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("  PASSED\n")


def test_empty_schedule():
    print("TEST: POST /api/verfuegbarkeit (empty schedule)")
    payload = {**SAMPLE_PAYLOAD, "schedule": [], "absences": []}
    r = requests.post(f"{BASE_URL}/api/verfuegbarkeit", json=payload)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("  PASSED\n")


def test_custom_config():
    print("TEST: POST /api/verfuegbarkeit (custom config)")
    payload = {**SAMPLE_PAYLOAD}
    payload["schedule"] = _generate_schedule()
    payload["absences"] = _add_absences()
    payload["config"] = {
        "min_deviations": 6,
        "confidence_threshold": 90,
        "months": 6,
    }

    r = requests.post(f"{BASE_URL}/api/verfuegbarkeit", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    data = r.json()
    # With 90% threshold, only very high confidence findings should appear
    for f in data["findings"]:
        assert f["confidence"] >= 90, f"Confidence {f['confidence']} below threshold 90"

    print(f"  Findings with 90%+ confidence: {data['total_findings']}")
    print("  PASSED\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("PEP BALANCE - Verfügbarkeit API Tests")
    print("=" * 50)
    print(f"Server: {BASE_URL}\n")

    try:
        requests.get(f"{BASE_URL}/api/health", timeout=2)
    except requests.ConnectionError:
        print("ERROR: Server not running.")
        print("Start it with: uvicorn api_server:app --port 8000")
        sys.exit(1)

    test_health()
    payload = test_bulk_analysis()
    test_single_employee(payload)
    test_employee_not_found(payload)
    test_empty_schedule()
    test_custom_config()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
