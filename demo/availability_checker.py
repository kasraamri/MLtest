"""
PEP BALANCE AI Demo - Availability Checker

Rule-based deviation detection:
- Tracks "available but not scheduled" per employee per time slot
- After 6+ deviations in same slot over 6 months → flag for adjustment
- Excludes absence days (sick / vacation)
- Detects alternating patterns (early/late week rhythms)
- Confidence = (deviations / total_opportunities) * 100
"""

from datetime import datetime
import pandas as pd

from data_generator import EMPLOYEES, DAY_ABBREV


# ─── Core Detection ──────────────────────────────────────────────────────────

def detect_availability_patterns(employee_id, schedule_df, absences_df, months=6):
    """
    Find unused availability slots for an employee using the 6-deviation rule.

    Returns a list of dicts, each describing one flagged pattern:
        {
            "employee_id": int,
            "employee_name": str,
            "day": str,                # e.g. "Tue"
            "slot": str,               # e.g. "8:00-12:00"
            "total_available": int,     # days available (excl. absences)
            "times_scheduled": int,
            "deviations": int,
            "confidence": float,        # 0-100
            "recommendation": str,
        }
    """
    emp = next((e for e in EMPLOYEES if e["id"] == employee_id), None)
    if emp is None:
        return []

    absence_dates = set(
        absences_df[absences_df["employee_id"] == employee_id]["date"].tolist()
    )

    emp_sched = schedule_df[schedule_df["employee_id"] == employee_id].copy()

    findings = []
    for day_name, (avail_start, avail_end) in emp["availability"].items():
        day_rows = emp_sched[emp_sched["day"] == day_name]
        # Exclude absences
        day_rows = day_rows[~day_rows["date"].isin(absence_dates)]

        total = len(day_rows)
        if total == 0:
            continue

        scheduled_count = day_rows["scheduled"].sum()
        deviations = total - scheduled_count

        if deviations >= 6:
            confidence = (deviations / total) * 100 if total > 0 else 0
            slot_str = f"{avail_start}:00-{avail_end}:00"
            findings.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "day": day_name,
                "slot": slot_str,
                "total_available": total,
                "times_scheduled": int(scheduled_count),
                "deviations": int(deviations),
                "confidence": round(confidence, 1),
                "recommendation": (
                    f"Remove {day_name} {slot_str} from availability"
                    if confidence >= 80
                    else f"Review {day_name} {slot_str} scheduling"
                ),
            })

    # Sort by confidence descending
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings


def detect_all_patterns(schedule_df, absences_df, months=6):
    """Run detection across all employees, return consolidated results."""
    all_findings = []
    for emp in EMPLOYEES:
        findings = detect_availability_patterns(
            emp["id"], schedule_df, absences_df, months
        )
        all_findings.extend(findings)
    all_findings.sort(key=lambda f: f["confidence"], reverse=True)
    return all_findings


# ─── Alternating Pattern Detection ───────────────────────────────────────────

def detect_alternating_patterns(employee_id, schedule_df, absences_df):
    """
    Detect alternating-week rhythms, e.g. only scheduled on even weeks.

    Returns list of detected alternating patterns.
    """
    emp = next((e for e in EMPLOYEES if e["id"] == employee_id), None)
    if emp is None:
        return []

    absence_dates = set(
        absences_df[absences_df["employee_id"] == employee_id]["date"].tolist()
    )

    emp_sched = schedule_df[schedule_df["employee_id"] == employee_id].copy()
    patterns = []

    for day_name in emp["availability"]:
        day_rows = emp_sched[emp_sched["day"] == day_name].copy()
        day_rows = day_rows[~day_rows["date"].isin(absence_dates)]

        if len(day_rows) < 8:
            continue

        # Check even/odd week pattern
        day_rows = day_rows.copy()
        day_rows["week_num"] = pd.to_datetime(day_rows["date"]).dt.isocalendar().week
        day_rows["is_even_week"] = day_rows["week_num"] % 2 == 0

        even_weeks = day_rows[day_rows["is_even_week"]]
        odd_weeks = day_rows[~day_rows["is_even_week"]]

        even_rate = even_weeks["scheduled"].mean() if len(even_weeks) > 0 else 0
        odd_rate = odd_weeks["scheduled"].mean() if len(odd_weeks) > 0 else 0

        # Significant difference means alternating pattern
        if abs(even_rate - odd_rate) > 0.5:
            preferred = "even" if even_rate > odd_rate else "odd"
            patterns.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "day": day_name,
                "pattern": f"Alternating weeks - prefers {preferred} weeks",
                "even_week_rate": round(even_rate * 100, 1),
                "odd_week_rate": round(odd_rate * 100, 1),
            })

    return patterns


# ─── Priority Change Suggestions ─────────────────────────────────────────────

def suggest_priority_changes(employee_id, schedule_df, completions_df):
    """
    Check for upgrade / downgrade / removal based on usage patterns.

    Rules:
    - Scheduled for non-assigned task 6+ times → suggest add
    - Assigned task not used in 3 months → downgrade to "secondary"
    - Assigned task not used in 6 months → suggest removing
    - Frequently used (10+ in 3 months) → upgrade to "priority"
    """
    emp = next((e for e in EMPLOYEES if e["id"] == employee_id), None)
    if emp is None:
        return []

    suggestions = []
    today = schedule_df["date"].max()
    three_months_ago = today - pd.Timedelta(days=90)
    six_months_ago = today - pd.Timedelta(days=180)

    emp_sched = schedule_df[
        (schedule_df["employee_id"] == employee_id) & (schedule_df["scheduled"])
    ]
    emp_comps = completions_df[completions_df["employee_id"] == employee_id]

    # ── Non-assigned tasks used frequently → suggest ADD ──
    for task in emp.get("skills", []):
        if task in emp.get("assigned_tasks", []):
            continue  # already assigned
        task_count = len(emp_sched[emp_sched["task_type"] == task])
        task_comps = emp_comps[emp_comps["task_type"] == task]
        success_rate = task_comps["success"].mean() if len(task_comps) > 0 else 0

        if task_count >= 6:
            suggestions.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "task": task,
                "action": "ADD",
                "reason": (
                    f"Scheduled for '{task}' {task_count} times "
                    f"(success rate: {success_rate * 100:.0f}%)"
                ),
                "times_used": task_count,
                "success_rate": round(success_rate * 100, 1),
            })

    # ── Assigned tasks unused → suggest DOWNGRADE or REMOVE ──
    for task in emp.get("assigned_tasks", []):
        recent = emp_sched[
            (emp_sched["task_type"] == task)
            & (pd.to_datetime(emp_sched["date"]) >= pd.Timestamp(six_months_ago))
        ]
        last_3mo = emp_sched[
            (emp_sched["task_type"] == task)
            & (pd.to_datetime(emp_sched["date"]) >= pd.Timestamp(three_months_ago))
        ]

        if len(recent) == 0:
            # Not used in 6 months
            last_comp = emp_comps[emp_comps["task_type"] == task]
            last_date = last_comp["date"].max() if len(last_comp) > 0 else None
            days_since = (today - pd.Timestamp(last_date)).days if last_date else 999
            suggestions.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "task": task,
                "action": "REMOVE",
                "reason": (
                    f"'{task}' not scheduled in 6 months "
                    f"(last completion: {days_since} days ago)"
                ),
                "days_since_last": days_since,
            })
        elif len(last_3mo) == 0:
            # Not used in 3 months but used in 6
            suggestions.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "task": task,
                "action": "DOWNGRADE",
                "reason": f"'{task}' not used in last 3 months - downgrade to secondary",
            })
        elif len(last_3mo) >= 10:
            # Frequently used → upgrade
            suggestions.append({
                "employee_id": employee_id,
                "employee_name": emp["name"],
                "task": task,
                "action": "UPGRADE",
                "reason": (
                    f"'{task}' used {len(last_3mo)} times in last 3 months "
                    f"- upgrade to priority"
                ),
                "times_used_3mo": len(last_3mo),
            })

    return suggestions
