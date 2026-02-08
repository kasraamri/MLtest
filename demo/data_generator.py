"""
PEP BALANCE AI Demo - Synthetic Data Generator

Generates realistic workforce data for demonstration:
- 15 employees with varied skill profiles
- 6 months of scheduling history (200+ records)
- 150 task completion records
- Intentional patterns for demo scenarios
"""

import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────

DEMO_START = datetime(2025, 8, 1)
DEMO_END = datetime(2026, 1, 31)
TASK_TYPES = ["cashier", "stock", "cleaning", "customer_service", "special_events"]

EMPLOYEES = [
    {
        "id": 1, "name": "Maria Schmidt",
        "skills": ["cashier", "stock", "customer_service"],
        "assigned_tasks": ["cashier", "stock", "special_events"],
        "availability": {
            "Mon": (8, 16), "Tue": (8, 12), "Wed": (8, 16),
            "Thu": (8, 16), "Fri": (8, 16),
        },
    },
    {
        "id": 2, "name": "John Müller",
        "skills": ["cashier", "cleaning", "customer_service"],
        "assigned_tasks": ["cashier", "customer_service"],
        "availability": {
            "Mon": (10, 18), "Tue": (10, 18), "Wed": (10, 18),
            "Thu": (10, 18), "Fri": (10, 18), "Sat": (10, 18), "Sun": (10, 18),
        },
    },
    {
        "id": 3, "name": "Sarah Weber",
        "skills": ["cashier", "customer_service"],
        "assigned_tasks": ["cashier", "customer_service"],
        "availability": {
            "Mon": (8, 16), "Tue": (8, 16), "Wed": (8, 16),
            "Thu": (8, 16), "Fri": (8, 16),
        },
    },
    {
        "id": 4, "name": "Alex Fischer",
        "skills": ["cashier", "stock", "cleaning"],
        "assigned_tasks": ["stock", "cleaning"],
        "availability": {
            "Mon": (12, 20), "Tue": (12, 20), "Wed": (12, 20),
            "Thu": (12, 20), "Fri": (12, 20),
        },
    },
    {
        "id": 5, "name": "Lisa Bauer",
        "skills": ["cashier", "customer_service", "special_events"],
        "assigned_tasks": ["cashier", "customer_service", "special_events"],
        "availability": {
            "Mon": (8, 14), "Wed": (8, 14), "Fri": (8, 14),
        },
    },
    {
        "id": 6, "name": "Tom Schneider",
        "skills": ["stock", "cleaning"],
        "assigned_tasks": ["stock", "cleaning"],
        "availability": {
            "Mon": (6, 14), "Tue": (6, 14), "Wed": (6, 14),
            "Thu": (6, 14), "Fri": (6, 14),
        },
    },
    {
        "id": 7, "name": "Emma Wagner",
        "skills": ["cashier", "customer_service", "special_events"],
        "assigned_tasks": ["cashier", "customer_service"],
        "availability": {
            "Tue": (10, 18), "Thu": (10, 18), "Sat": (10, 18),
        },
    },
    {
        "id": 8, "name": "Max Klein",
        "skills": ["stock", "cleaning", "cashier"],
        "assigned_tasks": ["stock"],
        "availability": {
            "Mon": (8, 16), "Tue": (8, 16), "Wed": (8, 16),
            "Thu": (8, 16), "Fri": (8, 16),
        },
    },
    {
        "id": 9, "name": "Laura Hoffmann",
        "skills": ["cashier", "customer_service"],
        "assigned_tasks": ["cashier", "customer_service"],
        "availability": {
            "Mon": (14, 20), "Tue": (14, 20), "Wed": (14, 20),
            "Thu": (14, 20), "Fri": (14, 20), "Sat": (10, 16),
        },
    },
    {
        "id": 10, "name": "Daniel Braun",
        "skills": ["cleaning", "stock"],
        "assigned_tasks": ["cleaning", "stock"],
        "availability": {
            "Mon": (6, 12), "Tue": (6, 12), "Wed": (6, 12),
            "Thu": (6, 12), "Fri": (6, 12),
        },
    },
    {
        "id": 11, "name": "Sophie Richter",
        "skills": ["cashier", "customer_service", "cleaning"],
        "assigned_tasks": ["cashier", "customer_service"],
        "availability": {
            "Mon": (8, 16), "Wed": (8, 16), "Thu": (8, 16),
            "Fri": (8, 16), "Sat": (8, 14),
        },
    },
    {
        "id": 12, "name": "Felix Neumann",
        "skills": ["stock", "cleaning", "special_events"],
        "assigned_tasks": ["stock", "special_events"],
        "availability": {
            "Mon": (10, 18), "Tue": (10, 18), "Wed": (10, 18),
            "Thu": (10, 18), "Fri": (10, 18),
        },
    },
    {
        "id": 13, "name": "Hannah Zimmermann",
        "skills": ["cashier", "customer_service"],
        "assigned_tasks": ["cashier"],
        "availability": {
            "Tue": (8, 16), "Wed": (8, 16), "Thu": (8, 16),
        },
    },
    {
        "id": 14, "name": "Lukas Schwarz",
        "skills": ["stock", "cleaning", "cashier"],
        "assigned_tasks": ["stock", "cleaning"],
        "availability": {
            "Mon": (8, 16), "Tue": (8, 16), "Wed": (8, 16),
            "Thu": (8, 16), "Fri": (8, 16), "Sat": (8, 12),
        },
    },
    {
        "id": 15, "name": "Anna Krause",
        "skills": ["cashier", "customer_service", "special_events", "cleaning"],
        "assigned_tasks": ["cashier", "customer_service", "special_events"],
        "availability": {
            "Mon": (8, 16), "Tue": (8, 16), "Wed": (8, 16),
            "Thu": (8, 16), "Fri": (8, 16),
        },
    },
]

DAY_ABBREV = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _random_time_slot(start_hour, end_hour):
    """Return a (slot_start, slot_end) tuple within the given range."""
    slot_start = random.randint(start_hour, max(start_hour, end_hour - 4))
    slot_end = min(slot_start + 4, end_hour)
    return slot_start, slot_end


def _date_range(start, end):
    """Yield each date between start and end (inclusive)."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ─── Absence Generator ───────────────────────────────────────────────────────

def generate_absences():
    """Generate random sick days and vacation periods for each employee."""
    absences = []
    for emp in EMPLOYEES:
        # 2-5 random sick days
        sick_days = random.randint(2, 5)
        for _ in range(sick_days):
            day = DEMO_START + timedelta(days=random.randint(0, (DEMO_END - DEMO_START).days))
            absences.append({
                "employee_id": emp["id"],
                "date": day.date(),
                "type": "sick",
            })
        # 1 vacation block of 5-10 days
        vac_start = DEMO_START + timedelta(days=random.randint(30, 150))
        vac_len = random.randint(5, 10)
        for offset in range(vac_len):
            absences.append({
                "employee_id": emp["id"],
                "date": (vac_start + timedelta(days=offset)).date(),
                "type": "vacation",
            })
    return pd.DataFrame(absences)


# ─── Schedule History Generator ───────────────────────────────────────────────

def generate_schedule_history(absences_df):
    """
    Generate 6 months of scheduling records.

    Intentional patterns baked in:
    - Maria (ID 1): NEVER scheduled on Tuesdays (triggers availability adjustment)
    - Maria (ID 1): NEVER scheduled for special_events (triggers assignment removal)
    - John (ID 2): Frequently assigned cleaning (not his official task) → triggers add
    - Alex (ID 4): Only 4 cashier completions (below threshold)
    """
    absence_set = set()
    for _, row in absences_df.iterrows():
        absence_set.add((row["employee_id"], row["date"]))

    records = []
    for d in _date_range(DEMO_START, DEMO_END):
        day_name = DAY_ABBREV[d.weekday()]

        for emp in EMPLOYEES:
            if day_name not in emp["availability"]:
                continue
            if (emp["id"], d.date()) in absence_set:
                continue

            avail_start, avail_end = emp["availability"][day_name]

            # ── Maria: never scheduled Tuesdays ──
            if emp["id"] == 1 and day_name == "Tue":
                records.append({
                    "employee_id": emp["id"],
                    "date": d.date(),
                    "day": day_name,
                    "available_start": avail_start,
                    "available_end": avail_end,
                    "scheduled": False,
                    "task_type": None,
                    "shift_start": None,
                    "shift_end": None,
                })
                continue

            # Normal scheduling: ~70 % chance of being scheduled
            scheduled = random.random() < 0.70
            task_type = None
            shift_start = shift_end = None

            if scheduled:
                # Pick a task from skills (weighted toward assigned tasks)
                pool = emp["assigned_tasks"] * 3 + emp["skills"]
                # Maria: never gets special_events
                if emp["id"] == 1:
                    pool = [t for t in pool if t != "special_events"]
                # John: extra cleaning assignments (not officially assigned)
                if emp["id"] == 2 and random.random() < 0.35:
                    task_type = "cleaning"
                else:
                    task_type = random.choice(pool)
                shift_start, shift_end = _random_time_slot(avail_start, avail_end)

            records.append({
                "employee_id": emp["id"],
                "date": d.date(),
                "day": day_name,
                "available_start": avail_start,
                "available_end": avail_end,
                "scheduled": scheduled,
                "task_type": task_type,
                "shift_start": shift_start,
                "shift_end": shift_end,
            })

    return pd.DataFrame(records)


# ─── Task Completion History ──────────────────────────────────────────────────

def generate_task_completions(schedule_df):
    """
    Build ~150 task completion records from the schedule.

    Patterns:
    - John (ID 2): high cashier success, decent cleaning success
    - Sarah (ID 3): good cashier, moderate success
    - Alex (ID 4): only 4 cashier completions (below 5 threshold)
    - Maria (ID 1): 0 special_events completions in 6 months
    """
    scheduled = schedule_df[schedule_df["scheduled"]].copy()
    scheduled = scheduled.sample(frac=1, random_state=42).reset_index(drop=True)

    completions = []
    emp_task_counts = {}  # (emp_id, task_type) → count

    # Employee-specific success rates
    success_profiles = {
        1: {"cashier": 0.90, "stock": 0.85, "special_events": 0.0},
        2: {"cashier": 0.92, "cleaning": 0.86, "customer_service": 0.88},
        3: {"cashier": 0.88, "customer_service": 0.82},
        4: {"stock": 0.80, "cleaning": 0.75, "cashier": 0.70},
    }

    target = 150
    for _, row in scheduled.iterrows():
        if len(completions) >= target:
            break
        eid = row["employee_id"]
        task = row["task_type"]
        if task is None:
            continue

        key = (eid, task)
        count = emp_task_counts.get(key, 0)

        # Alex: cap cashier at 4
        if eid == 4 and task == "cashier" and count >= 4:
            continue

        # Determine success
        profile = success_profiles.get(eid, {})
        base_rate = profile.get(task, 0.80)
        success = random.random() < base_rate

        duration_minutes = random.randint(60, 240)
        completions.append({
            "employee_id": eid,
            "task_type": task,
            "date": row["date"],
            "shift_start": row["shift_start"],
            "shift_end": row["shift_end"],
            "success": success,
            "duration_minutes": duration_minutes,
            "time_of_day": row["shift_start"] if row["shift_start"] else 8,
        })
        emp_task_counts[key] = count + 1

    return pd.DataFrame(completions)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_all_data():
    """Generate and return all demo datasets."""
    absences = generate_absences()
    schedule = generate_schedule_history(absences)
    completions = generate_task_completions(schedule)

    return {
        "employees": pd.DataFrame(EMPLOYEES),
        "absences": absences,
        "schedule": schedule,
        "completions": completions,
    }


if __name__ == "__main__":
    data = generate_all_data()
    print(f"Employees:   {len(data['employees'])} records")
    print(f"Absences:    {len(data['absences'])} records")
    print(f"Schedule:    {len(data['schedule'])} records")
    print(f"Completions: {len(data['completions'])} records")

    # Quick sanity checks
    sched = data["schedule"]
    maria_tue = sched[(sched["employee_id"] == 1) & (sched["day"] == "Tue")]
    print(f"\nMaria Tue available: {len(maria_tue)}, "
          f"scheduled: {maria_tue['scheduled'].sum()}")

    comps = data["completions"]
    alex_cashier = comps[(comps["employee_id"] == 4) & (comps["task_type"] == "cashier")]
    print(f"Alex cashier completions: {len(alex_cashier)}")
