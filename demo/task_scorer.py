"""
PEP BALANCE AI Demo - Task Scorer

Scores employees for task assignment using:
1. Rule-based 40-30-20-10 formula
2. Lightweight ML model (DecisionTreeClassifier) for confidence prediction

Scoring formula:
  Score = Experience(40) + Success_Rate(30) + Recency(20) + Workload(10)
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from data_generator import EMPLOYEES


MODEL_PATH = Path(__file__).parent / "ml_model.pkl"


# ─── ML Model Training ───────────────────────────────────────────────────────

def train_model(completions_df):
    """
    Train a DecisionTreeClassifier to predict task completion success.

    Features: emp_historical_rate, task_encoded, current_workload,
              days_since_last_completion, time_of_day
    Target:   success (bool)
    """
    df = completions_df.copy()
    if len(df) < 10:
        raise ValueError("Need at least 10 records to train")

    # Encode task_type as integer
    task_map = {t: i for i, t in enumerate(sorted(df["task_type"].unique()))}
    df["task_encoded"] = df["task_type"].map(task_map)

    # Per-employee historical success rate (generalizes better than raw ID)
    emp_rates = df.groupby("employee_id")["success"].mean()
    df["emp_historical_rate"] = df["employee_id"].map(emp_rates) * 100

    # Simulate workload: random 0-3 (in real system, would come from schedule)
    rng = np.random.RandomState(42)
    df["current_workload"] = rng.randint(0, 4, size=len(df))

    # Days since last completion for same employee+task (approx)
    df = df.sort_values(["employee_id", "task_type", "date"])
    df["prev_date"] = df.groupby(["employee_id", "task_type"])["date"].shift(1)
    df["days_since_last"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_date"])
    ).dt.days.fillna(30).clip(upper=90)

    features = ["emp_historical_rate", "task_encoded", "current_workload",
                "days_since_last", "time_of_day"]
    X = df[features].values
    y = df["success"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # Save model and metadata, including employee rates for prediction
    artifact = {
        "model": model,
        "task_map": task_map,
        "emp_rates": emp_rates.to_dict(),
        "accuracy": accuracy,
        "feature_names": features,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    return model, task_map, accuracy


def load_model():
    """Load saved model and metadata."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_task_success(employee_id, task_type, current_workload=1,
                         days_since_last=7, time_of_day=10):
    """
    ML model prediction for task success.

    Returns probability 0-100%.
    """
    artifact = load_model()
    model = artifact["model"]
    task_map = artifact["task_map"]
    emp_rates = artifact.get("emp_rates", {})

    task_encoded = task_map.get(task_type, 0)
    # Use historical rate; default to 80% for unseen employees
    emp_rate = emp_rates.get(employee_id, 0.80) * 100
    features = np.array([[emp_rate, task_encoded, current_workload,
                          days_since_last, time_of_day]])

    proba = model.predict_proba(features)
    # Index 1 = success probability
    success_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    confidence = proba[0][success_idx] * 100
    return round(confidence, 1)


# ─── Rule-Based Scoring ──────────────────────────────────────────────────────

def _calc_experience_score(completed_count, max_points=40):
    """Experience = (completed_count / 10) * 40, capped at 40."""
    return min((completed_count / 10) * max_points, max_points)


def _calc_success_rate_score(success_pct, max_points=30):
    """Success_Rate = (success_% / 100) * 30."""
    return min((success_pct / 100) * max_points, max_points)


def _calc_recency_score(days_since_last, max_points=20):
    """Recency score: decays over time. Max 20 at 0 days, ~0 at 90+ days."""
    if days_since_last <= 0:
        return max_points
    # Inverse decay: 20 * (1 / (1 + days/10))
    score = max_points * (1 / (1 + days_since_last / 10))
    return min(round(score, 1), max_points)


def _calc_workload_score(current_tasks, max_points=10):
    """Workload = (3 - current_tasks) * 3.33, capped at 10."""
    return max(0, min((3 - current_tasks) * 3.33, max_points))


def check_qualifications(employee, task_type, required_skills,
                         completions_df, schedule_df, shift_start=None):
    """
    Check if employee passes all qualification filters.

    Returns (qualified: bool, reasons: list[str])
    """
    reasons = []
    qualified = True

    # 1. Has required skills?
    emp_skills = set(employee.get("skills", []))
    missing = set(required_skills) - emp_skills
    if missing:
        qualified = False
        reasons.append(f"Missing skills: {', '.join(missing)}")

    # 2. Completion threshold: 5+ times with 80%+ success
    emp_comps = completions_df[
        (completions_df["employee_id"] == employee["id"])
        & (completions_df["task_type"] == task_type)
    ]
    comp_count = len(emp_comps)
    success_rate = emp_comps["success"].mean() * 100 if comp_count > 0 else 0

    if comp_count < 5:
        qualified = False
        reasons.append(f"Only {comp_count} completions (needs 5)")
    elif success_rate < 80:
        qualified = False
        reasons.append(f"Success rate {success_rate:.0f}% (needs 80%+)")

    # 3. Under 3 concurrent tasks (simulated)
    emp_sched_today = schedule_df[
        (schedule_df["employee_id"] == employee["id"])
        & (schedule_df["scheduled"])
    ]
    # Approximate current tasks from recent schedule
    current_tasks = min(len(emp_sched_today) % 4, 3)  # Simplified for demo

    if current_tasks >= 3:
        qualified = False
        reasons.append(f"At max capacity ({current_tasks} concurrent tasks)")

    return qualified, reasons, comp_count, success_rate, current_tasks


def score_employees_for_task(task_type, required_skills, schedule_df,
                             completions_df, shift_start=None):
    """
    Calculate assignment scores for all employees using the 40-30-20-10 formula.

    Returns list of dicts sorted by score descending:
        {
            "employee": dict,
            "qualified": bool,
            "disqualify_reasons": list,
            "score": float,
            "experience_score": float,
            "success_score": float,
            "recency_score": float,
            "workload_score": float,
            "completed_count": int,
            "success_rate": float,
            "days_since_last": int,
            "current_tasks": int,
            "ml_confidence": float,
        }
    """
    today = pd.Timestamp(schedule_df["date"].max())
    results = []

    for emp in EMPLOYEES:
        qualified, reasons, comp_count, success_rate, current_tasks = (
            check_qualifications(
                emp, task_type, required_skills,
                completions_df, schedule_df, shift_start
            )
        )

        # Days since last completion of this task type
        emp_task_comps = completions_df[
            (completions_df["employee_id"] == emp["id"])
            & (completions_df["task_type"] == task_type)
        ]
        if len(emp_task_comps) > 0:
            last_date = pd.Timestamp(emp_task_comps["date"].max())
            days_since = (today - last_date).days
        else:
            days_since = 999

        # Calculate sub-scores
        exp_score = _calc_experience_score(comp_count)
        suc_score = _calc_success_rate_score(success_rate)
        rec_score = _calc_recency_score(days_since)
        wkl_score = _calc_workload_score(current_tasks)
        total = round(exp_score + suc_score + rec_score + wkl_score, 1)

        # ML confidence
        ml_conf = 0
        if MODEL_PATH.exists():
            try:
                ml_conf = predict_task_success(
                    emp["id"], task_type,
                    current_workload=current_tasks,
                    days_since_last=days_since,
                    time_of_day=shift_start or 10,
                )
            except Exception:
                ml_conf = 0

        results.append({
            "employee": emp,
            "qualified": qualified,
            "disqualify_reasons": reasons,
            "score": total,
            "experience_score": round(exp_score, 1),
            "success_score": round(suc_score, 1),
            "recency_score": round(rec_score, 1),
            "workload_score": round(wkl_score, 1),
            "completed_count": comp_count,
            "success_rate": round(success_rate, 1),
            "days_since_last": days_since,
            "current_tasks": current_tasks,
            "ml_confidence": ml_conf,
        })

    # Qualified first, then sorted by score
    results.sort(key=lambda r: (r["qualified"], r["score"]), reverse=True)
    return results
