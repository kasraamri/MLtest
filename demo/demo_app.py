#!/usr/bin/env python3
"""
PEP BALANCE AI Demo - Main Application

Runs all four demo scenarios with formatted console output:
  1. Unused Availability Detection
  2. Intelligent Task Assignment Ranking
  3. Priority Upgrade Suggestion
  4. Assignment Removal Suggestion

Usage:
    python demo_app.py               # Run all scenarios (synthetic data)
    python demo_app.py --real        # Run all scenarios (real Excel data)
    python demo_app.py --scenario 2  # Run specific scenario (1-4)
"""

import sys
import time
import argparse

from data_generator import generate_all_data, EMPLOYEES
from availability_checker import (
    detect_availability_patterns,
    detect_all_patterns,
    detect_alternating_patterns,
    suggest_priority_changes,
)
from task_scorer import (
    train_model,
    score_employees_for_task,
)


# ─── Display Helpers ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BLUE = "\033[94m"
RESET = "\033[0m"

DOUBLE_LINE = "=" * 60
SINGLE_LINE = "-" * 60
THIN_LINE = "." * 60


def header(text):
    print(f"\n{BOLD}{DOUBLE_LINE}{RESET}")
    print(f"{BOLD}  PEP BALANCE AI DEMO - {text}{RESET}")
    print(f"{BOLD}{DOUBLE_LINE}{RESET}\n")


def section(text):
    print(f"\n{CYAN}{SINGLE_LINE}{RESET}")
    print(f"{CYAN}  {text}{RESET}")
    print(f"{CYAN}{SINGLE_LINE}{RESET}\n")


def bar(value, max_val, width=40):
    """Render a text progress bar."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = min(filled, width)
    return f"{'█' * filled}{'░' * (width - filled)}"


def status_icon(qualified, confidence=None):
    if not qualified:
        return f"{RED}✗{RESET}"
    if confidence and confidence >= 80:
        return f"{GREEN}✓{RESET}"
    return f"{YELLOW}~{RESET}"


# ─── Scenario 1: Unused Availability Detection ───────────────────────────────

def run_scenario_1(data, emp_list):
    header("Scenario 1: Availability Analysis")

    schedule = data["schedule"]
    absences = data["absences"]

    print(f"  {DIM}Analyzing scheduling data...{RESET}")
    print(f"  {DIM}Schedule records: {len(schedule):,}{RESET}")
    print(f"  {DIM}Excluding {len(absences)} absence records{RESET}")
    print()

    findings = detect_all_patterns(schedule, absences, employees=emp_list)

    # Show top findings (high confidence first)
    shown = 0
    for f in findings:
        if f["confidence"] < 50:
            continue
        shown += 1
        if shown > 5:
            break

        conf_color = GREEN if f["confidence"] >= 90 else YELLOW
        sched_word = "Rarely" if f["times_scheduled"] > 0 else "Never"
        print(f"  {BOLD}[{shown}] AVAILABILITY PATTERN DETECTED{RESET}")
        print(f"  Employee: {BOLD}{f['employee_name']}{RESET} (ID: {f['employee_id']:03d})")
        print(f"  Pattern:  {sched_word} scheduled on {f['day']}s despite availability")
        print(f"    - Available: {f['total_available']} {f['day']}s")
        print(f"    - Scheduled: {f['times_scheduled']} times")
        print(f"    - Deviations: {f['deviations']}")
        print(f"    - Confidence: {conf_color}{f['confidence']}%{RESET}")
        print()
        print(f"  {GREEN}-> RECOMMENDATION:{RESET} {f['recommendation']}")
        print(f"     Impact: Simplifies planning, clarifies actual availability")
        print()
        print(f"  {THIN_LINE}")
        print()

    if shown == 0:
        print(f"  {DIM}No patterns detected above 50% confidence.{RESET}")
        # Show lower-confidence findings as info
        low = [f for f in findings if f["confidence"] >= 25]
        if low:
            print(f"  {DIM}Lower-confidence findings:{RESET}")
            for f in low[:3]:
                print(f"    - {f['employee_name']} {f['day']}: "
                      f"{f['deviations']}/{f['total_available']} deviations "
                      f"({f['confidence']}%)")
        print()

    # Alternating patterns - check first employee
    first_emp_id = emp_list[0]["id"] if emp_list else 1
    alt_patterns = detect_alternating_patterns(
        first_emp_id, schedule, absences, employees=emp_list
    )
    if alt_patterns:
        print(f"\n  {BOLD}ALTERNATING PATTERNS:{RESET}")
        for p in alt_patterns:
            print(f"  - {p['employee_name']}: {p['day']} - {p['pattern']}")
            print(f"    Even weeks: {p['even_week_rate']}% scheduled, "
                  f"Odd weeks: {p['odd_week_rate']}% scheduled")


# ─── Scenario 2: Task Assignment Ranking ──────────────────────────────────────

def run_scenario_2(data, emp_list, task_type=None, required_skills=None):
    header("Scenario 2: Task Assignment")

    # Determine best task to demo based on data
    if task_type is None:
        # Pick the most common task in completions
        top_tasks = data["completions"]["task_type"].value_counts()
        task_type = top_tasks.index[0] if len(top_tasks) > 0 else "cashier"

    if required_skills is None:
        required_skills = [task_type]

    shift_start = 8

    print(f"  {BOLD}Task:{RESET} {task_type} - Morning Shift")
    print(f"  {BOLD}Skills Required:{RESET} {', '.join(required_skills)}")
    print()

    results = score_employees_for_task(
        task_type, required_skills,
        data["schedule"], data["completions"],
        shift_start=shift_start,
        employees=emp_list,
    )

    qualified = [r for r in results if r["qualified"]]
    disqualified = [r for r in results if not r["qualified"]]

    print(f"  {GREEN}Qualified Employees:{RESET} {len(qualified)}")
    print(f"  {RED}Disqualified:{RESET} {len(disqualified)}")
    print(f"  {BOLD}Ranked by Score:{RESET}")
    print()

    medals = ["1st", "2nd", "3rd"]
    medal_colors = [GREEN, YELLOW, CYAN]

    for i, r in enumerate(qualified[:3]):
        emp = r["employee"]
        color = medal_colors[i] if i < 3 else ""
        medal = medals[i] if i < 3 else f"#{i+1}"

        ml_icon = f"{GREEN}✓{RESET}" if r["ml_confidence"] >= 75 else f"{YELLOW}~{RESET}"

        print(f"  {color}{BOLD}  {medal}: {emp['name']} "
              f"(Score: {r['score']:.0f}/100){RESET}")
        print(f"      Experience: {bar(r['experience_score'], 40)} "
              f"{r['experience_score']:.0f}/40  ({r['completed_count']} completions)")
        print(f"      Success:    {bar(r['success_score'], 30)} "
              f"{r['success_score']:.0f}/30  ({r['success_rate']:.0f}%)")
        print(f"      Recency:    {bar(r['recency_score'], 20)} "
              f"{r['recency_score']:.0f}/20  ({r['days_since_last']}d ago)")
        print(f"      Workload:   {bar(r['workload_score'], 10)} "
              f"{r['workload_score']:.0f}/10  ({r['current_tasks']} tasks)")
        print(f"      ML Confidence: {r['ml_confidence']:.0f}% {ml_icon}")
        print()

    # Show disqualified
    if disqualified:
        print(f"  {RED}{BOLD}  DISQUALIFIED:{RESET}")
        for r in disqualified[:3]:
            emp = r["employee"]
            print(f"  {RED}  ✗ {emp['name']}{RESET}")
            for reason in r["disqualify_reasons"]:
                print(f"      Reason: {reason}")
            if r["completed_count"] < 5:
                print(f"      {YELLOW}Suggestion: Assign training tasks first{RESET}")
            print()


# ─── Scenario 3: Priority Upgrade Suggestion ─────────────────────────────────

def run_scenario_3(data, emp_list):
    header("Scenario 3: Priority Upgrade")

    # Find employees with ADD or UPGRADE suggestions
    all_add = []
    all_upgrade = []
    for emp in emp_list:
        suggestions = suggest_priority_changes(
            emp["id"], data["schedule"], data["completions"],
            employees=emp_list,
        )
        all_add.extend([s for s in suggestions if s["action"] == "ADD"])
        all_upgrade.extend([s for s in suggestions if s["action"] == "UPGRADE"])

    for s in all_add[:3]:
        print(f"  {BOLD}TASK ASSIGNMENT ADDITION SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\" {DIM}(not officially assigned){RESET}")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {GREEN}-> RECOMMENDATION: ADD \"{s['task']}\" to official task assignments{RESET}")
        print(f"     Reason: Consistently used and performs well")
        print()

    for s in all_upgrade[:3]:
        print(f"  {BOLD}PRIORITY UPGRADE SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\"")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {BLUE}-> RECOMMENDATION: UPGRADE to priority task{RESET}")
        print()

    if not all_add and not all_upgrade:
        print(f"  {DIM}No priority upgrade suggestions found.{RESET}")
        print(f"  {DIM}All employee task assignments align with actual usage.{RESET}")
        print()


# ─── Scenario 4: Assignment Removal Suggestion ───────────────────────────────

def run_scenario_4(data, emp_list):
    header("Scenario 4: Assignment Removal")

    # Find employees with REMOVE or DOWNGRADE suggestions
    all_remove = []
    all_downgrade = []
    for emp in emp_list:
        suggestions = suggest_priority_changes(
            emp["id"], data["schedule"], data["completions"],
            employees=emp_list,
        )
        all_remove.extend([s for s in suggestions if s["action"] == "REMOVE"])
        all_downgrade.extend([s for s in suggestions if s["action"] == "DOWNGRADE"])

    for s in all_remove[:3]:
        days = s.get("days_since_last", "N/A")
        print(f"  {BOLD}ASSIGNMENT REMOVAL SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\" {DIM}(officially assigned){RESET}")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {RED}-> RECOMMENDATION: REMOVE from assignments{RESET}")
        print(f"     Timeline: Downgraded at 3 months, removing at 6 months")
        print()

    for s in all_downgrade[:3]:
        print(f"  {BOLD}ASSIGNMENT DOWNGRADE{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\"")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {YELLOW}-> RECOMMENDATION: Downgrade to secondary{RESET}")
        print()

    if not all_remove and not all_downgrade:
        print(f"  {DIM}No removal/downgrade suggestions found.{RESET}")
        print(f"  {DIM}All assigned tasks are being actively used.{RESET}")
        print()


# ─── Summary ──────────────────────────────────────────────────────────────────

def run_summary(data, emp_list, is_real=False):
    section("DEMO SUMMARY")
    source = "Real Excel Data" if is_real else "Synthetic Data"
    print(f"  {BOLD}Data Source:{RESET} {source}")
    print(f"  {BOLD}Data Overview:{RESET}")
    print(f"    Employees:         {len(emp_list)}")
    print(f"    Schedule records:  {len(data['schedule']):,}")
    print(f"    Absences:          {len(data['absences'])}")
    print(f"    Task completions:  {len(data['completions'])}")
    if is_real and len(data['completions']) > 0:
        sr = data['completions']['success'].mean() * 100
        print(f"    Success rate:      {sr:.1f}% (actual within 15% of planned hours)")
    print()
    print(f"  {BOLD}Key Takeaways:{RESET}")
    print(f"    {GREEN}✓{RESET} Automated detection of unused availability saves manual review")
    print(f"    {GREEN}✓{RESET} Objective scoring removes bias from task assignment")
    print(f"    {GREEN}✓{RESET} ML confidence adds predictive layer to rule-based decisions")
    print(f"    {GREEN}✓{RESET} Priority suggestions keep assignments aligned with actual usage")
    print()
    print(f"  {BOLD}Impact:{RESET} Reduces manual planning work by surfacing actionable insights")
    print(f"  {BOLD}Transparency:{RESET} Every recommendation can be explained with clear data points")
    print()
    print(f"{BOLD}{DOUBLE_LINE}{RESET}")
    print(f"{BOLD}  Demo complete.{RESET}")
    print(f"{BOLD}{DOUBLE_LINE}{RESET}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PEP BALANCE AI Demo")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4],
                        help="Run a specific scenario (1-4). Omit for all.")
    parser.add_argument("--real", action="store_true",
                        help="Use real Excel data instead of synthetic data.")
    args = parser.parse_args()

    is_real = args.real

    print(f"\n{BOLD}{DOUBLE_LINE}{RESET}")
    print(f"{BOLD}  PEP BALANCE AI - Workforce Intelligence Demo{RESET}")
    if is_real:
        print(f"{BOLD}  [REAL DATA MODE]{RESET}")
    print(f"{BOLD}{DOUBLE_LINE}{RESET}")
    print()

    # Step 1: Load data
    if is_real:
        from excel_loader import load_real_data
        print(f"  {DIM}[1/3] Loading real workforce data from Excel...{RESET}")
        t0 = time.time()
        data = load_real_data()
        emp_list = data["employees_list"]
        t1 = time.time()
        print(f"  {GREEN}✓{RESET} Data loaded in {t1 - t0:.2f}s "
              f"({len(emp_list)} employees, "
              f"{len(data['schedule']):,} schedule records, "
              f"{len(data['completions'])} completions)")
    else:
        print(f"  {DIM}[1/3] Generating synthetic workforce data...{RESET}")
        t0 = time.time()
        data = generate_all_data()
        emp_list = EMPLOYEES
        t1 = time.time()
        print(f"  {GREEN}✓{RESET} Data generated in {t1 - t0:.2f}s "
              f"({len(data['schedule']):,} schedule records, "
              f"{len(data['completions'])} completions)")

    # Step 2: Train ML model
    print(f"  {DIM}[2/3] Training ML model (DecisionTreeClassifier)...{RESET}")
    t2 = time.time()
    model, task_map, accuracy = train_model(data["completions"])
    t3 = time.time()
    print(f"  {GREEN}✓{RESET} Model trained in {t3 - t2:.2f}s "
          f"(accuracy: {accuracy * 100:.1f}%, "
          f"{len(data['completions'])} training samples)")

    # Step 3: Run scenarios
    print(f"  {DIM}[3/3] Running demo scenarios...{RESET}")
    print()

    scenarios = {
        1: lambda d: run_scenario_1(d, emp_list),
        2: lambda d: run_scenario_2(d, emp_list),
        3: lambda d: run_scenario_3(d, emp_list),
        4: lambda d: run_scenario_4(d, emp_list),
    }

    if args.scenario:
        scenarios[args.scenario](data)
    else:
        for num in sorted(scenarios):
            scenarios[num](data)

    run_summary(data, emp_list, is_real=is_real)


if __name__ == "__main__":
    main()
