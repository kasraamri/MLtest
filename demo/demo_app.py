#!/usr/bin/env python3
"""
PEP BALANCE AI Demo - Main Application

Runs all four demo scenarios with formatted console output:
  1. Unused Availability Detection
  2. Intelligent Task Assignment Ranking
  3. Priority Upgrade Suggestion
  4. Assignment Removal Suggestion

Usage:
    python demo_app.py          # Run all scenarios
    python demo_app.py --scenario 1   # Run specific scenario (1-4)
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

def run_scenario_1(data):
    header("Scenario 1: Availability Analysis")

    schedule = data["schedule"]
    absences = data["absences"]

    # Focus on Maria (ID 1) - the strongest pattern
    print(f"  {DIM}Analyzing 6 months of scheduling data...{RESET}")
    print(f"  {DIM}Schedule records: {len(schedule):,}{RESET}")
    print(f"  {DIM}Excluding {len(absences)} absence records{RESET}")
    print()

    findings = detect_all_patterns(schedule, absences)

    # Show top findings (high confidence first)
    shown = 0
    for f in findings:
        if f["confidence"] < 80:
            continue
        shown += 1
        if shown > 3:
            break

        conf_color = GREEN if f["confidence"] >= 90 else YELLOW
        print(f"  {BOLD}[{shown}] AVAILABILITY PATTERN DETECTED{RESET}")
        print(f"  Employee: {BOLD}{f['employee_name']}{RESET} (ID: {f['employee_id']:03d})")
        print(f"  Pattern:  Never scheduled on {f['day']}s despite availability")
        print(f"    - Available: {f['total_available']} {f['day']}s (last 6 months)")
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
        print(f"  {DIM}No high-confidence patterns detected.{RESET}")

    # Alternating patterns
    alt_patterns = detect_alternating_patterns(1, schedule, absences)
    if alt_patterns:
        print(f"\n  {BOLD}ALTERNATING PATTERNS:{RESET}")
        for p in alt_patterns:
            print(f"  - {p['employee_name']}: {p['day']} - {p['pattern']}")
            print(f"    Even weeks: {p['even_week_rate']}% scheduled, "
                  f"Odd weeks: {p['odd_week_rate']}% scheduled")


# ─── Scenario 2: Task Assignment Ranking ──────────────────────────────────────

def run_scenario_2(data):
    header("Scenario 2: Task Assignment")

    task_type = "cashier"
    required_skills = ["cashier", "customer_service"]
    shift_start = 14

    print(f"  {BOLD}Task:{RESET} Cashier Shift - Tuesday 14:00-18:00")
    print(f"  {BOLD}Skills Required:{RESET} {', '.join(required_skills)}")
    print()

    results = score_employees_for_task(
        task_type, required_skills,
        data["schedule"], data["completions"],
        shift_start=shift_start,
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

def run_scenario_3(data):
    header("Scenario 3: Priority Upgrade")

    # John (ID 2) - frequently does cleaning (not officially assigned)
    suggestions = suggest_priority_changes(2, data["schedule"], data["completions"])

    add_suggestions = [s for s in suggestions if s["action"] == "ADD"]
    upgrade_suggestions = [s for s in suggestions if s["action"] == "UPGRADE"]

    for s in add_suggestions:
        print(f"  {BOLD}TASK ASSIGNMENT ADDITION SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\" {DIM}(not officially assigned){RESET}")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {GREEN}-> RECOMMENDATION: ADD \"{s['task']}\" to official task assignments{RESET}")
        print(f"     Reason: Consistently used and performs well")
        print()

    for s in upgrade_suggestions:
        print(f"  {BOLD}PRIORITY UPGRADE SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\"")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {BLUE}-> RECOMMENDATION: UPGRADE to priority task{RESET}")
        print()

    if not add_suggestions and not upgrade_suggestions:
        # Fallback: show all suggestions for John
        print(f"  {DIM}Checking priority changes for John Muller...{RESET}")
        for s in suggestions:
            print(f"  - {s['action']}: {s['reason']}")
        print()


# ─── Scenario 4: Assignment Removal Suggestion ───────────────────────────────

def run_scenario_4(data):
    header("Scenario 4: Assignment Removal")

    # Maria (ID 1) - special_events never used
    suggestions = suggest_priority_changes(1, data["schedule"], data["completions"])

    remove_suggestions = [s for s in suggestions if s["action"] == "REMOVE"]
    downgrade_suggestions = [s for s in suggestions if s["action"] == "DOWNGRADE"]

    for s in remove_suggestions:
        days = s.get("days_since_last", "N/A")
        print(f"  {BOLD}ASSIGNMENT REMOVAL SUGGESTED{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET} (ID: {s['employee_id']:03d})")
        print(f"  Task:     \"{s['task']}\" {DIM}(officially assigned){RESET}")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {RED}-> RECOMMENDATION: REMOVE from assignments{RESET}")
        print(f"     Timeline: Downgraded at 3 months, removing at 6 months")
        print()

    for s in downgrade_suggestions:
        print(f"  {BOLD}ASSIGNMENT DOWNGRADE{RESET}")
        print(f"  Employee: {BOLD}{s['employee_name']}{RESET}")
        print(f"  Task:     \"{s['task']}\"")
        print(f"  Finding:  {s['reason']}")
        print()
        print(f"  {YELLOW}-> RECOMMENDATION: Downgrade to secondary{RESET}")
        print()

    if not remove_suggestions and not downgrade_suggestions:
        print(f"  {DIM}No removal/downgrade suggestions for Maria.{RESET}")
        for s in suggestions:
            print(f"  - {s['action']}: {s['reason']}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def run_summary(data):
    section("DEMO SUMMARY")
    print(f"  {BOLD}Data Generated:{RESET}")
    print(f"    Employees:         {len(EMPLOYEES)}")
    print(f"    Schedule records:  {len(data['schedule']):,}")
    print(f"    Absences:          {len(data['absences'])}")
    print(f"    Task completions:  {len(data['completions'])}")
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
    args = parser.parse_args()

    print(f"\n{BOLD}{DOUBLE_LINE}{RESET}")
    print(f"{BOLD}  PEP BALANCE AI - Workforce Intelligence Demo{RESET}")
    print(f"{BOLD}{DOUBLE_LINE}{RESET}")
    print()

    # Step 1: Generate data
    print(f"  {DIM}[1/3] Generating 6 months of synthetic workforce data...{RESET}")
    t0 = time.time()
    data = generate_all_data()
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
        1: run_scenario_1,
        2: run_scenario_2,
        3: run_scenario_3,
        4: run_scenario_4,
    }

    if args.scenario:
        scenarios[args.scenario](data)
    else:
        for num in sorted(scenarios):
            scenarios[num](data)

    run_summary(data)


if __name__ == "__main__":
    main()
