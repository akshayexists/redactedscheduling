from datetime import datetime, timedelta
import pandas as pd
import random
import math
import json
from ortools.sat.python import cp_model
from dateutil.relativedelta import relativedelta

##############################
# Data Loading & Configuration
##############################

def load_data(filename="data.json"):
    """
    Loads all input from the JSON file.
    Expected keys:
      - "total_periods": integer scheduling horizon (default 52)
      - "base_date": start date in YYYY-MM-DD format (default "2025-03-31")
      - "period_type": unit of scheduling period ("minutes", "hours", "days", "weeks", "months", "years"; default "weeks")
      - "period_length": length of each period (default 1)
      - "output_filename": name of the Excel file to export (default "schedule.xlsx")
      - "sheet_name": sheet name for Excel output (default "Packed")
      - "task_label": label for the scheduled task (default "event")
      - "penalty_parameters": object with keys "consec_penalty", "extra_penalty", and "spacing_threshold"
      - "event_types": list of event type objects. Each may optionally include "final_week" and "deadline".
      - "reserved", "public_holidays", "unavailable": lists of period numbers.
    Returns:
      A dictionary with the loaded configuration.
    """
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except Exception as e:
        print("Error reading JSON file:", e)
        exit(1)
    
    # Set defaults if not provided.
    data.setdefault("total_periods", 52)
    data.setdefault("base_date", "2025-03-31")
    data.setdefault("period_type", "weeks")
    data.setdefault("period_length", 1)
    data.setdefault("output_filename", "schedule.xlsx")
    data.setdefault("sheet_name", "Packed")
    data.setdefault("task_label", "event")
    data.setdefault("penalty_parameters", {"consec_penalty": 500, "extra_penalty": 300, "spacing_threshold": 4})
    
    return data

##############################
# Helper Functions
##############################

def add_period(base_date, period_number, period_type, period_length):
    """
    Returns a new datetime by adding (period_number-1)*period_length of period_type to base_date.
    period_type can be: minutes, hours, days, weeks, months, or years.
    """
    multiplier = period_number - 1
    pt = period_type.lower()
    if pt == "minutes":
        return base_date + timedelta(minutes=multiplier * period_length)
    elif pt == "hours":
        return base_date + timedelta(hours=multiplier * period_length)
    elif pt == "days":
        return base_date + timedelta(days=multiplier * period_length)
    elif pt == "weeks":
        return base_date + timedelta(weeks=multiplier * period_length)
    elif pt == "months":
        return base_date + relativedelta(months=multiplier * period_length)
    elif pt == "years":
        return base_date + relativedelta(years=multiplier * period_length)
    else:
        return base_date + timedelta(weeks=multiplier * period_length)

def extract_division(s, task_label="event"):
    """
    Extracts the division (or task type) name from a schedule string.
    E.g., if s is "Marine event" or "Marine event #3", returns "Marine".
    """
    delim = f" {task_label}"
    if delim in s:
        return s.split(delim)[0]
    return s

##############################
# CP-SAT Model: Packed Schedule
##############################

def cp_schedule_packed(fixed, total_periods, event_types, task_label, consec_penalty, extra_penalty, spacing_threshold):
    # Compute free periods (those not fixed)
    free_periods = sorted([p for p in range(1, total_periods+1) if fixed[p] is None])
    R = sum(et["regular_count"] for et in event_types)
    if len(free_periods) < R:
        print("Error: Not enough free periods to schedule all tasks.")
        exit(1)
    # Use the first R free periods.
    used_free_periods = free_periods[:R]
    eligible_for_period = {}
    for idx, period in enumerate(used_free_periods):
        # For event types with no final_week, assume they can be scheduled any time.
        eligible = [i for i, et in enumerate(event_types) if period < et.get("final_week", total_periods + 1)]
        if not eligible:
            print(f"Error: No division is eligible in free period {period}.")
            exit(1)
        eligible_for_period[idx] = eligible

    model = cp_model.CpModel()
    n = len(event_types)
    # Decision variables: d[j] is the division assigned to free period index j.
    d = []
    for j in range(R):
        d_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(eligible_for_period[j]), f"d_{j}")
        d.append(d_var)

    # Enforce each division's task count.
    for i in range(n):
        x_vars = []
        for j in range(R):
            x_ji = model.NewBoolVar(f"x_{j}_{i}")
            model.Add(d[j] == i).OnlyEnforceIf(x_ji)
            model.Add(d[j] != i).OnlyEnforceIf(x_ji.Not())
            x_vars.append(x_ji)
        model.Add(sum(x_vars) == event_types[i]["regular_count"])

    penalty_terms = []
    # High penalty for consecutive repetition.
    for j in range(R-1):
        b = model.NewBoolVar(f"b_{j}")
        model.Add(d[j] == d[j+1]).OnlyEnforceIf(b)
        model.Add(d[j] != d[j+1]).OnlyEnforceIf(b.Not())
        penalty_terms.append(consec_penalty * b)
    # Extra penalty for near-adjacency if gap < spacing_threshold.
    for j in range(R):
        for k in range(j+2, R):
            gap = used_free_periods[k] - used_free_periods[j]
            if gap < spacing_threshold:
                b = model.NewBoolVar(f"b_{j}_{k}")
                model.Add(d[j] == d[k]).OnlyEnforceIf(b)
                model.Add(d[j] != d[k]).OnlyEnforceIf(b.Not())
                penalty_terms.append(extra_penalty * b)
    model.Minimize(sum(penalty_terms))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found for packed schedule!")
        exit(1)
    
    regular_assignments = {}
    for j in range(R):
        division_index = solver.Value(d[j])
        period = used_free_periods[j]
        regular_assignments[period] = f"{event_types[division_index]['name']} {task_label}"
    
    schedule = {}
    for period in range(1, total_periods+1):
        if fixed[period] is not None:
            schedule[period] = fixed[period]
        elif period in regular_assignments:
            schedule[period] = regular_assignments[period]
        else:
            schedule[period] = ""
    return schedule, solver.ObjectiveValue(), used_free_periods

##############################
# Variety Penalty Function
##############################

def compute_variety_penalty(schedule, total_periods, task_label="event"):
    penalty = 0
    high_penalty = 1000  # penalty for consecutive same-division tasks
    for period in range(1, total_periods):
        if task_label in schedule[period] and task_label in schedule[period+1]:
            div1 = extract_division(schedule[period], task_label)
            div2 = extract_division(schedule[period+1], task_label)
            if div1 == div2:
                penalty += high_penalty
    return penalty

##############################
# Simulated Annealing Improvement
##############################

def simulated_annealing_improvement(schedule, used_free_periods, total_periods, event_types, task_label="event"):
    T = 1000.0
    T_min = 1.0
    alpha = 0.95
    current = schedule.copy()
    current_penalty = compute_variety_penalty(current, total_periods, task_label)
    best = current.copy()
    best_penalty = current_penalty

    while T > T_min:
        i, j = random.sample(used_free_periods, 2)
        candidate = current.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        # Check feasibility: each task must occur before its division's final_week.
        feasible = True
        for period in (i, j):
            if task_label in candidate[period]:
                div = extract_division(candidate[period], task_label)
                for et in event_types:
                    if et["name"] == div and period >= et.get("final_week", total_periods + 1):
                        feasible = False
                        break
            if not feasible:
                break
        if not feasible:
            T *= alpha
            continue
        cand_penalty = compute_variety_penalty(candidate, total_periods, task_label)
        delta = cand_penalty - current_penalty
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate.copy()
            current_penalty = cand_penalty
            if current_penalty < best_penalty:
                best = current.copy()
                best_penalty = current_penalty
        T *= alpha
    return best, best_penalty

##############################
# Simple Tabu Search Improvement
##############################

def tabu_search_improvement(schedule, used_free_periods, total_periods, event_types, task_label="event", iterations=100, tabu_tenure=5):
    current = schedule.copy()
    current_penalty = compute_variety_penalty(current, total_periods, task_label)
    best = current.copy()
    best_penalty = current_penalty
    tabu_list = {}
    for _ in range(iterations):
        best_candidate = None
        best_candidate_penalty = float('inf')
        move = None
        for i in used_free_periods:
            for j in used_free_periods:
                if i >= j:
                    continue
                if (i, j) in tabu_list:
                    continue
                candidate = current.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                feasible = True
                for period in (i, j):
                    if task_label in candidate[period]:
                        div = extract_division(candidate[period], task_label)
                        for et in event_types:
                            if et["name"] == div and period >= et.get("final_week", total_periods + 1):
                                feasible = False
                                break
                    if not feasible:
                        break
                if not feasible:
                    continue
                cand_penalty = compute_variety_penalty(candidate, total_periods, task_label)
                if cand_penalty < best_candidate_penalty:
                    best_candidate = candidate.copy()
                    best_candidate_penalty = cand_penalty
                    move = (i, j)
        if best_candidate is not None and best_candidate_penalty < current_penalty:
            current = best_candidate.copy()
            current_penalty = best_candidate_penalty
            tabu_list[move] = tabu_tenure
            if current_penalty < best_penalty:
                best = current.copy()
                best_penalty = current_penalty
        tabu_list = {move: tenure-1 for move, tenure in tabu_list.items() if tenure-1 > 0}
    return best, best_penalty

##############################
# Hybrid Reoptimization using CP-SAT with Bound Constraint
##############################

def cp_reoptimize_with_bound(fixed, total_periods, event_types, bound, consec_penalty, extra_penalty, task_label):
    free_periods = sorted([p for p in range(1, total_periods+1) if fixed[p] is None])
    R = sum(et["regular_count"] for et in event_types)
    if len(free_periods) < R:
        print("Error: Not enough free periods to schedule all tasks.")
        exit(1)
    used_free_periods = free_periods[:R]
    eligible_for_period = {}
    for idx, period in enumerate(used_free_periods):
        eligible = [i for i, et in enumerate(event_types) if period < et.get("final_week", total_periods + 1)]
        if not eligible:
            print(f"Error: No division is eligible in free period {period}.")
            exit(1)
        eligible_for_period[idx] = eligible

    model = cp_model.CpModel()
    n = len(event_types)
    d = []
    for j in range(R):
        d_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(eligible_for_period[j]), f"d_{j}")
        d.append(d_var)

    for i in range(n):
        x_vars = []
        for j in range(R):
            x_ji = model.NewBoolVar(f"x_{j}_{i}")
            model.Add(d[j] == i).OnlyEnforceIf(x_ji)
            model.Add(d[j] != i).OnlyEnforceIf(x_ji.Not())
            x_vars.append(x_ji)
        model.Add(sum(x_vars) == event_types[i]["regular_count"])

    penalty_terms = []
    for j in range(R-1):
        b = model.NewBoolVar(f"b_{j}")
        model.Add(d[j] == d[j+1]).OnlyEnforceIf(b)
        model.Add(d[j] != d[j+1]).OnlyEnforceIf(b.Not())
        penalty_terms.append(consec_penalty * b)
    spacing_threshold = 4
    for j in range(R):
        for k in range(j+2, R):
            gap = used_free_periods[k] - used_free_periods[j]
            if gap < spacing_threshold:
                b = model.NewBoolVar(f"b_{j}_{k}")
                model.Add(d[j] == d[k]).OnlyEnforceIf(b)
                model.Add(d[j] != d[k]).OnlyEnforceIf(b.Not())
                penalty_terms.append(extra_penalty * b)
    total_obj = model.NewIntVar(0, 1000000, "total_obj")
    model.Add(total_obj == sum(penalty_terms))
    model.Add(total_obj <= bound)
    model.Minimize(total_obj)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found in reoptimization!")
        exit(1)
    
    regular_assignments = {}
    for j in range(R):
        division_index = solver.Value(d[j])
        period = used_free_periods[j]
        regular_assignments[period] = f"{event_types[division_index]['name']} {task_label} #{j}"
    
    schedule = {}
    for period in range(1, total_periods+1):
        if fixed[period] is not None:
            schedule[period] = fixed[period]
        elif period in regular_assignments:
            schedule[period] = regular_assignments[period]
        else:
            schedule[period] = "None"
    return schedule, solver.ObjectiveValue()

##############################
# Date Range Formatting Function
##############################

def format_date_range(p, base_date, period_type, period_length):
    start = add_period(base_date, p, period_type, period_length)
    pt = period_type.lower()
    
    if pt == "minutes":
        end = start + timedelta(minutes=period_length) - timedelta(seconds=1)
        return f"{start.strftime('%H:%M:%S')} -- {end.strftime('%H:%M:%S')}"
    elif pt == "hours":
        end = start + timedelta(hours=period_length) - timedelta(seconds=1)
        return f"{start.strftime('%d/%m/%y %H:%M')} -- {end.strftime('%d/%m/%y %H:%M')}"
    elif pt in ["days", "weeks"]:
        if pt == "weeks":
            end = start + timedelta(weeks=period_length) - timedelta(seconds=1)
        else:
            end = start + timedelta(days=period_length) - timedelta(seconds=1)
        return f"{start.strftime('%d/%m/%y')} -- {end.strftime('%d/%m/%y')}"
    elif pt == "months":
        end = add_period(base_date, p+1, period_type, period_length) - timedelta(seconds=1)
        return f"{start.strftime('%B %Y')} -- {end.strftime('%B %Y')}"
    elif pt == "years":
        end = add_period(base_date, p+1, period_type, period_length) - timedelta(seconds=1)
        return f"{start.strftime('%Y')} -- {end.strftime('%Y')}"
    else:
        end = add_period(base_date, p+1, period_type, period_length) - timedelta(seconds=1)
        return f"{start} -- {end}"

##############################
# Main Scheduling Process
##############################

def schedule_events():
    # Load configuration and data from JSON.
    config = load_data("data.json")
    total_periods = config.get("total_periods", 52)
    base_date_str = config.get("base_date", "2025-03-31")
    output_filename = config.get("output_filename", "schedule.xlsx")
    sheet_name = config.get("sheet_name", "Packed")
    task_label = config.get("task_label", "event")
    penalty_params = config.get("penalty_parameters", {"consec_penalty": 500, "extra_penalty": 300, "spacing_threshold": 4})
    consec_penalty = penalty_params.get("consec_penalty", 500)
    extra_penalty = penalty_params.get("extra_penalty", 300)
    spacing_threshold = penalty_params.get("spacing_threshold", 4)
    
    period_type = config.get("period_type", "weeks")
    period_length = config.get("period_length", 1)
    event_types = config["event_types"]
    
    # Build fixed assignments: mark Reserved, Public Holiday, and Unavailable separately.
    reserved = set(config.get("reserved", []))
    public_holidays = set(config.get("public_holidays", []))
    unavailable = set(config.get("unavailable", []))
    fixed = {}
    for p in range(1, total_periods+1):
        statuses = []
        if p in reserved:
            statuses.append("Reserved")
        if p in public_holidays:
            statuses.append("Public Holiday")
        if p in unavailable:
            statuses.append("Unavailable")
        fixed[p] = ", ".join(statuses) if statuses else None

    # For event types without a final_week, set final_week to total_periods+1 and deadline to total_periods.
    for et in event_types:
        if "final_week" not in et or et["final_week"] in [None, ""]:
            et["final_week"] = total_periods + 1
            et["deadline"] = total_periods
        elif "deadline" not in et or et["deadline"] in [None, ""]:
            et["deadline"] = et["final_week"] - 1

    # Build fixed assignments for Final events.
    final_event_periods = set()
    for et in event_types:
        fp = et["final_week"]
        if fp <= total_periods:
            if fp in final_event_periods:
                print(f"Error: Period {fp} is already used for a Final event; conflict for {et['name']}.")
                exit(1)
            fixed[fp] = f"{et['name']} Final event"
            final_event_periods.add(fp)
    
    available_periods = {p for p in range(1, total_periods+1) if fixed[p] is None}
    print(f"Available periods: {sorted(available_periods)}\n")
    
    # Adaptive CP-SAT tuning: try several penalty parameter pairs.
    candidate_params = [(500, 300), (700, 400), (900, 500)]
    best_solution = None
    best_obj = float('inf')
    best_used_free = None
    for (cp_pen, ex_pen) in candidate_params:
        sol, obj, used_free = cp_schedule_packed(fixed, total_periods, event_types, task_label, consec_penalty, extra_penalty, spacing_threshold)
        var_pen = compute_variety_penalty(sol, total_periods, task_label)
        if var_pen < best_obj:
            best_solution = sol.copy()
            best_obj = var_pen
            best_used_free = used_free[:]
    print("\nPacked Schedule (CP-SAT best candidate):")
    for p in range(1, total_periods+1):
        print(f"Period {p:2d}: {best_solution[p]}")
    print(f"Variety Penalty (CP-SAT candidate): {best_obj}")
    
    # Apply simulated annealing improvement.
    sa_solution, sa_penalty = simulated_annealing_improvement(best_solution, best_used_free, total_periods, event_types, task_label)
    print("\nPacked Schedule (After Simulated Annealing):")
    for p in range(1, total_periods+1):
        print(f"Period {p:2d}: {sa_solution[p]}")
    print(f"Variety Penalty (SA): {sa_penalty}")
    
    # Apply tabu search improvement.
    tabu_solution, tabu_penalty = tabu_search_improvement(sa_solution, best_used_free, total_periods, event_types, task_label, iterations=100, tabu_tenure=5)
    print("\nPacked Schedule (After Tabu Search):")
    for p in range(1, total_periods+1):
        print(f"Period {p:2d}: {tabu_solution[p]}")
    print(f"Variety Penalty (Tabu): {tabu_penalty}")
    
    improved_solution = tabu_solution.copy()
    final_solution = improved_solution.copy()
    
    # Prepare the Excel output.
    schedule_list = [{"Period": p, "Task": final_solution[p]} for p in range(1, total_periods+1)]
    df = pd.DataFrame(schedule_list)
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
    df["Date Range"] = df["Period"].apply(lambda p: format_date_range(p, base_date, period_type, period_length))
    df = df[["Period", "Date Range", "Task"]]
    try:
        df.to_excel(output_filename, sheet_name=sheet_name, index=False)
        print(f"\nExcel file '{output_filename}' created successfully with sheet '{sheet_name}'.")
    except Exception as e:
        print("Error exporting to Excel:", e)

if __name__ == "__main__":
    random.seed(42)
    schedule_events()
