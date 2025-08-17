I wrote this code as a [redacted] at [redacted]. Beyond the application I was using it for which was [redacted] scheduling, it has some potential for future use, at least for my personal purposes.
# Constraint-Bound Scheduling Framework

## Overview

This project is a fully customizable, constraint-bound scheduling framework built from scratch using Google's OR‑Tools CP‑SAT solver along with advanced metaheuristic improvement techniques. Initially developed to address a specific scheduling challenge, the framework has evolved into a general-purpose tool that can be applied to many domains—including manufacturing, employee timetabling, classroom scheduling, resource allocation, and more.

The system optimizes a schedule subject to both hard constraints (such as deadlines and unavailable time slots) and soft constraints (minimizing repetitive assignments). It uses a hybrid optimization approach that combines a CP‑SAT model with simulated annealing, tabu search, and optional reoptimization phases.

## Key Features

- **Fully Customizable Timeframes:**  
  The scheduling period is fully configurable. Users can define the scheduling horizon in any time unit (minutes, hours, days, weeks, months, or years) with a customizable base date and period length.

- **General-Purpose & Constraint-Bound:**  
  The framework is not tied to a specific application. It’s designed to schedule any task type. Users can define multiple task types (or event types) with specific requirements (task counts, deadlines, etc.) and non‑available periods.

- **Advanced Optimization Techniques:**  
  - **CP‑SAT Scheduling:** Uses a CP‑SAT model to generate an initial “packed” schedule by assigning tasks to the earliest available periods while enforcing hard constraints.
  - **Metaheuristic Improvements:** Implements simulated annealing and tabu search to further refine the schedule, reducing consecutive or closely spaced repetitions.
  - **Hybrid Re‑optimization (Optional):** Re-runs CP‑SAT with an objective bound to further refine the solution within a promising search space.

- **Easy Configuration via JSON:**  
  All parameters—including scheduling horizon, base date, period type/length, output filenames, penalty parameters, and task definitions—are provided in a JSON file. This enables non-technical users to tailor the framework to their needs without modifying the code.

- **Robust Output:**  
  The final schedule is exported to an Excel file with formatted date ranges, making it easy to review and integrate with existing planning systems.

## Applications

This scheduling framework can be applied to a wide range of real-world problems:
- **Manufacturing Scheduling:** Optimize production runs while ensuring that the same machine or production line is not overused consecutively.
- **Employee Timetabling:** Create shift schedules that avoid consecutive shifts for the same employee, ensuring fair distribution of work.
- **Classroom and Exam Timetabling:** Schedule classes or exams to minimize the risk of overlapping subjects or overburdening instructors.
- **Resource Allocation:** Plan maintenance tasks, project work, or other resource-bound activities with strict constraints on availability and workload.

## Detailed Algorithm Explanation

### 1. Data Loading and Configuration

The system reads all configuration data from a JSON file (by default `data.json`). The JSON file contains:
- **Timeframe Configuration:**  
  - `total_weeks` (or more generally, `total_periods`): The number of scheduling periods.
  - `base_date`: The start date from which scheduling periods are calculated.
  - `period_type`: The unit of time (e.g., minutes, hours, days, weeks, months, years).
  - `period_length`: The length of each period.
- **Output Settings:**  
  - `output_filename`: Name of the Excel file to generate.
  - `sheet_name`: Excel sheet name.
  - `task_label`: A string label for the tasks (e.g., "event", "task", "build).
- **Penalty Parameters:**  
  A set of parameters (`consec_penalty`, `extra_penalty`, `spacing_threshold`) that control the penalties applied to consecutive or near‑consecutive identical tasks.
- **Task (Event) Types:**  
  An array of objects defining each task type, including:
  - `name`: The name of the task type.
  - `regular_count`: The number of times this task must be scheduled.
  - `final_period`: The last period in which the task must occur (if applicable).
  - `deadline`: An optional field; if omitted, the deadline is inferred.
- **Non-Available Periods:**  
  Lists for `public_holidays`, `reserved`, and `unavailable` periods, which are merged to create a set of periods during which no tasks may be scheduled.

### 2. Preprocessing

- **Non-Available Periods:**  
  The system computes the union of all non‑available periods and excludes these from the scheduling horizon.
  
- **Task Deadline Handling:**  
  For each task type without a specified `final_week`, the system sets a default final period of `total_periods + 1` (effectively removing a deadline within the horizon) and assigns the deadline to the end of the horizon.

- **Fixed Assignments:**  
  Fixed periods (non‑available times and final events for tasks with deadlines) are marked in the schedule before the optimization phase.

### 3. CP‑SAT Scheduling Model

- **Free Period Identification:**  
  The scheduler identifies all free periods (those not fixed) and uses the first _R_ free periods (where _R_ is the total number of tasks to be scheduled) to build a “packed” schedule.

- **Decision Variables & Domains:**  
  For each free period slot, a decision variable is created to assign an eligible task type. Eligibility is determined by whether the period falls before the task type’s final period.

- **Constraint Enforcement:**  
  The model enforces that:
  - Each task type is scheduled exactly the number of times specified (`regular_count`).
  - Tasks are only assigned to periods that occur before their final period.
  
- **Objective Function:**  
  The model minimizes a penalty function that adds:
  - A high penalty for consecutive assignments of the same task type.
  - An extra penalty for assignments that are too close together (within a configurable spacing threshold).

### 4. Metaheuristic Enhancements

After generating an initial solution with CP‑SAT, the framework applies local search techniques to improve variety:
- **Simulated Annealing:**  
  Randomly swaps task assignments between free periods. The swap is accepted if it reduces the variety penalty or with a probability that decreases over time, allowing the solution to escape local optima.
- **Tabu Search:**  
  Explores a neighborhood of solutions by making pairwise swaps while using a tabu list to avoid cycling back to recently visited solutions.
- **Hybrid Re‑optimization (Optional):**  
  The CP‑SAT model can be re-run with an additional bound constraint (using the best variety penalty from the metaheuristics) to refine the solution further.

### 5. Output Generation

- **Date Range Calculation:**  
  The system converts period numbers into human‑readable date ranges using the configured base date, period type, and period length.
- **Excel Export:**  
  The final schedule is exported to an Excel workbook with the specified file name and sheet name, including columns for period numbers, date ranges, and scheduled tasks.

## Iterative Development Process

I began with a basic dynamic programmic greedy algorithm to assign tasks to available periods while meeting hard constraints. Recognizing the need for increased efficiency, I implemented CP-Sat and messed around with the weights. As real-world tests revealed further opportunities for improvement, I integrated simulated annealing and tabu search to refine the solution. Finally, I made the entire system configurable via JSON so that it could be easily adapted for any scheduling problem without modifying and combing through the code.

## Getting Started

1. **Prepare Your Data:**  
   Create a `data.json` file with your scheduling parameters. Use the provided sample file as a template and modify it to match your application requirements (e.g., change `period_type` to "hours" for hourly scheduling).

2. **Run the Scheduler:**  
   Execute the script by running `python schedule.py`. The system will load your configuration, generate the optimal schedule, and export the result to an Excel file.

3. **Review & Adapt:**  
   Open the Excel file to review your schedule. Customize the JSON file as needed to experiment with different configurations and constraints.

## Conclusion

This Constraint-Bound Scheduling Framework is a robust, versatile tool designed to solve complex scheduling problems under various constraints. Its high level of customizability—enabled by a comprehensive JSON configuration—allows it to be adapted for manufacturing, timetabling, resource allocation, and more. With its integration of CP‑SAT and advanced metaheuristic techniques, the framework delivers high-quality, diverse schedules that can be fine-tuned to your specific needs.
