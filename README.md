# SET3065 — Intelligent Electrical Power Grids

Starter repository for practical coding work in the SET3065 course.

## Quick start

1. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Run the example simulation:

   ```bash
   python3 examples/frequency_control_example.py
   ```

3. (Optional) Export the simulation trace to CSV:

   ```bash
   python3 examples/frequency_control_example.py --csv outputs/frequency_trace.csv
   ```

## What the example does

The example models one control area with a simple governor-frequency response:

- A load disturbance is applied at a configured time.
- Generation automatically adjusts according to droop control.
- Grid frequency deviates from nominal and then settles.

This is intentionally lightweight and dependency-free so you can run it immediately
and extend it for your coursework.

## Repository layout

```text
examples/
  frequency_control_example.py   # runnable starter model
README.md
```