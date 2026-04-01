# SET3065 — Intelligent Electrical Power Grids

Coursework repository for practical coding work in SET3065.

## Quick start

1. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the first assignment example (A1):

   ```bash
   python3 A1/case4gs_simple.py
   ```

## Assignment files and run commands

### A1

```bash
python3 A1/case4gs_simple.py
```

### A2

```bash
python3 A2/opf_ieee9_wind.py
python3 A2/prob_opf_ieee9_wind.py
```

Notes:
- `ieee9-wind.xlsx` is loaded using the script directory path.
- Plot images are saved into `A2/` as PNG files.

### A3 / A4 notebooks

Install Jupyter if needed:

```bash
pip install jupyterlab
```

Launch notebooks:

```bash
jupyter lab
```

Open:
- `A3/A6.ipynb`
- `A4/Assignment_generativeAI_system_planning.ipynb`
- `A4/Answer_of_Assignment_generativeAI_system_planning.ipynb`

## Optional starter simulation (custom example)

Run the lightweight standalone frequency-control demo:

   ```bash
   python3 examples/frequency_control_example.py
   ```

Export simulation trace to CSV:

   ```bash
   python3 examples/frequency_control_example.py --csv outputs/frequency_trace.csv
   ```

## Repository layout

```text
A1/
A2/
A3/
A4/
examples/
  frequency_control_example.py
requirements.txt
README.md
```