# Task 2: Probabilistic Power Flow Analysis - Explanation

## Overview

This task performs **Probabilistic Power Flow (PPF)** analysis on the IEEE 9-bus system with a wind power plant. It evaluates how **wind plant location** (Bus 5, Bus 7, or Bus 9 baseline) and **reactive power control strategy** (unity power factor, 0.95 overexcited, 0.95 underexcited) affect voltage profiles, line loadings, and losses under uncertainty in load and wind power.

## Key Concepts

### Probabilistic Power Flow (PPF)

PPF extends deterministic power flow by treating inputs as random variables:

- **Load demand**: Random variation (normal distribution) around nominal values.
- **Wind power**: Derived from random wind speed (Weibull distribution) via a power curve.

For each random sample, a power flow (here, **optimal power flow**, OPF) is solved. Results are collected across many samples to obtain **statistics** (min, max, mean, std) of voltages, line loadings, and losses.

### Monte Carlo Method

- **N = 100** samples are drawn (configurable; use small N for quick tests).
- The **same** wind speed and load sequences are used for all scenarios (fixed random seed) so comparisons are fair.
- Not every sample leads to a convergent OPF; only converged cases are stored. Convergence rate per scenario is reported.

### Wind Power Model

Wind speed \(v\) is Weibull-distributed (shape \(k = 2.02\), scale \(\lambda = 11\) m/s). Active power \(P\) [MW] is:

- \(v < v_{\text{in}}\) (3 m/s): \(P = 0\)
- \(v_{\text{in}} \leq v < v_{\text{rated}}\) (12 m/s): \(P = P_{\text{wpp}} \frac{v^3 - v_{\text{in}}^3}{v_{\text{rated}}^3 - v_{\text{in}}^3}\)
- \(v_{\text{rated}} \leq v < v_{\text{out}}\) (20 m/s): \(P = P_{\text{wpp}} = 180\) MW
- \(v \geq v_{\text{out}}\): \(P = 0\)

### Reactive Power Modes

The wind generator is modelled as a static generator with fixed P and Q per sample:

| Mode            | Power factor | Q behaviour              | Effect                    |
|-----------------|-------------|---------------------------|---------------------------|
| **Unity**       | 1.0         | \(Q = 0\)                 | No reactive exchange      |
| **Overexcited** | 0.95        | \(Q > 0\) (supplies Q)    | Voltage support           |
| **Underexcited**| 0.95        | \(Q < 0\) (absorbs Q)     | Reduces voltage           |

For PF 0.95: \(Q = \pm P \tan(\arccos(0.95))\).

## Code Structure

### Main Components

1. **`calculate_wind_power(wind_speed)`**  
   Implements the wind power curve; returns active power in MW.

2. **`configure_wind_reactive_power(net, p_wind, pf, mode)`**  
   Sets the wind static generator’s \(P\), \(Q\), and min/max limits according to power factor and mode (unity / overexcited / underexcited).

3. **`initialize_results_storage()`**  
   Builds a dictionary of result containers (voltages, loadings, losses, converged count) for each scenario.

4. **`calculate_statistics(results)`**  
   Converts stored lists to arrays and computes per-bus/per-line min, max, mean, and std for voltages, loadings, and losses.

5. **`run_ppf_analysis()`**  
   - Loads the base network and load parameters (means, PQ ratio).  
   - Generates one set of wind speeds (Weibull) and corresponding wind powers for all scenarios.  
   - For each scenario: re-loads the network, sets the wind connection bus (transformer 3’s HV bus), then for each of the N samples: draws random loads, sets wind P and Q, runs PF + OPF; on convergence, appends voltages, line loadings, and line losses.  
   - Returns the results dictionary.

6. **`plot_voltage_variability(results)`**  
   For three selected scenarios (e.g. Baseline, Wind@Bus7, Wind@Bus5 at 0.95 overexcited), plots bus voltage min–max band and mean, with ±5% limits.

7. **`plot_loading_variability(results)`**  
   Same idea for line loading (%); includes 100% limit.

8. **`plot_loss_variability(results)`**  
   Same idea for line losses (MW).

9. **`generate_summary_table(results)`**  
   Prints a table: scenario, convergence %, global V_min/V_max, max line loading %, total loss (mean).

## Scenario Definitions

Nine scenarios combine:

- **3 locations**: Baseline (Bus 9), Bus 7, Bus 5 (pp indices 8, 6, 4).  
  The wind farm is connected to the main grid via **transformer 3**. Its **LV side** is always **bus 10** (pp index 9), internal to the wind farm. The **HV side** is the grid connection point; we set `net.trafo.at[3, 'hv_bus'] = scenario['trafo_hv_bus']` to move the wind plant to Bus 5, 7, or 9 (indices 4, 6, 8).

- **3 reactive modes**: Unit PF, 0.95 overexcited, 0.95 underexcited.

So: Baseline_UnitPF, Baseline_PF095_OE, Baseline_PF095_UE, Wind@Bus7_*, Wind@Bus5_*.

## Network and Data Flow

- **Base network**: `ieee9-wind.xlsx` (loaded from the script directory via `THIS_DIR`).
- **Load uncertainty**: For each sample, load active power is drawn from a normal distribution (mean = nominal, std = fixed per load); reactive power keeps the nominal P–Q ratio.
- **Wind**: One Weibull-driven wind power sequence; reactive power is set by the scenario’s PF and mode.
- **OPF**: `pp.runpp()` then `pp.runopp(init='pf', ...)`; failed cases are skipped (no result stored).

## Results Interpretation

- **Convergence**: With N = 100, not all samples need converge; the script reports converged/total and convergence %. Low convergence for small N (e.g. N = 10) is expected; use N = 100 for the full analysis.
- **Voltage plots**: Tighter min–max bands and mean closer to 1.0 pu indicate more stable voltages; check violations of the ±5% band.
- **Loading plots**: High mean or max loading, or large spread, indicates more stressed or variable line usage.
- **Loss plots**: Show how losses vary with scenario; total loss in the summary table is the sum of mean line losses.
- **Summary table**: Use it to compare scenarios (convergence, worst voltage, worst loading, total loss).

## Key Insights

1. **Wind location** changes power flow paths and thus voltages, loadings, and losses.
2. **Overexcited (Q > 0)** typically improves voltage profile; **underexcited (Q < 0)** can lower voltages.
3. **Unity PF** is a neutral reference; comparing to 0.95 OE/UE shows the value of reactive control.
4. **Monte Carlo** gives bands (min–max) and variability (std), not only a single snapshot.

## How to Use the Code

1. **Run the full analysis** (from the A2 directory):
   ```bash
   python task2_ppf_analysis.py
   ```

2. **Quick test**: In the script, set `N = 10` (or 5); expect low or zero convergence. Set `N = 100` for the intended analysis (~5–10 minutes).

3. **Modify scenarios**: Edit the `scenarios` list to add/remove locations or PF modes.

4. **Change plots**: Adjust `compare_scenarios` in each plot function to choose which scenarios appear in the 3-panel figures.

5. **Reuse results**: Load `task2_ppf_results.pkl` (e.g. with `pickle.load`) to recompute statistics or replot without re-running the PPF.

## Files Generated

- **`task2_voltage_variability.png`**: Bus voltage min/max/mean for three scenarios (e.g. Baseline, Wind@Bus7, Wind@Bus5 at PF 0.95 OE).
- **`task2_loading_variability.png`**: Line loading (%) min/max/mean for the same three scenarios.
- **`task2_loss_variability.png`**: Line loss (MW) min/max/mean for the same three scenarios.
- **`task2_ppf_results.pkl`**: Full results dictionary (voltages, loadings, losses, converged counts, and after `calculate_statistics()` the `*_stats` dicts) for post-processing and custom plots.
- **Console**: Progress per scenario, convergence counts, and the summary table.
