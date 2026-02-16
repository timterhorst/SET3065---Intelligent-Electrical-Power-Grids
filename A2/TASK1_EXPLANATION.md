# Task 1: Economic Dispatch Analysis - Explanation and Results

## Overview
This task analyzes the impact of wind power plant location on economic dispatch (ED) and optimal power flow (OPF) results in the IEEE 9-bus system.

## Key Concepts

### Economic Dispatch (ED)
Economic Dispatch is the process of determining the optimal output of generators to meet load demand at minimum cost, while respecting:
- Power balance constraints (generation = load + losses)
- Generator capacity limits
- Transmission line capacity limits
- Voltage limits

### Optimal Power Flow (OPF)
OPF extends ED by also considering:
- AC power flow equations (non-linear)
- Voltage magnitude constraints
- Reactive power balance
- Line loading limits

In this analysis, we use `pp.runopp()` which solves the AC OPF problem.

## Code Structure

### Main Components

1. **`run_opf_analysis()`**: Runs OPF for a given network configuration
   - Scales load demand by multiplier (1.0 = baseline, 1.5 = +50%, 2.0 = +100%)
   - Executes OPF optimization
   - Extracts and returns results (cost, voltages, line loading)
   - Handles convergence failures gracefully

2. **`change_wind_location()`**: Moves wind power plant subsystem connection
   - Modifies transformer 3's high-voltage bus connection (the transformer connecting wind subsystem to main grid)
   - Transformer 3 connects the main 230 kV grid to Bus 9 (110 kV) of the wind subsystem
   - Changes transformer 3's `hv_bus` from 4 to target bus (5 or 7)
   - **Important**: Transformer 3's voltage rating (`vn_hv_kv`) remains 230 kV since buses 4, 5, and 7 are all 230 kV
   - Transformer 4 (internal to wind subsystem) and the line (Bus 9 → Bus 10) remain unchanged

3. **`plot_comparison()`**: Creates visualization plots
   - Voltage magnitude comparison
   - Line loading comparison
   - Total cost comparison
   - Voltage deviation analysis

4. **`print_results_summary()`**: Displays results in tabular format

## Network Structure

### Baseline Configuration
- **Wind subsystem**: Consists of buses 9, 10, 11 (110 kV, 110 kV, 33 kV)
- **Wind generator**: At bus 11 (33 kV)
- **Connection to main grid**: Via Transformer 3 connecting Bus 4 (230 kV) → Bus 9 (110 kV)
- **Internal wind subsystem**: 
  - Line: Bus 9 → Bus 10 (both 110 kV)
  - Transformer 4: Bus 10 (110 kV) → Bus 11 (33 kV)
- **Loads**: At buses 4, 6, 8 (total: 315 MW, 115 MVAr)
- **Generators**: At buses 1, 2 (conventional generators)

### Modified Configurations
- **Wind@Bus7**: Transformer 3 connects bus 7 (230 kV) → Bus 9 (110 kV)
  - The entire wind subsystem (buses 9, 10, 11) is now connected to bus 7 instead of bus 4
- **Wind@Bus5**: Transformer 3 connects bus 5 (230 kV) → Bus 9 (110 kV)
  - The entire wind subsystem (buses 9, 10, 11) is now connected to bus 5 instead of bus 4

## Results Analysis

### Baseline Case (Wind@Bus4)
- **Total Cost**: $2,082.99
- **Voltage Range**: 0.9830 - 1.0500 pu
- **Max Line Loading**: 90.65%
- **Note**: Wind subsystem connected to Bus 4 via Transformer 3

### Wind@Bus7 (100% Demand)
- **Total Cost**: $2,136.47 (+2.57% vs baseline)
- **Voltage Range**: 0.9954 - 1.0500 pu (slightly improved minimum voltage)
- **Max Line Loading**: 89.10% (slight reduction)
- **Analysis**: Slightly better voltage profile, similar line loading, but higher cost

### Wind@Bus5 (100% Demand)
- **Total Cost**: $2,141.64 (+2.82% vs baseline)
- **Voltage Range**: 1.0031 - 1.0500 pu (best minimum voltage)
- **Max Line Loading**: 88.32% (slight reduction)
- **Analysis**: Best voltage profile, similar line loading, but highest cost

### Impact of Increased Demand (+50%)
- **Cost**: Increases dramatically (~141% increase) due to need for more expensive generation
- **Voltages**: Minimum voltage drops (0.95-0.96 pu range)
- **Line Loading**: Increases but still manageable
- **Note**: 200% demand cases did not converge (system capacity exceeded)

## Key Insights

1. **Wind Location Matters**: Different locations affect:
   - Power flow patterns
   - Voltage profiles
   - Line loadings
   - Total generation cost

2. **Bus 7 Connection**: Provides best line loading reduction but slightly higher cost

3. **Bus 5 Connection**: Provides best voltage profile

4. **Demand Increase**: 
   - 50% increase is manageable but costly
   - 100% increase exceeds system capacity (no convergence)

5. **Voltage vs. Cost Trade-off**: Better voltage profiles may come with slightly higher costs

## How to Use the Code

1. **Run the analysis**:
   ```python
   python task1_economic_dispatch.py
   ```

2. **Modify parameters**:
   - Change `wind_locations` list to test different buses
   - Adjust `demand_mult` values to test different load levels
   - Modify plot settings in `plot_comparison()`

3. **Interpret results**:
   - Check convergence status
   - Compare costs, voltages, and line loadings
   - Analyze trade-offs between different configurations

## Files Generated

- `task1_comparison_analysis.png`: Comprehensive comparison plots
- Console output: Detailed results summary and analysis
