"""
Task 1: Economic Dispatch Analysis for IEEE 9-bus System with Wind Power Plant

This script performs economic dispatch analysis by:
1. Running baseline case (wind connected via transformer to bus 9, pp index 8)
2. Moving wind power plant to bus 7 (changing transformer connection)
3. Moving wind power plant to bus 5 (changing transformer connection)
4. For each location, testing with 50% and 100% increased demand
5. Comparing voltage magnitudes and branch loading

Understanding:
- Economic Dispatch (ED) optimizes generator outputs to minimize total cost
- Optimal Power Flow (OPF) solves ED while respecting power flow constraints
- Wind location affects power flow patterns, voltages, and line loadings
- Increased demand requires more generation, affecting system operation
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp

pd.options.display.max_columns = None

THIS_DIR = Path(__file__).resolve().parent

# ============================================================================
# STEP 1: Define helper functions
# ============================================================================

def run_opf_analysis(net, case_name, demand_multiplier=1.0):
    """
    Run optimal power flow analysis for a given network configuration.
    
    Parameters:
    -----------
    net : pandapower network
        The power system network
    case_name : str
        Name of the case for identification
    demand_multiplier : float
        Multiplier for load demand (1.0 = baseline, 1.5 = 50% increase, 2.0 = 100% increase)
    
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    # Store original loads
    original_p_mw = net.load.p_mw.copy()
    original_q_mvar = net.load.q_mvar.copy()
    
    # Scale loads if multiplier is not 1.0
    if demand_multiplier != 1.0:
        net.load.p_mw = original_p_mw * demand_multiplier
        net.load.q_mvar = original_q_mvar * demand_multiplier
    
    try:
        # Run optimal power flow
        pp.runopp(net, init='pf', verbose=False, numba=False)
        
        # Extract results
        results = {
            'case_name': case_name,
            'demand_multiplier': demand_multiplier,
            'total_cost': net.res_cost,
            'voltages': net.res_bus.vm_pu.copy(),
            'voltage_angles': net.res_bus.va_degree.copy(),
            'line_loading': net.res_line.loading_percent.copy(),
            'branch_losses': net.res_line.pl_mw.copy(),  # Active power losses [MW]
            'branch_losses_reactive': net.res_line.ql_mvar.copy(),  # Reactive power losses [MVAr]
            'total_losses': net.res_line.pl_mw.sum(),  # Total system active losses
            'converged': True
        }
        
        # Restore original loads
        net.load.p_mw = original_p_mw
        net.load.q_mvar = original_q_mvar
        
        return results
        
    except Exception as e:
        # Restore original loads even if OPF fails
        net.load.p_mw = original_p_mw
        net.load.q_mvar = original_q_mvar
        print(f"  Warning: OPF did not converge for {case_name} with {demand_multiplier*100:.0f}% demand")
        # Note: Non-converged cases are excluded from results summary and plots
        # This indicates the system cannot handle this demand level
        return {
            'case_name': case_name,
            'demand_multiplier': demand_multiplier,
            'converged': False
        }


def change_wind_location(net, target_bus):
    """
    Change the wind power plant subsystem connection to the main grid.
    
    Understanding:
    - The wind power plant subsystem consists of buses 9, 10, 11 (110 kV, 110 kV, 33 kV)
    - Transformer 3 connects the main grid (230 kV) to the wind subsystem (lv_bus 9 at 110 kV)
    - Baseline: Transformer 3 hv_bus = 8 (diagram bus 9). Other cases: hv_bus = 4 or 6 (diagram buses 5 or 7).
    - To move wind subsystem: Change Transformer 3's hv_bus to target_bus (pp index 4 = bus 5, pp index 6 = bus 7)
    - Transformer 4 (Bus 10 -> Bus 11) and the line (Bus 9 -> Bus 10) remain unchanged
      as they are internal to the wind subsystem
    
    Parameters:
    -----------
    net : pandapower network
        The power system network
    target_bus : int
        Pandapower (0-based) bus index in the main 230 kV grid.
        Use 4 for diagram bus 5, 6 for diagram bus 7.
    """
    # Transformer 3 connects the wind subsystem (bus 9) to the main grid
    # Change its high-voltage bus to connect to different location in main grid
    trafo_idx = 3  # Transformer connecting wind subsystem to main grid
    
    # Get the voltage level of the target bus (should be 230 kV for buses 5 or 7)
    target_voltage = net.bus.loc[target_bus, 'vn_kv']
    
    # Verify target bus is at correct voltage level
    if target_voltage != 230.0:
        print(f"  Warning: Target bus {target_bus} voltage ({target_voltage} kV) may not match expected 230 kV")
    
    # Update transformer connection
    # Note: vn_hv_kv should remain 230 kV since target_bus is 230 kV
    net.trafo.loc[trafo_idx, 'hv_bus'] = target_bus
    
    print(f"  Changed transformer {trafo_idx} connection: Wind subsystem (Bus 9) now connected to bus {target_bus} ({target_voltage} kV)")


def plot_comparison(results_list, output_dir):
    """
    Create comparison plots for voltage magnitudes, line loading, and power losses.
    
    Parameters:
    -----------
    results_list : list of dict
        List of result dictionaries from run_opf_analysis
    output_dir : Path
        Directory to save plots
    """
    # Filter converged results
    converged_results = [r for r in results_list if r.get('converged', False)]
    
    if not converged_results:
        print("No converged results to plot")
        return
    
    # Define marker mapping based on scenario
    def get_marker(case_name, demand_multiplier):
        """Return marker style based on case name and demand multiplier."""
        if case_name == "Baseline" and demand_multiplier == 1.0:
            return 'o'  # circle
        elif case_name == "Wind@Bus7" and demand_multiplier == 1.0:
            return 's'  # square
        elif case_name == "Wind@Bus7" and demand_multiplier == 1.5:
            return '^'  # triangle up
        elif case_name == "Wind@Bus5" and demand_multiplier == 1.0:
            return 'D'  # diamond
        elif case_name == "Wind@Bus5" and demand_multiplier == 1.5:
            return 'v'  # triangle down
        else:
            return 'o'  # default fallback
    
    # Create comparison plots - 2x3 grid layout (Option B: Logical grouping)
    # Row 1: Voltage metrics | Row 2: Power flow metrics
    # Col 1-2: Detailed metrics | Col 3: Summary metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Diagram bus number = 1-based (as on single-line diagram); pandapower uses 0-based indices.
    def pp_bus_to_diagram(pp_bus_index):
        return pp_bus_index + 1
    
    # ========================================================================
    # ROW 1: Voltage Metrics + Cost Summary
    # ========================================================================
    
    # Plot 1: Voltage magnitudes comparison (Row 1, Col 1) — x-axis: diagram bus number
    ax1 = axes[0, 0]
    for result in converged_results:
        label = f"{result['case_name']} ({result['demand_multiplier']*100:.0f}% demand)"
        marker = get_marker(result['case_name'], result['demand_multiplier'])
        bus_numbers = [pp_bus_to_diagram(i) for i in result['voltages'].index]
        ax1.plot(bus_numbers, result['voltages'].values,
                marker=marker, label=label, linewidth=2.0, markersize=5)
    # Add voltage limit lines (match pandapower OPF constraints: ±5%)
    ax1.axhline(y=1.05, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Upper limit (1.05 pu)')
    ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Lower limit (0.95 pu)')
    ax1.set_xlabel('Bus number (diagram)', fontsize=12)
    ax1.set_ylabel('Voltage Magnitude (pu)', fontsize=12)
    ax1.set_title('Voltage Magnitude Comparison', fontsize=14, fontweight='bold')
    ax1.set_axisbelow(True)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(fontsize=8, loc='best')
    ax1.set_ylim([0.90, 1.10])  # Extended range to show ±5% limits
    
    # Plot 2: Voltage deviation from nominal (1.0 pu) (Row 1, Col 2) — x-axis: diagram bus number
    ax2 = axes[0, 1]
    for result in converged_results:
        voltage_deviation = (result['voltages'] - 1.0) * 100  # Convert to percentage
        label = f"{result['case_name']} ({result['demand_multiplier']*100:.0f}% demand)"
        marker = get_marker(result['case_name'], result['demand_multiplier'])
        bus_numbers = [pp_bus_to_diagram(i) for i in voltage_deviation.index]
        ax2.plot(bus_numbers, voltage_deviation.values,
                marker=marker, label=label, linewidth=2.0, markersize=5)
    # Add voltage limit lines (converted to percentage deviation; match OPF ±5%)
    ax2.axhline(y=5.0, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Upper limit (+5%)')
    ax2.axhline(y=-5.0, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Lower limit (-5%)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    ax2.set_xlabel('Bus number (diagram)', fontsize=12)
    ax2.set_ylabel('Voltage Deviation from 1.0 pu (%)', fontsize=12)
    ax2.set_title('Voltage Deviation Analysis', fontsize=14, fontweight='bold')
    ax2.set_axisbelow(True)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.legend(fontsize=8, loc='best')
    
    # Plot 3: Total cost comparison (Row 1, Col 3)
    ax3 = axes[0, 2]
    bar_labels = [f"{r['case_name']} ({r['demand_multiplier']*100:.0f}%)" for r in converged_results]
    costs = [r['total_cost'] for r in converged_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(bar_labels)))
    bars = ax3.bar(range(len(bar_labels)), costs, color=colors)
    ax3.set_xticks(range(len(bar_labels)))
    ax3.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax3.set_ylabel('Total Cost ($)', fontsize=12)
    ax3.set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
    ax3.set_axisbelow(True)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.0f}', ha='center', va='bottom', fontsize=9)
    
    # ========================================================================
    # ROW 2: Power Flow Metrics + Losses Summary
    # ========================================================================
    
    # Plot 4: Line loading comparison (Row 2, Col 1)
    ax4 = axes[1, 0]
    for result in converged_results:
        label = f"{result['case_name']} ({result['demand_multiplier']*100:.0f}% demand)"
        marker = get_marker(result['case_name'], result['demand_multiplier'])
        ax4.plot(result['line_loading'].index, result['line_loading'].values,
                marker=marker, label=label, linewidth=2.0, markersize=5)
    ax4.axhline(y=100, color='r', linestyle='--', linewidth=1.0, alpha=0.7, label='100% Loading Limit')
    ax4.set_xlabel('Line Index', fontsize=12)
    ax4.set_ylabel('Line Loading (%)', fontsize=12)
    ax4.set_title('Line Loading Comparison', fontsize=14, fontweight='bold')
    ax4.set_axisbelow(True)
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax4.legend(fontsize=8, loc='best')
    
    # Plot 5: Branch Active Power Losses (Row 2, Col 2)
    ax5 = axes[1, 1]
    for result in converged_results:
        label = f"{result['case_name']} ({result['demand_multiplier']*100:.0f}% demand)"
        marker = get_marker(result['case_name'], result['demand_multiplier'])
        ax5.plot(result['branch_losses'].index, result['branch_losses'].values,
                marker=marker, label=label, linewidth=2.0, markersize=5)
    ax5.set_xlabel('Line Index', fontsize=12)
    ax5.set_ylabel('Active Power Losses [MW]', fontsize=12)
    ax5.set_title('Branch Active Power Losses Comparison', fontsize=14, fontweight='bold')
    ax5.set_axisbelow(True)
    ax5.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax5.legend(fontsize=8, loc='best')
    
    # Plot 6: Total System Losses Comparison (Row 2, Col 3)
    ax6 = axes[1, 2]
    total_losses = [r['total_losses'] for r in converged_results]
    colors_losses = plt.cm.plasma(np.linspace(0, 1, len(bar_labels)))
    bars_losses = ax6.bar(range(len(bar_labels)), total_losses, color=colors_losses)
    ax6.set_xticks(range(len(bar_labels)))
    ax6.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax6.set_ylabel('Total System Losses [MW]', fontsize=12)
    ax6.set_title('Total Active Power Losses', fontsize=14, fontweight='bold')
    ax6.set_axisbelow(True)
    ax6.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    # Add value labels on bars
    for bar, loss in zip(bars_losses, total_losses):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / "task1_comparison_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")


def print_results_summary(results_list):
    """
    Print a summary table of all results.
    """
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    converged_results = [r for r in results_list if r.get('converged', False)]
    
    if not converged_results:
        print("No converged results to display")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in converged_results:
        # Find maximum branch loss location
        max_loss_idx = result['branch_losses'].idxmax()
        max_loss_value = result['branch_losses'].max()
        
        summary_data.append({
            'Case': result['case_name'],
            'Demand Multiplier': f"{result['demand_multiplier']*100:.0f}%",
            'Total Cost ($)': f"{result['total_cost']:.2f}",
            'Min Voltage (pu)': f"{result['voltages'].min():.4f}",
            'Max Voltage (pu)': f"{result['voltages'].max():.4f}",
            'Avg Voltage (pu)': f"{result['voltages'].mean():.4f}",
            'Max Line Loading (%)': f"{result['line_loading'].max():.2f}",
            'Avg Line Loading (%)': f"{result['line_loading'].mean():.2f}",
            'Total Losses (MW)': f"{result['total_losses']:.3f}",
            'Max Branch Loss (MW)': f"{max_loss_value:.3f}",
            'Max Loss Line': f"Line {max_loss_idx}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80)


# ============================================================================
# STEP 2: Main analysis
# ============================================================================

def main():
    print("="*80)
    print("TASK 1: Economic Dispatch Analysis - Wind Power Plant Location Study")
    print("="*80)
    
    all_results = []
    
    # Load baseline network and set wind connection to bus 9 (pp index 8)
    print("\n[1] Loading baseline network...")
    net_baseline = pp.from_excel(THIS_DIR / "ieee9-wind.xlsx")
    # Verify bus voltage limits used by OPF (expected: min_vm_pu=0.95, max_vm_pu=1.05)
    print("Bus voltage limits (OPF constraints):")
    print(net_baseline.bus[['min_vm_pu', 'max_vm_pu']].drop_duplicates())
    BASELINE_PP_BUS = 8  # diagram bus 9
    net_baseline.trafo.loc[3, 'hv_bus'] = BASELINE_PP_BUS
    print(f"  Baseline: Wind subsystem connected via transformer 3 to pp bus {BASELINE_PP_BUS} (diagram bus 9)")
    
    # Run baseline case at 100%, 150%, and 200% demand
    print("\n[2] Running baseline case (Wind@Bus9)...")
    for demand_mult in [1.0, 1.5, 2.0]:
        baseline_result = run_opf_analysis(net_baseline, "Baseline", demand_multiplier=demand_mult)
        all_results.append(baseline_result)
        if baseline_result['converged']:
            print(f"  ✓ Baseline with {demand_mult*100:.0f}% demand: Cost = ${baseline_result['total_cost']:.2f}")
    
    # Test different wind locations
    # Assignment uses diagram bus numbers (1-based); pandapower uses 0-based indices.
    # Diagram bus 5 -> pp index 4; diagram bus 7 -> pp index 6.
    wind_locations = [
        (6, 7),   # (pp_bus_index, diagram_bus_number) for Wind@Bus7
        (4, 5),   # (pp_bus_index, diagram_bus_number) for Wind@Bus5
    ]
    
    for pp_bus_idx, diagram_bus in wind_locations:
        print(f"\n[3] Testing wind power plant at bus {diagram_bus} (pp index {pp_bus_idx})...")
        
        # Load fresh network for each location
        net = pp.from_excel(THIS_DIR / "ieee9-wind.xlsx")
        
        # Change wind location (pass pandapower bus index)
        change_wind_location(net, pp_bus_idx)
        
        # Test with different demand levels
        # Note: Wind@Bus5 at 150% demand may not converge (OPF not solvable).
        # 200% demand (2.0 multiplier) typically does not converge for any location.
        for demand_mult in [1.0, 1.5, 2.0]:  # Baseline, +50%, +100%
            case_name = f"Wind@Bus{diagram_bus}"
            result = run_opf_analysis(net, case_name, demand_multiplier=demand_mult)
            all_results.append(result)
            
            if result['converged']:
                print(f"  ✓ {case_name} with {demand_mult*100:.0f}% demand: Cost = ${result['total_cost']:.2f}")
    
    # Print summary
    print_results_summary(all_results)
    
    # Create comparison plots
    print("\n[4] Generating comparison plots...")
    plot_comparison(all_results, THIS_DIR)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Additional analysis: Compare to baseline
    print("\n[5] Comparison with baseline:")
    baseline = next((r for r in all_results if r['case_name'] == 'Baseline' and r['converged']), None)
    
    if baseline:
        print(f"\nBaseline reference:")
        print(f"  Total Cost: ${baseline['total_cost']:.2f}")
        print(f"  Voltage range: {baseline['voltages'].min():.4f} - {baseline['voltages'].max():.4f} pu")
        print(f"  Max line loading: {baseline['line_loading'].max():.2f}%")
        print(f"  Total system losses: {baseline['total_losses']:.3f} MW")
        max_loss_line_baseline = baseline['branch_losses'].idxmax()
        print(f"  Max branch loss: {baseline['branch_losses'].max():.3f} MW at Line {max_loss_line_baseline}")
        
        print(f"\nChanges relative to baseline:")
        for result in all_results:
            if result['converged'] and result['case_name'] != 'Baseline':
                cost_change = ((result['total_cost'] - baseline['total_cost']) / baseline['total_cost']) * 100
                voltage_change_min = ((result['voltages'].min() - baseline['voltages'].min()) / baseline['voltages'].min()) * 100
                loading_change = result['line_loading'].max() - baseline['line_loading'].max()
                losses_change = result['total_losses'] - baseline['total_losses']
                losses_change_pct = ((result['total_losses'] - baseline['total_losses']) / baseline['total_losses']) * 100
                max_loss_line = result['branch_losses'].idxmax()
                max_loss_value = result['branch_losses'].max()
                
                print(f"\n  {result['case_name']} ({result['demand_multiplier']*100:.0f}% demand):")
                print(f"    Cost change: {cost_change:+.2f}%")
                print(f"    Min voltage change: {voltage_change_min:+.3f}%")
                print(f"    Max loading change: {loading_change:+.2f} percentage points")
                print(f"    Total losses change: {losses_change:+.3f} MW ({losses_change_pct:+.2f}%)")
                print(f"    Max branch loss: {max_loss_value:.3f} MW at Line {max_loss_line}")


if __name__ == "__main__":
    main()
