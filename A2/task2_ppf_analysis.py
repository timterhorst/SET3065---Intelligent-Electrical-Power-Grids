"""
Task 2: Probabilistic Power Flow Analysis
SET3065 - Intelligent Electrical Power Grids - Assignment A2

Analyzes wind power plant location and reactive power control strategies
using Monte Carlo probabilistic power flow (N=100 samples).
"""

from pathlib import Path
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

pd.options.display.max_columns = None

THIS_DIR = Path(__file__).resolve().parent

# ============================================================================
# CONFIGURATION
# ============================================================================

# Wind power plant parameters (from provided script)
k = 2.02              # Weibull shape parameter [p.u]
lambda_ = 11          # Weibull scale parameter [m/s]
Pwpp = 180            # Max. MW of wind power plant
wsin = 3              # Cut-in wind speed [m/s]
wsr = 12              # Rated wind speed [m/s]
wsout = 20            # Cut-off wind speed [m/s]

# PPF parameters
N = 100               # Number of Monte Carlo samples
RANDOM_SEED = 5489    # For reproducibility

# Scenario definitions
# Transformer 3 connects wind farm (LV = bus 10, pp index 9, internal) to main grid (HV).
# We change hv_bus to move the grid connection point; lv_bus stays 9 (wind farm internal).
scenarios = [
    # Baseline (Wind @ Bus 9, pp hv_bus 8)
    {'name': 'Baseline_UnitPF', 'trafo_hv_bus': 8, 'pf': 1.0, 'mode': 'unity'},
    {'name': 'Baseline_PF095_OE', 'trafo_hv_bus': 8, 'pf': 0.95, 'mode': 'overexcited'},
    {'name': 'Baseline_PF095_UE', 'trafo_hv_bus': 8, 'pf': 0.95, 'mode': 'underexcited'},
    # Wind @ Bus 7 (pp hv_bus 6)
    {'name': 'Wind@Bus7_UnitPF', 'trafo_hv_bus': 6, 'pf': 1.0, 'mode': 'unity'},
    {'name': 'Wind@Bus7_PF095_OE', 'trafo_hv_bus': 6, 'pf': 0.95, 'mode': 'overexcited'},
    {'name': 'Wind@Bus7_PF095_UE', 'trafo_hv_bus': 6, 'pf': 0.95, 'mode': 'underexcited'},
    # Wind @ Bus 5 (pp hv_bus 4)
    {'name': 'Wind@Bus5_UnitPF', 'trafo_hv_bus': 4, 'pf': 1.0, 'mode': 'unity'},
    {'name': 'Wind@Bus5_PF095_OE', 'trafo_hv_bus': 4, 'pf': 0.95, 'mode': 'overexcited'},
    {'name': 'Wind@Bus5_PF095_UE', 'trafo_hv_bus': 4, 'pf': 0.95, 'mode': 'underexcited'},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_wind_power(wind_speed):
    """
    Calculate wind power output based on wind speed.
    Logic from provided prob_opf_ieee9_wind.py script.
    """
    if wind_speed < wsin:
        return 0
    elif wsin <= wind_speed < wsr:
        return Pwpp * (wind_speed**3 - wsin**3) / (wsr**3 - wsin**3)
    elif wsr <= wind_speed < wsout:
        return Pwpp
    else:  # wind_speed >= wsout
        return 0


def configure_wind_reactive_power(net, p_wind, pf, mode):
    """
    Configure wind generator reactive power based on power factor and mode.

    Args:
        net: pandapower network
        p_wind: Active power output [MW]
        pf: Power factor [0-1]
        mode: 'unity', 'overexcited', or 'underexcited'
    """
    if mode == 'unity':
        Q = 0
    elif mode == 'overexcited':
        # Overexcited: supplies reactive power (positive Q, voltage support)
        Q = p_wind * np.tan(np.arccos(pf))
    elif mode == 'underexcited':
        # Underexcited: absorbs reactive power (negative Q, voltage reduction)
        Q = -p_wind * np.tan(np.arccos(pf))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    net.sgen.q_mvar = Q
    net.sgen.max_q_mvar = Q
    net.sgen.min_q_mvar = Q
    net.sgen.max_p_mw = p_wind
    net.sgen.min_p_mw = p_wind
    net.sgen.p_mw = p_wind


def initialize_results_storage():
    """Initialize data structure for storing PPF results."""
    return {scenario['name']: {
        'voltages': [],      # List of voltage arrays (one per converged sample)
        'loadings': [],      # List of loading arrays
        'losses': [],        # List of loss arrays
        'converged': 0,      # Count of converged samples
        'total': N,          # Total samples attempted
    } for scenario in scenarios}


def calculate_statistics(results):
    """Calculate min/max/mean/std statistics for all scenarios."""
    for scenario_name in results:
        if results[scenario_name]['converged'] > 0:
            # Convert lists to numpy arrays
            results[scenario_name]['voltages'] = np.array(results[scenario_name]['voltages'])
            results[scenario_name]['loadings'] = np.array(results[scenario_name]['loadings'])
            results[scenario_name]['losses'] = np.array(results[scenario_name]['losses'])

            # Calculate statistics (axis=0 means across samples, per bus/line)
            results[scenario_name]['voltage_stats'] = {
                'min': np.min(results[scenario_name]['voltages'], axis=0),
                'max': np.max(results[scenario_name]['voltages'], axis=0),
                'mean': np.mean(results[scenario_name]['voltages'], axis=0),
                'std': np.std(results[scenario_name]['voltages'], axis=0),
            }

            results[scenario_name]['loading_stats'] = {
                'min': np.min(results[scenario_name]['loadings'], axis=0),
                'max': np.max(results[scenario_name]['loadings'], axis=0),
                'mean': np.mean(results[scenario_name]['loadings'], axis=0),
                'std': np.std(results[scenario_name]['loadings'], axis=0),
            }

            results[scenario_name]['loss_stats'] = {
                'min': np.min(results[scenario_name]['losses'], axis=0),
                'max': np.max(results[scenario_name]['losses'], axis=0),
                'mean': np.mean(results[scenario_name]['losses'], axis=0),
                'std': np.std(results[scenario_name]['losses'], axis=0),
            }


# ============================================================================
# MAIN PPF ANALYSIS
# ============================================================================

def run_ppf_analysis():
    """Execute probabilistic power flow for all scenarios."""

    print("=" * 80)
    print("TASK 2: PROBABILISTIC POWER FLOW ANALYSIS")
    print("=" * 80)

    # Initialize results storage
    results = initialize_results_storage()

    # Generate wind speed samples (same for all scenarios for fair comparison)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ws_samples = [random.weibullvariate(lambda_, k) for _ in range(N)]
    wind_power_samples = [calculate_wind_power(ws) for ws in ws_samples]

    # Load base network to get load parameters
    net_base = pp.from_excel(THIS_DIR / 'ieee9-wind.xlsx')
    load_means = np.asarray(net_base.load.p_mw)
    load_std = np.random.uniform(1, 30, len(net_base.load))
    pq_ratio = np.asarray(net_base.load.q_mvar / net_base.load.p_mw)

    # Run each scenario
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Running: {scenario['name']}")
        print(f"  Wind location: Trafo HV bus (grid) = {scenario['trafo_hv_bus']} (diagram bus {scenario['trafo_hv_bus']+1})")
        print(f"  Power factor: {scenario['pf']} ({scenario['mode']})")
        print(f"{'='*80}")

        # Reload network for clean state
        net = pp.from_excel(THIS_DIR / 'ieee9-wind.xlsx')

        # Configure wind plant location: transformer 3 HV side = grid connection point.
        # LV side stays at bus 10 (pp index 9), internal to wind farm.
        net.trafo.at[3, 'hv_bus'] = scenario['trafo_hv_bus']

        # Reset random seed for reproducible load variations
        np.random.seed(RANDOM_SEED)

        # Run N Monte Carlo samples
        for sample_idx, p_wind in enumerate(wind_power_samples):
            # Generate random load variations (normal distribution)
            net.load.p_mw = np.random.normal(load_means, load_std)
            net.load.q_mvar = np.asarray(net.load.p_mw) * pq_ratio

            # Configure wind generator reactive power
            configure_wind_reactive_power(net, p_wind, scenario['pf'], scenario['mode'])

            # Run OPF with exception handling
            try:
                pp.runpp(net, numba=False)
                pp.runopp(net, init='pf', verbose=False, numba=False)

                # Store results
                results[scenario['name']]['voltages'].append(net.res_bus.vm_pu.values.copy())
                results[scenario['name']]['loadings'].append(net.res_line.loading_percent.values.copy())
                results[scenario['name']]['losses'].append(net.res_line.pl_mw.values.copy())
                results[scenario['name']]['converged'] += 1

            except Exception:
                # OPF did not converge for this sample - skip
                continue

        convergence_rate = 100 * results[scenario['name']]['converged'] / N
        print(f"  ✓ Converged: {results[scenario['name']]['converged']}/{N} ({convergence_rate:.1f}%)")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

# Shared layout for 3×3 comprehensive grids (rows = PF mode, cols = location)
GRID_PF_MODES = ['UnitPF', 'PF095_OE', 'PF095_UE']
GRID_LOCATIONS = ['Baseline', 'Wind@Bus7', 'Wind@Bus5']
GRID_PF_LABELS = {
    'UnitPF': 'Unity PF (1.0)',
    'PF095_OE': 'PF 0.95 Overexcited',
    'PF095_UE': 'PF 0.95 Underexcited',
}


def plot_voltage_variability_comprehensive(results):
    """
    Create 3×3 grid showing all wind location × reactive power combinations.
    Rows = PF modes (Unity, OE, UE), Columns = Locations (Baseline/Bus9, Bus7, Bus5).
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Voltage Variability: Wind Location vs Reactive Power Control',
                 fontsize=16, fontweight='bold', y=0.995)

    for row_idx, pf_mode in enumerate(GRID_PF_MODES):
        for col_idx, location in enumerate(GRID_LOCATIONS):
            ax = axes[row_idx, col_idx]
            scenario_name = f'{location}_{pf_mode}'

            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                ax.text(0.5, 0.5, 'No converged\nsamples',
                        ha='center', va='center', fontsize=12, color='red',
                        transform=ax.transAxes)
                ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n(0% conv.)',
                             fontsize=10)
                ax.set_xlim([0, 12])
                ax.set_ylim([0.92, 1.08])
                continue

            stats = results[scenario_name]['voltage_stats']
            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total

            bus_indices = np.arange(len(stats['min'])) + 1  # diagram numbering

            ax.fill_between(bus_indices, stats['min'], stats['max'],
                            alpha=0.3, color='blue', label='Min-Max Range')
            ax.plot(bus_indices, stats['mean'], 'o-', color='darkblue',
                    linewidth=2, markersize=4, label='Mean', markerfacecolor='blue')

            ax.axhline(y=1.05, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='Limits (±5%)' if row_idx == 0 and col_idx == 0 else '')
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            ax.set_ylim([0.92, 1.08])
            ax.set_xlim([0.5, len(stats['min']) + 0.5])
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            if row_idx == 2:
                ax.set_xlabel('Bus Number', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('Voltage (pu)', fontsize=10)

            title_color = 'black' if conv_rate >= 70 else 'orange' if conv_rate >= 50 else 'red'
            ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n({conv_rate:.0f}% conv.)',
                         fontsize=9, color=title_color, fontweight='bold' if conv_rate < 70 else 'normal')

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = THIS_DIR / 'task2_voltage_variability_comprehensive.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out_path}")


def plot_loading_variability_comprehensive(results):
    """
    Create 3×3 grid for line loading variability.
    Rows = PF modes, Columns = Locations (same layout as voltage).
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Line Loading Variability: Wind Location vs Reactive Power Control',
                 fontsize=16, fontweight='bold', y=0.995)

    n_lines_ref = None  # infer from first valid scenario for empty-cell xlim
    for row_idx, pf_mode in enumerate(GRID_PF_MODES):
        for col_idx, location in enumerate(GRID_LOCATIONS):
            ax = axes[row_idx, col_idx]
            scenario_name = f'{location}_{pf_mode}'

            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                ax.text(0.5, 0.5, 'No converged\nsamples',
                        ha='center', va='center', fontsize=12, color='red',
                        transform=ax.transAxes)
                ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n(0% conv.)',
                             fontsize=10)
                ax.set_xlim([-0.5, (n_lines_ref or 7) - 0.5])
                ax.set_ylim([0, 110])
                continue

            stats = results[scenario_name]['loading_stats']
            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            n_lines = len(stats['min'])
            if n_lines_ref is None:
                n_lines_ref = n_lines

            line_indices = np.arange(n_lines)

            ax.fill_between(line_indices, stats['min'], stats['max'],
                            alpha=0.3, color='green', label='Min-Max Range')
            ax.plot(line_indices, stats['mean'], 'o-', color='darkgreen',
                    linewidth=2, markersize=4, label='Mean', markerfacecolor='green')

            ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='Limit (100%)' if row_idx == 0 and col_idx == 0 else '')

            if n_lines > 6:
                ax.axvline(x=6, color='purple', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(6, 105, 'Line 6\n(wind)', ha='center', fontsize=7, color='purple')

            y_max = 110
            if np.any(np.isfinite(stats['max'])):
                y_max = max(110, np.nanmax(stats['max']) * 1.05)
            ax.set_ylim([0, y_max])
            ax.set_xlim([-0.5, n_lines - 0.5])
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            if row_idx == 2:
                ax.set_xlabel('Line Index', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('Loading (%)', fontsize=10)

            title_color = 'black' if conv_rate >= 70 else 'orange' if conv_rate >= 50 else 'red'
            ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n({conv_rate:.0f}% conv.)',
                         fontsize=9, color=title_color, fontweight='bold' if conv_rate < 70 else 'normal')

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = THIS_DIR / 'task2_loading_variability_comprehensive.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_loss_variability_comprehensive(results):
    """
    Create 3×3 grid for power loss variability.
    Rows = PF modes, Columns = Locations (same layout as voltage/loading).
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Active Power Loss Variability: Wind Location vs Reactive Power Control',
                 fontsize=16, fontweight='bold', y=0.995)

    n_lines_ref = None
    for row_idx, pf_mode in enumerate(GRID_PF_MODES):
        for col_idx, location in enumerate(GRID_LOCATIONS):
            ax = axes[row_idx, col_idx]
            scenario_name = f'{location}_{pf_mode}'

            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                ax.text(0.5, 0.5, 'No converged\nsamples',
                        ha='center', va='center', fontsize=12, color='red',
                        transform=ax.transAxes)
                ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n(0% conv.)',
                             fontsize=10)
                ax.set_xlim([-0.5, (n_lines_ref or 7) - 0.5])
                ax.set_ylim([0, 6])
                continue

            stats = results[scenario_name]['loss_stats']
            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            n_lines = len(stats['min'])
            if n_lines_ref is None:
                n_lines_ref = n_lines

            line_indices = np.arange(n_lines)

            ax.fill_between(line_indices, stats['min'], stats['max'],
                            alpha=0.3, color='orange', label='Min-Max Range')
            ax.plot(line_indices, stats['mean'], 'o-', color='darkorange',
                    linewidth=2, markersize=4, label='Mean', markerfacecolor='orange')

            total_loss_mean = np.nansum(stats['mean'])
            ax.text(0.98, 0.98, f'Total: {total_loss_mean:.2f} MW',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=8, fontweight='bold')

            y_max = max(6, (np.nanmax(stats['max']) * 1.1) if np.any(np.isfinite(stats['max'])) else 6)
            ax.set_ylim([0, y_max])
            ax.set_xlim([-0.5, n_lines - 0.5])
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            if row_idx == 2:
                ax.set_xlabel('Line Index', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('Active Power Loss (MW)', fontsize=10)

            title_color = 'black' if conv_rate >= 70 else 'orange' if conv_rate >= 50 else 'red'
            ax.set_title(f'{location.replace("Wind@", "")}\n{GRID_PF_LABELS[pf_mode]}\n({conv_rate:.0f}% conv.)',
                         fontsize=9, color=title_color, fontweight='bold' if conv_rate < 70 else 'normal')

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = THIS_DIR / 'task2_loss_variability_comprehensive.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# Shared config for overlay comparison plots (all 9 scenarios: label, color, linestyle)
SCENARIOS_OVERLAY_CONFIG = [
    {'name': 'Baseline_UnitPF', 'label': 'Baseline Unity PF', 'color': 'blue', 'linestyle': '-'},
    {'name': 'Baseline_PF095_OE', 'label': 'Baseline PF0.95 OE', 'color': 'green', 'linestyle': '-'},
    {'name': 'Baseline_PF095_UE', 'label': 'Baseline PF0.95 UE', 'color': 'orange', 'linestyle': '-'},
    {'name': 'Wind@Bus7_UnitPF', 'label': 'Bus7 Unity PF', 'color': 'blue', 'linestyle': '--'},
    {'name': 'Wind@Bus7_PF095_OE', 'label': 'Bus7 PF0.95 OE', 'color': 'green', 'linestyle': '--'},
    {'name': 'Wind@Bus7_PF095_UE', 'label': 'Bus7 PF0.95 UE', 'color': 'orange', 'linestyle': '--'},
    {'name': 'Wind@Bus5_UnitPF', 'label': 'Bus5 Unity PF', 'color': 'blue', 'linestyle': ':'},
    {'name': 'Wind@Bus5_PF095_OE', 'label': 'Bus5 PF0.95 OE', 'color': 'green', 'linestyle': ':'},
    {'name': 'Wind@Bus5_PF095_UE', 'label': 'Bus5 PF0.95 UE', 'color': 'orange', 'linestyle': ':'},
]


def plot_voltage_comparison_comprehensive(results):
    """
    Single plot showing selected scenarios overlaid.
    Reduced to 6 key scenarios; improved clarity via lighter bands and thicker median lines.
    """
    scenarios_config = [
        {'name': 'Baseline_UnitPF', 'label': 'Baseline Unity', 'color': '#4477AA', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_UnitPF', 'label': 'Bus7 Unity', 'color': '#228833', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus5_UnitPF', 'label': 'Bus5 Unity', 'color': '#EE6677', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_PF095_OE', 'label': 'Bus7 PF0.95 OE', 'color': '#228833', 'linestyle': '--', 'linewidth': 2.5, 'band_alpha': 0.10},
        {'name': 'Wind@Bus7_PF095_UE', 'label': 'Bus7 PF0.95 UE', 'color': '#44AA99', 'linestyle': ':', 'linewidth': 2.0, 'band_alpha': 0.08},
        {'name': 'Baseline_PF095_OE', 'label': 'Baseline PF0.95 OE', 'color': '#4477AA', 'linestyle': '--', 'linewidth': 2.0, 'band_alpha': 0.08},
    ]

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    n_buses = 12

    for scenario in scenarios_config:
        scenario_name = scenario['name']
        if scenario_name not in results or results[scenario_name]['converged'] == 0:
            continue

        voltages = results[scenario_name]['voltages']
        n_buses = voltages.shape[1]
        bus_indices = np.arange(n_buses) + 1

        median = np.percentile(voltages, 50, axis=0)
        p5 = np.percentile(voltages, 5, axis=0)
        p95 = np.percentile(voltages, 95, axis=0)

        total = results[scenario_name]['total']
        conv_rate = 100 * results[scenario_name]['converged'] / total
        line_alpha = 0.95 if conv_rate >= 70 else 0.4
        band_alpha = scenario['band_alpha'] if conv_rate >= 70 else scenario['band_alpha'] * 0.5

        ax.fill_between(bus_indices, p5, p95,
                        alpha=band_alpha, color=scenario['color'], linewidth=0,
                        edgecolor='none')

        label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
        marker = 'o' if conv_rate >= 90 else 'x'
        ax.plot(bus_indices, median,
                color=scenario['color'],
                linestyle=scenario['linestyle'],
                linewidth=scenario['linewidth'],
                label=label_text,
                marker=marker,
                markersize=5,
                markevery=2,
                alpha=line_alpha,
                zorder=10)

    ax.axhline(y=1.05, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8,
               label='Voltage Limits (±5%)', zorder=5)
    ax.axhline(y=0.95, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8, zorder=5)

    ax.set_xlabel('Bus Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Voltage (pu)', fontsize=13, fontweight='bold')
    ax.set_title('Voltage Variability: Key Scenarios Comparison (Median + 5-95% Band)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim([0.94, 1.06])
    ax.set_xlim([0.5, n_buses + 0.5])
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5, zorder=0)

    legend = ax.legend(loc='lower left', fontsize=10, ncol=2, framealpha=0.98,
                      title='Scenario (convergence rate)', edgecolor='gray')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(10)

    ax.text(0.99, 0.97,
            'Line style:\nSolid = Unity PF\nDashed = Overexcited\nDotted = Underexcited\n\nFaded = <70% conv.',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    out_path = THIS_DIR / 'task2_voltage_comparison_reduced.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out_path}")


def plot_loading_comparison_comprehensive(results):
    """
    Single plot for line loading - 6 key scenarios; improved clarity.
    """
    scenarios_config = [
        {'name': 'Baseline_UnitPF', 'label': 'Baseline Unity', 'color': '#4477AA', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_UnitPF', 'label': 'Bus7 Unity', 'color': '#228833', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus5_UnitPF', 'label': 'Bus5 Unity', 'color': '#EE6677', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_PF095_OE', 'label': 'Bus7 PF0.95 OE', 'color': '#228833', 'linestyle': '--', 'linewidth': 2.5, 'band_alpha': 0.10},
        {'name': 'Wind@Bus7_PF095_UE', 'label': 'Bus7 PF0.95 UE', 'color': '#44AA99', 'linestyle': ':', 'linewidth': 2.0, 'band_alpha': 0.08},
        {'name': 'Baseline_PF095_OE', 'label': 'Baseline PF0.95 OE', 'color': '#4477AA', 'linestyle': '--', 'linewidth': 2.0, 'band_alpha': 0.08},
    ]

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    n_lines = 7

    for scenario in scenarios_config:
        scenario_name = scenario['name']
        if scenario_name not in results or results[scenario_name]['converged'] == 0:
            continue

        loadings = results[scenario_name]['loadings']
        n_lines = loadings.shape[1]
        line_indices = np.arange(n_lines)

        median = np.percentile(loadings, 50, axis=0)
        p5 = np.percentile(loadings, 5, axis=0)
        p95 = np.percentile(loadings, 95, axis=0)

        total = results[scenario_name]['total']
        conv_rate = 100 * results[scenario_name]['converged'] / total
        line_alpha = 0.95 if conv_rate >= 70 else 0.4
        band_alpha = scenario['band_alpha'] if conv_rate >= 70 else scenario['band_alpha'] * 0.5

        ax.fill_between(line_indices, p5, p95,
                        alpha=band_alpha, color=scenario['color'], linewidth=0,
                        edgecolor='none')

        label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
        marker = 'o' if conv_rate >= 90 else 'x'
        ax.plot(line_indices, median,
                color=scenario['color'],
                linestyle=scenario['linestyle'],
                linewidth=scenario['linewidth'],
                label=label_text,
                marker=marker,
                markersize=5,
                alpha=line_alpha,
                zorder=10)

    ax.axhline(y=100, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8,
               label='Thermal Limit (100%)', zorder=5)
    if n_lines > 6:
        ax.axvline(x=6, color='purple', linestyle=':', linewidth=2.5, alpha=0.6, zorder=3)
        ax.text(6, 107, 'Line 6\n(wind)', ha='center', fontsize=10, color='purple',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.85, edgecolor='purple', linewidth=1.5))

    ax.set_xlabel('Line Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loading (%)', fontsize=13, fontweight='bold')
    ax.set_title('Line Loading Variability: Key Scenarios Comparison (Median + 5-95% Band)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim([0, 110])
    ax.set_xlim([-0.5, n_lines - 0.5])
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5, zorder=0)

    legend = ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.98,
                      title='Scenario (convergence rate)', edgecolor='gray')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(10)

    ax.text(0.99, 0.97,
            'Line style:\nSolid = Unity PF\nDashed = Overexcited\nDotted = Underexcited\n\nFaded = <70% conv.',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    out_path = THIS_DIR / 'task2_loading_comparison_reduced.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_loss_comparison_comprehensive(results):
    """
    Single plot for power losses - 6 key scenarios; improved clarity.
    """
    scenarios_config = [
        {'name': 'Baseline_UnitPF', 'label': 'Baseline Unity', 'color': '#4477AA', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_UnitPF', 'label': 'Bus7 Unity', 'color': '#228833', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus5_UnitPF', 'label': 'Bus5 Unity', 'color': '#EE6677', 'linestyle': '-', 'linewidth': 3.0, 'band_alpha': 0.12},
        {'name': 'Wind@Bus7_PF095_OE', 'label': 'Bus7 PF0.95 OE', 'color': '#228833', 'linestyle': '--', 'linewidth': 2.5, 'band_alpha': 0.10},
        {'name': 'Wind@Bus7_PF095_UE', 'label': 'Bus7 PF0.95 UE', 'color': '#44AA99', 'linestyle': ':', 'linewidth': 2.0, 'band_alpha': 0.08},
        {'name': 'Baseline_PF095_OE', 'label': 'Baseline PF0.95 OE', 'color': '#4477AA', 'linestyle': '--', 'linewidth': 2.0, 'band_alpha': 0.08},
    ]

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    n_lines = 7
    y_max = 5.0
    for scenario in scenarios_config:
        sn = scenario['name']
        if sn in results and results[sn]['converged'] > 0:
            p95 = np.percentile(results[sn]['losses'], 95, axis=0)
            if np.any(np.isfinite(p95)):
                y_max = max(y_max, np.nanmax(p95) * 1.1)

    for scenario in scenarios_config:
        scenario_name = scenario['name']
        if scenario_name not in results or results[scenario_name]['converged'] == 0:
            continue

        losses = results[scenario_name]['losses']
        n_lines = losses.shape[1]
        line_indices = np.arange(n_lines)

        median = np.percentile(losses, 50, axis=0)
        p5 = np.percentile(losses, 5, axis=0)
        p95 = np.percentile(losses, 95, axis=0)

        total = results[scenario_name]['total']
        conv_rate = 100 * results[scenario_name]['converged'] / total
        line_alpha = 0.95 if conv_rate >= 70 else 0.4
        band_alpha = scenario['band_alpha'] if conv_rate >= 70 else scenario['band_alpha'] * 0.5

        ax.fill_between(line_indices, p5, p95,
                        alpha=band_alpha, color=scenario['color'], linewidth=0,
                        edgecolor='none')

        label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
        marker = 'o' if conv_rate >= 90 else 'x'
        ax.plot(line_indices, median,
                color=scenario['color'],
                linestyle=scenario['linestyle'],
                linewidth=scenario['linewidth'],
                label=label_text,
                marker=marker,
                markersize=5,
                alpha=line_alpha,
                zorder=10)

    ax.set_xlabel('Line Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Active Power Loss (MW)', fontsize=13, fontweight='bold')
    ax.set_title('Power Loss Variability: Key Scenarios Comparison (Median + 5-95% Band)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim([0, y_max])
    ax.set_xlim([-0.5, n_lines - 0.5])
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5, zorder=0)

    legend = ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.98,
                      title='Scenario (convergence rate)', edgecolor='gray')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(10)

    ax.text(0.99, 0.97,
            'Line style:\nSolid = Unity PF\nDashed = Overexcited\nDotted = Underexcited\n\nFaded = <70% conv.',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    out_path = THIS_DIR / 'task2_loss_comparison_reduced.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_total_loss_boxplots(results):
    """
    Boxplot showing distribution of total system losses per scenario.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    scenarios_order = [
        'Baseline_UnitPF', 'Baseline_PF095_OE', 'Baseline_PF095_UE',
        'Wind@Bus7_UnitPF', 'Wind@Bus7_PF095_OE', 'Wind@Bus7_PF095_UE',
        'Wind@Bus5_UnitPF', 'Wind@Bus5_PF095_OE', 'Wind@Bus5_PF095_UE',
    ]
    color_map = {'Baseline': '#1f77b4', 'Wind@Bus7': '#2ca02c', 'Wind@Bus5': '#ff7f0e'}

    data_to_plot = []
    labels = []
    colors = []

    for scenario_name in scenarios_order:
        if scenario_name not in results or results[scenario_name]['converged'] == 0:
            continue
        losses = results[scenario_name]['losses']
        total_losses = np.sum(losses, axis=1)
        data_to_plot.append(total_losses)
        total = results[scenario_name]['total']
        conv_rate = 100 * results[scenario_name]['converged'] / total
        location = scenario_name.split('_')[0]
        pf_mode = '_'.join(scenario_name.split('_')[1:])
        labels.append(f"{location}\n{pf_mode}\n({conv_rate:.0f}%)")
        colors.append(color_map.get(location, '#888888'))

    if not data_to_plot:
        ax.text(0.5, 0.5, 'No converged scenarios', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    else:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='orange', linewidth=2, linestyle='--'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel('Total Active Power Losses (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Total System Loss Distribution per Scenario',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    out_path = THIS_DIR / 'task2_total_loss_boxplots.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_critical_bus_voltage_boxplots(results):
    """
    Boxplot for voltage at critical buses (Bus 3, Bus 9, Bus 12 in diagram; pp indices 2, 8, 11).
    """
    critical_buses = [2, 8, 11]  # pp indices for diagram buses 3, 9, 12
    bus_names = ['Bus 3', 'Bus 9', 'Bus 12']

    scenarios_order = [
        'Baseline_UnitPF', 'Baseline_PF095_OE', 'Baseline_PF095_UE',
        'Wind@Bus7_UnitPF', 'Wind@Bus7_PF095_OE', 'Wind@Bus7_PF095_UE',
        'Wind@Bus5_UnitPF', 'Wind@Bus5_PF095_OE', 'Wind@Bus5_PF095_UE',
    ]
    location_color_map = {'Baseline': '#1f77b4', 'Wind@Bus7': '#2ca02c', 'Wind@Bus5': '#ff7f0e'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Voltage Distribution at Critical Buses', fontsize=14, fontweight='bold')

    for ax, bus_idx, bus_name in zip(axes, critical_buses, bus_names):
        data_to_plot = []
        labels = []
        colors = []

        for scenario_name in scenarios_order:
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue
            voltages = results[scenario_name]['voltages']
            if bus_idx >= voltages.shape[1]:
                continue
            data_to_plot.append(voltages[:, bus_idx])
            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            labels.append(f"{scenario_name.replace('_', ' ')}\n({conv_rate:.0f}%)")
            location = scenario_name.split('_')[0]
            colors.append(location_color_map.get(location, '#888888'))

        if not data_to_plot:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                            boxprops=dict(linewidth=1),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1),
                            medianprops=dict(color='red', linewidth=2))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.axhline(y=1.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Voltage (pu)', fontsize=10, fontweight='bold')
        ax.set_title(bus_name, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim([0.93, 1.07])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right', fontsize=7)

    plt.tight_layout()
    out_path = THIS_DIR / 'task2_critical_bus_boxplots.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# ============================================================================
# 2-PANEL COMPARISON FIGURES (Location effect + Reactive power effect)
# ============================================================================

def plot_voltage_location_comparison(results):
    """
    2-panel figure comparing wind locations.
    Panel 1: Unity PF - 3 locations. Panel 2: PF 0.95 OE - 3 locations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Voltage Variability: Wind Location Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Unity Power Factor (1.0)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_UnitPF', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Power Factor 0.95 Overexcited',
            'scenarios': [
                {'name': 'Baseline_PF095_OE', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_PF095_OE', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_buses = 12
    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            voltages = results[scenario_name]['voltages']
            n_buses = voltages.shape[1]
            bus_indices = np.arange(n_buses) + 1

            median = np.percentile(voltages, 50, axis=0)
            p5 = np.percentile(voltages, 5, axis=0)
            p95 = np.percentile(voltages, 95, axis=0)

            ax.fill_between(bus_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(bus_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='o',
                    markersize=4,
                    markevery=2)

        ax.axhline(y=1.05, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

        ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voltage (pu)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0.94, 1.06])
        ax.set_xlim([0.5, n_buses + 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval  |  Red dashed: ±5% voltage limits',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_voltage_location_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out_path}")


def plot_loading_location_comparison(results):
    """2-panel figure comparing wind locations for line loading."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Line Loading Variability: Wind Location Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Unity Power Factor (1.0)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_UnitPF', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Power Factor 0.95 Overexcited',
            'scenarios': [
                {'name': 'Baseline_PF095_OE', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_PF095_OE', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_lines = 7
    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            loadings = results[scenario_name]['loadings']
            n_lines = loadings.shape[1]
            line_indices = np.arange(n_lines)

            median = np.percentile(loadings, 50, axis=0)
            p5 = np.percentile(loadings, 5, axis=0)
            p95 = np.percentile(loadings, 95, axis=0)

            ax.fill_between(line_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(line_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='s',
                    markersize=4)

        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
        if n_lines > 6:
            ax.axvline(x=6, color='purple', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
            ax.text(6, 102, 'Line 6\n(wind)', ha='center', fontsize=8, color='purple', fontweight='bold')

        ax.set_xlabel('Line Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loading (%)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, 110])
        ax.set_xlim([-0.5, n_lines - 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval  |  Red dashed: 100% thermal limit',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_loading_location_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_loss_location_comparison(results):
    """2-panel figure comparing wind locations for power losses."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Power Loss Variability: Wind Location Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Unity Power Factor (1.0)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_UnitPF', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Power Factor 0.95 Overexcited',
            'scenarios': [
                {'name': 'Baseline_PF095_OE', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'Wind @ Bus 7', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus5_PF095_OE', 'label': 'Wind @ Bus 5', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_lines = 7
    y_max = 5.0
    for panel in panels:
        for scenario in panel['scenarios']:
            sn = scenario['name']
            if sn not in results or results[sn]['converged'] == 0:
                continue
            p95 = np.percentile(results[sn]['losses'], 95, axis=0)
            if np.any(np.isfinite(p95)):
                y_max = max(y_max, np.nanmax(p95) * 1.1)

    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            losses = results[scenario_name]['losses']
            n_lines = losses.shape[1]
            line_indices = np.arange(n_lines)

            median = np.percentile(losses, 50, axis=0)
            p5 = np.percentile(losses, 5, axis=0)
            p95 = np.percentile(losses, 95, axis=0)

            ax.fill_between(line_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(line_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='^',
                    markersize=4)

        ax.set_xlabel('Line Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Active Power Loss (MW)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, y_max])
        ax.set_xlim([-0.5, n_lines - 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_loss_location_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_voltage_reactive_comparison(results):
    """
    2-panel figure comparing reactive power modes.
    Panel 1: Baseline location. Panel 2: Wind @ Bus 7.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Voltage Variability: Reactive Power Control Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Baseline (Wind @ Bus 9)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Baseline_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Baseline_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Wind @ Bus 7',
            'scenarios': [
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus7_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_buses = 12
    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            voltages = results[scenario_name]['voltages']
            n_buses = voltages.shape[1]
            bus_indices = np.arange(n_buses) + 1

            median = np.percentile(voltages, 50, axis=0)
            p5 = np.percentile(voltages, 5, axis=0)
            p95 = np.percentile(voltages, 95, axis=0)

            ax.fill_between(bus_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(bus_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='o',
                    markersize=4,
                    markevery=2,
                    alpha=0.9 if conv_rate >= 70 else 0.6)

        ax.axhline(y=1.05, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

        ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voltage (pu)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0.94, 1.06])
        ax.set_xlim([0.5, n_buses + 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval  |  Faded lines: <70% convergence (unreliable)',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_voltage_reactive_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_loading_reactive_comparison(results):
    """2-panel figure comparing reactive power modes for line loading."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Line Loading Variability: Reactive Power Control Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Baseline (Wind @ Bus 9)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Baseline_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Baseline_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Wind @ Bus 7',
            'scenarios': [
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus7_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_lines = 7
    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            loadings = results[scenario_name]['loadings']
            n_lines = loadings.shape[1]
            line_indices = np.arange(n_lines)

            median = np.percentile(loadings, 50, axis=0)
            p5 = np.percentile(loadings, 5, axis=0)
            p95 = np.percentile(loadings, 95, axis=0)

            ax.fill_between(line_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(line_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='s',
                    markersize=4,
                    alpha=0.9 if conv_rate >= 70 else 0.6)

        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
        if n_lines > 6:
            ax.axvline(x=6, color='purple', linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
            ax.text(6, 102, 'Line 6\n(wind)', ha='center', fontsize=8, color='purple', fontweight='bold')

        ax.set_xlabel('Line Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loading (%)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, 110])
        ax.set_xlim([-0.5, n_lines - 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval  |  Faded lines: <70% convergence (unreliable)',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_loading_reactive_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_loss_reactive_comparison(results):
    """2-panel figure comparing reactive power modes for power losses."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Power Loss Variability: Reactive Power Control Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    panels = [
        {
            'title': 'Baseline (Wind @ Bus 9)',
            'scenarios': [
                {'name': 'Baseline_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Baseline_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Baseline_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        },
        {
            'title': 'Wind @ Bus 7',
            'scenarios': [
                {'name': 'Wind@Bus7_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4', 'linestyle': '-'},
                {'name': 'Wind@Bus7_PF095_OE', 'label': 'PF 0.95 Overexcited', 'color': '#2ca02c', 'linestyle': '--'},
                {'name': 'Wind@Bus7_PF095_UE', 'label': 'PF 0.95 Underexcited', 'color': '#ff7f0e', 'linestyle': ':'},
            ]
        }
    ]

    n_lines = 7
    y_max = 5.0
    for panel in panels:
        for scenario in panel['scenarios']:
            sn = scenario['name']
            if sn not in results or results[sn]['converged'] == 0:
                continue
            p95 = np.percentile(results[sn]['losses'], 95, axis=0)
            if np.any(np.isfinite(p95)):
                y_max = max(y_max, np.nanmax(p95) * 1.1)

    for ax, panel in zip(axes, panels):
        for scenario in panel['scenarios']:
            scenario_name = scenario['name']
            if scenario_name not in results or results[scenario_name]['converged'] == 0:
                continue

            losses = results[scenario_name]['losses']
            n_lines = losses.shape[1]
            line_indices = np.arange(n_lines)

            median = np.percentile(losses, 50, axis=0)
            p5 = np.percentile(losses, 5, axis=0)
            p95 = np.percentile(losses, 95, axis=0)

            ax.fill_between(line_indices, p5, p95,
                            alpha=0.25, color=scenario['color'], linewidth=0)

            total = results[scenario_name]['total']
            conv_rate = 100 * results[scenario_name]['converged'] / total
            label_text = f"{scenario['label']} ({conv_rate:.0f}%)"
            ax.plot(line_indices, median,
                    color=scenario['color'],
                    linestyle=scenario['linestyle'],
                    linewidth=2.5,
                    label=label_text,
                    marker='^',
                    markersize=4,
                    alpha=0.9 if conv_rate >= 70 else 0.6)

        ax.set_xlabel('Line Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Active Power Loss (MW)', fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, y_max])
        ax.set_xlim([-0.5, n_lines - 0.5])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.01, 'Shaded area: 5-95% confidence interval  |  Faded lines: <70% convergence (unreliable)',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = THIS_DIR / 'task2_loss_reactive_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# ============================================================================
# REPORT-OPTIMIZED FIGURES (2-panel: location effect + reactive power effect)
# ============================================================================

_LOCATION_SCENARIOS = [
    {'name': 'Baseline_UnitPF', 'label': 'Baseline (Bus 9)', 'color': '#1f77b4'},
    {'name': 'Wind@Bus7_UnitPF', 'label': 'Wind @ Bus 7', 'color': '#2ca02c'},
    {'name': 'Wind@Bus5_UnitPF', 'label': 'Wind @ Bus 5', 'color': '#d62728'},
]

_REACTIVE_SCENARIOS = [
    {'name': 'Wind@Bus7_UnitPF', 'label': 'Unity PF', 'color': '#1f77b4'},
    {'name': 'Wind@Bus7_PF095_OE', 'label': 'PF 0.95 OE', 'color': '#2ca02c'},
    {'name': 'Wind@Bus7_PF095_UE', 'label': 'PF 0.95 UE', 'color': '#ff7f0e'},
]


def _report_panel(ax, scenario_list, results, metric, ylabel,
                  hlimits=None, ylim=None, legend_loc='best'):
    """Plot one panel of a 2-panel report figure (median + 5-95% band)."""
    n_elem = None
    for sc in scenario_list:
        name = sc['name']
        if name not in results or results[name]['converged'] == 0:
            continue
        data = results[name][metric]
        n_elem = data.shape[1]
        x = np.arange(n_elem) + (1 if metric == 'voltages' else 0)

        med = np.percentile(data, 50, axis=0)
        p5 = np.percentile(data, 5, axis=0)
        p95 = np.percentile(data, 95, axis=0)

        conv = 100 * results[name]['converged'] / results[name]['total']
        reliable = conv >= 70
        label = f"{sc['label']} ({conv:.0f}%)" + ('' if reliable else ' *')

        ax.fill_between(x, p5, p95,
                        alpha=0.20 if reliable else 0.08,
                        color=sc['color'])
        ax.plot(x, med, color=sc['color'], linewidth=2,
                marker='o', markersize=5,
                label=label, alpha=0.9 if reliable else 0.45)

    if hlimits:
        for yv in hlimits:
            ax.axhline(y=yv, color='darkred', linestyle='--',
                       linewidth=1.5, alpha=0.7)
    if ylim:
        ax.set_ylim(ylim)
    if n_elem is not None:
        if metric == 'voltages':
            ax.set_xlim([0.5, n_elem + 0.5])
        else:
            ax.set_xlim([-0.5, n_elem - 0.5])
    ax.set_xlabel('Bus Number' if metric == 'voltages' else 'Line Index',
                  fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(fontsize=9, loc=legend_loc, framealpha=0.95)


def plot_report_voltage(results):
    """Report figure: voltage variability (location + reactive power panels)."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    _report_panel(ax_a, _LOCATION_SCENARIOS, results, 'voltages',
                  'Voltage (pu)', hlimits=[0.95, 1.05],
                  ylim=[0.93, 1.07], legend_loc='lower left')
    ax_a.set_title('(a) Wind Plant Location Effect\n(Unity Power Factor)',
                   fontsize=11, fontweight='bold')

    _report_panel(ax_b, _REACTIVE_SCENARIOS, results, 'voltages',
                  'Voltage (pu)', hlimits=[0.95, 1.05],
                  ylim=[0.93, 1.07], legend_loc='lower left')
    ax_b.set_title('(b) Reactive Power Control Effect\n(Wind @ Bus 7)',
                   fontsize=11, fontweight='bold')

    fig.text(0.5, -0.01,
             'Shaded: 5\u201395th percentile  |  Lines: median  |'
             '  Red dashed: \u00b15% voltage limits  |  * <70% convergence',
             ha='center', fontsize=8.5, style='italic')

    plt.tight_layout()
    out = THIS_DIR / 'report_task2_voltage.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2713 Report figure: {out}")


def plot_report_loading(results):
    """Report figure: loading variability (location + reactive power panels)."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    _report_panel(ax_a, _LOCATION_SCENARIOS, results, 'loadings',
                  'Loading (%)', hlimits=[100], ylim=[0, 110],
                  legend_loc='upper left')
    ax_a.set_title('(a) Wind Plant Location Effect\n(Unity Power Factor)',
                   fontsize=11, fontweight='bold')

    _report_panel(ax_b, _REACTIVE_SCENARIOS, results, 'loadings',
                  'Loading (%)', hlimits=[100], ylim=[0, 110],
                  legend_loc='upper left')
    ax_b.set_title('(b) Reactive Power Control Effect\n(Wind @ Bus 7)',
                   fontsize=11, fontweight='bold')

    fig.text(0.5, -0.01,
             'Shaded: 5\u201395th percentile  |  Lines: median  |'
             '  Red dashed: thermal limit (100%)  |  * <70% convergence',
             ha='center', fontsize=8.5, style='italic')

    plt.tight_layout()
    out = THIS_DIR / 'report_task2_loading.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2713 Report figure: {out}")


def plot_report_losses(results):
    """Report figure: loss variability (location + reactive power panels)."""
    y_max = 1.0
    for sc_list in [_LOCATION_SCENARIOS, _REACTIVE_SCENARIOS]:
        for sc in sc_list:
            name = sc['name']
            if name in results and results[name]['converged'] > 0:
                p95 = np.percentile(results[name]['losses'], 95, axis=0)
                y_max = max(y_max, np.nanmax(p95) * 1.15)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    _report_panel(ax_a, _LOCATION_SCENARIOS, results, 'losses',
                  'Active Power Loss (MW)', ylim=[0, y_max],
                  legend_loc='upper right')
    ax_a.set_title('(a) Wind Plant Location Effect\n(Unity Power Factor)',
                   fontsize=11, fontweight='bold')

    _report_panel(ax_b, _REACTIVE_SCENARIOS, results, 'losses',
                  'Active Power Loss (MW)', ylim=[0, y_max],
                  legend_loc='upper right')
    ax_b.set_title('(b) Reactive Power Control Effect\n(Wind @ Bus 7)',
                   fontsize=11, fontweight='bold')

    fig.text(0.5, -0.01,
             'Shaded: 5\u201395th percentile  |  Lines: median  |'
             '  * <70% convergence',
             ha='center', fontsize=8.5, style='italic')

    plt.tight_layout()
    out = THIS_DIR / 'report_task2_losses.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2713 Report figure: {out}")


def generate_summary_table(results):
    """Generate enhanced comparison table with variability metrics for all scenarios."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - All Scenarios")
    print("=" * 100)

    data = []
    for scenario_name in results:
        if results[scenario_name]['converged'] == 0:
            data.append({
                'Scenario': scenario_name,
                'Conv. [%]': '0.0',
                'V_min [pu]': '--',
                'V_max [pu]': '--',
                'V_std_max': '--',
                'Max Load [%]': '--',
                'Load_std_max': '--',
                'Total Loss [MW]': '--',
            })
            continue

        total = results[scenario_name]['total']
        conv_rate = 100 * results[scenario_name]['converged'] / total
        v_stats = results[scenario_name]['voltage_stats']
        l_stats = results[scenario_name]['loading_stats']
        loss_stats = results[scenario_name]['loss_stats']

        data.append({
            'Scenario': scenario_name,
            'Conv. [%]': f"{conv_rate:.1f}",
            'V_min [pu]': f"{np.min(v_stats['min']):.4f}",
            'V_max [pu]': f"{np.max(v_stats['max']):.4f}",
            'V_std_max': f"{np.max(v_stats['std']):.4f}",
            'Max Load [%]': f"{np.max(l_stats['max']):.1f}",
            'Load_std_max': f"{np.max(l_stats['std']):.1f}",
            'Total Loss [MW]': f"{np.nansum(loss_stats['mean']):.2f}",
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("=" * 100)
    print("\nNotes:")
    print("  - V_std_max: Maximum voltage standard deviation across all buses (uncertainty metric)")
    print("  - Load_std_max: Maximum loading standard deviation across all lines")
    print("  - Scenarios with <70% convergence shown in orange/red in plots")
    print("=" * 100)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run PPF analysis
    results = run_ppf_analysis()

    # Calculate statistics
    print("\nCalculating statistics...")
    calculate_statistics(results)

    # Generate reduced comparison plots (6 key scenarios)
    print("\nGenerating comprehensive comparison plots (6 scenarios)...")
    plot_voltage_comparison_comprehensive(results)
    plot_loading_comparison_comprehensive(results)
    plot_loss_comparison_comprehensive(results)

    # Generate distribution summary
    print("\nGenerating distribution summary...")
    plot_total_loss_boxplots(results)

    # Generate report-optimized figures (2-panel: location + reactive power)
    print("\nGenerating report figures...")
    plot_report_voltage(results)
    plot_report_loading(results)
    plot_report_losses(results)

    # Generate summary table
    generate_summary_table(results)

    # Save results for later analysis
    out_pkl = THIS_DIR / 'task2_ppf_results.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved: {out_pkl}")

    print("\n" + "=" * 80)
    print("TASK 2 ANALYSIS COMPLETE")
    print("=" * 80)
