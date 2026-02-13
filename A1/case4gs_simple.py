import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as pp_plot


def run_scenario(load_at_bus2_p_mw: float):
    """Build case4gs, apply bus-2 active load, and solve power flow."""
    net = pn.case4gs()

    bus2_load_idx = net.load.index[net.load.bus == 2]
    if len(bus2_load_idx) == 0:
        raise ValueError("No load found at bus index 2 in case4gs.")

    # Apply the assignment scenario change (active power only).
    net.load.loc[bus2_load_idx, "p_mw"] = load_at_bus2_p_mw
    pp.runpp(net, numba=False)
    return net


def save_single_line_diagram(net, output_filename: str):
    """Save a simple one-line diagram for reporting."""
    plot_net = copy.deepcopy(net)
    try:
        plt.figure(figsize=(9, 6))
        pp_plot.simple_plot(
            plot_net,
            show_plot=False,
            bus_size=1.4,
            line_width=1.6,
            plot_loads=True,
            plot_gens=True,
            plot_sgens=True,
            scale_size=True,
        )
        plt.title("case4gs one-line diagram (base scenario)")
        plt.axis("off")

        output_path = Path(__file__).resolve().parent / output_filename
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nOne-line diagram saved to: {output_path}")
    except Exception as exc:
        print(f"\nCould not export one-line diagram in this environment: {exc}")


# Scenario 1: original case file value
base_net = run_scenario(load_at_bus2_p_mw=200.0)

# Scenario 2: assignment instruction (reduced active load at bus 2)
reduced_net = run_scenario(load_at_bus2_p_mw=80.0)

print("=== Scenario 1: Bus-2 load = 200 MW (base case) ===")
print(base_net.res_bus)

print("\n=== Scenario 2: Bus-2 load = 80 MW (reduced-load case) ===")
print(reduced_net.res_bus)

bus_comparison = pd.DataFrame(
    {
        "vm_pu_base_200MW": base_net.res_bus.vm_pu,
        "vm_pu_reduced_80MW": reduced_net.res_bus.vm_pu,
        "delta_vm_pu": reduced_net.res_bus.vm_pu - base_net.res_bus.vm_pu,
        "va_deg_base_200MW": base_net.res_bus.va_degree,
        "va_deg_reduced_80MW": reduced_net.res_bus.va_degree,
        "delta_va_deg": reduced_net.res_bus.va_degree - base_net.res_bus.va_degree,
    }
).round(6)

line_comparison = pd.DataFrame(
    {
        "loading_pct_base_200MW": base_net.res_line.loading_percent,
        "loading_pct_reduced_80MW": reduced_net.res_line.loading_percent,
        "delta_loading_pct": (
            reduced_net.res_line.loading_percent - base_net.res_line.loading_percent
        ),
    }
).round(6)

print("\n=== Bus voltage/angle comparison (reduced - base) ===")
print(bus_comparison)

print("\n=== Line loading comparison (reduced - base) ===")
print(line_comparison)

save_single_line_diagram(base_net, "case4gs_single_line.png")