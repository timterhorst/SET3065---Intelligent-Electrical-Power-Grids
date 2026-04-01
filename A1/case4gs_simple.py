import ast
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
    """Save an annotated one-line diagram with bus voltages, line loadings, and generator/load info."""
    plot_net = copy.deepcopy(net)
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Ensure geodata exists for coordinate access
        if "geo" not in plot_net.bus.columns or plot_net.bus.geo.isna().all():
            pp_plot.create_generic_coordinates(plot_net)
        
        # Create the base plot
        collections = pp_plot.simple_plot(
            plot_net,
            ax=ax,
            show_plot=False,
            bus_size=1.6,
            line_width=2.0,
            plot_loads=True,
            plot_gens=True,
            plot_sgens=True,
            scale_size=True,
        )
        
        # Extract bus coordinates from geodata
        bus_coords = {}
        for bus_idx in net.bus.index:
            geo = plot_net.bus.loc[bus_idx, "geo"]
            if pd.notna(geo):
                # Handle different geo formats (string or tuple)
                if isinstance(geo, str):
                    try:
                        coords = ast.literal_eval(geo)
                    except (ValueError, SyntaxError):
                        # If literal_eval fails, try parsing as comma-separated values
                        parts = geo.strip("()[]").split(",")
                        coords = [float(p.strip()) for p in parts]
                else:
                    coords = geo
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    bus_coords[bus_idx] = (float(coords[0]), float(coords[1]))
        
        # Annotate buses with voltage magnitude and angle
        for bus_idx in net.bus.index:
            if bus_idx in bus_coords:
                x, y = bus_coords[bus_idx]
                vm_pu = net.res_bus.loc[bus_idx, "vm_pu"]
                va_deg = net.res_bus.loc[bus_idx, "va_degree"]
                # Add bus number and voltage
                ax.annotate(
                    f"Bus {bus_idx}\nV={vm_pu:.3f} pu\nθ={va_deg:.2f}°",
                    xy=(x, y),
                    xytext=(12, 12),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="blue", linewidth=1.5),
                    ha="left",
                    va="bottom",
                )
        
        # Annotate lines with loading percentage
        for line_idx in net.line.index:
            from_bus = net.line.loc[line_idx, "from_bus"]
            to_bus = net.line.loc[line_idx, "to_bus"]
            if from_bus in bus_coords and to_bus in bus_coords:
                x1, y1 = bus_coords[from_bus]
                x2, y2 = bus_coords[to_bus]
                # Midpoint of the line
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                loading_pct = net.res_line.loc[line_idx, "loading_percent"]
                p_from_mw = net.res_line.loc[line_idx, "p_from_mw"]
                p_to_mw = net.res_line.loc[line_idx, "p_to_mw"]
                # Color annotation based on loading
                color = "green" if loading_pct < 50 else "orange" if loading_pct < 80 else "red"
                ax.annotate(
                    f"L{line_idx}: {loading_pct:.1f}%\n{p_from_mw:.1f}→{p_to_mw:.1f} MW",
                    xy=(x, y),
                    xytext=(0, -30),
                    textcoords="offset points",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2),
                    ha="center",
                    va="top",
                    color=color,
                    weight="bold" if loading_pct > 80 else "normal",
                )
        
        # Annotate generators
        if len(net.gen) > 0:
            for gen_idx in net.gen.index:
                bus_idx = net.gen.loc[gen_idx, "bus"]
                if bus_idx in bus_coords:
                    x, y = bus_coords[bus_idx]
                    p_mw = net.res_gen.loc[gen_idx, "p_mw"]
                    q_mvar = net.res_gen.loc[gen_idx, "q_mvar"]
                    ax.annotate(
                        f"Gen@{bus_idx}\nP={p_mw:.1f} MW\nQ={q_mvar:.1f} MVAr",
                        xy=(x, y),
                        xytext=(18, -18),
                        textcoords="offset points",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.85, edgecolor="darkgreen", linewidth=1.5),
                        ha="left",
                        va="top",
                    )
        
        # Annotate loads
        if len(net.load) > 0:
            for load_idx in net.load.index:
                bus_idx = net.load.loc[load_idx, "bus"]
                if bus_idx in bus_coords:
                    x, y = bus_coords[bus_idx]
                    p_mw = net.res_load.loc[load_idx, "p_mw"]
                    q_mvar = net.res_load.loc[load_idx, "q_mvar"]
                    ax.annotate(
                        f"Load@{bus_idx}\nP={p_mw:.1f} MW\nQ={q_mvar:.1f} MVAr",
                        xy=(x, y),
                        xytext=(-18, -18),
                        textcoords="offset points",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.85, edgecolor="darkred", linewidth=1.5),
                        ha="right",
                        va="top",
                    )
        
        plt.title("case4gs One-Line Diagram with Annotations (Base Scenario)", fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        
        output_path = Path(__file__).resolve().parent / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nAnnotated one-line diagram saved to: {output_path}")
    except Exception as exc:
        print(f"\nCould not export one-line diagram in this environment: {exc}")
        import traceback
        traceback.print_exc()


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