import pandas as pd
import pandapower as pp
import pandapower.networks as pn


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