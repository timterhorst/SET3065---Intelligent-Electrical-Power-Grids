from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

pd.options.display.max_columns = None

THIS_DIR = Path(__file__).resolve().parent
net = pp.from_excel(THIS_DIR / "ieee9-wind.xlsx")

# Running the optimization power flow problem
pp.runopp(net, init='pf',verbose=False)
print(f"The total cost is: {net.res_cost}.")

# Plot bus voltages
fig, ax1 = plt.subplots(1, 1)
voltages = net.res_bus.vm_pu.to_list()
ax1.bar(range(len(voltages)), voltages)
ax1.set_xlabel("Bus Index")
ax1.set_ylabel("Voltage (pu)")
fig.tight_layout()
plot_path = THIS_DIR / "opf_bus_voltages.png"
fig.savefig(plot_path, dpi=150)
print(f"Voltage plot saved to: {plot_path}")

# TO DO: plot line losses 
