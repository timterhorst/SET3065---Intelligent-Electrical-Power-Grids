import pandapower as pp
import pandapower.networks as pn

net = pn.case4gs()

pp.runpp(net, numba=False)

print(net.res_bus)