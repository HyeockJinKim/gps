import lib.RX as RX
import matplotlib.pyplot as plt



OSsig = RX.Simulation(0.02, 0)
OSsig.start()


PACsig = RX.Simulation(0.02, 1)
PACsig.start()


ENCsig = RX.Simulation(0.02, 2)
ENCsig.start()