
# looptools

A Python library for modeling, analyzing, and simulating feedback control loops, with a focus on scientific and engineering applications such as laser locking, electronics, and precision instrumentation.

---

## 🔧 What is `looptools`?

`looptools` provides an object-oriented framework for building and analyzing control loops as directed graphs of signal-processing components. It supports:

- Modular design of control systems using Python classes and NumPy/Scipy tools
- Construction of linear time-invariant (LTI) models via symbolic transfer functions or tabulated data
- Frequency-domain and time-domain simulation of system behavior
- Visualization and analysis of loop stability, gain, noise propagation, and performance
- Extensibility through subclassing and dynamic graph generation

The library is designed with scientific control systems in mind—especially applications in optics, interferometry, and laser metrology.

---

## 🧩 Key Features

- **Component-based modeling**: Each system block is a `Component` with defined transfer functions, inputs, and outputs.
- **Flexible interconnection**: Loops are defined by connecting components into directed graphs, not just serial chains.
- **Simulation support**: Built-in methods to simulate step responses, noise propagation, and open/closed-loop behavior.
- **Plotting utilities**: Visualize Bode plots, noise budgets, signal flow graphs, and more.
- **Extensible**: Easily define custom blocks, auto-generate loop structures, and integrate with hardware configurations.

---

## 🚀 Quick Start

### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from looptools.component import Component
from looptools.components import PIControllerComponent
from looptools.loop import LOOP

sps = 80e6 # Loop update frequency (Hz)
frfr = np.logspace(np.log10(1e2), np.log10(40e6), int(1e5)) # Fourier frequency array (Hz)
frfr = frfr[0:-1]

# Define components
plant = Component("Plant", sps, tf="1/(s^2 + 10*s - 20)")
sensor = Component("Sensor", sps, tf="s/(s^2 + 2*s - 3.5)")
controller = PIControllerComponent("Controller", sps, Kp=3, Ki=-3)

# Build loop
loop = LOOP(sps, [plant, sensor, controller], name='My Loop')

# Visualize
loop.block_diagram(dpi=200)

# Analyze
fig, ax = loop.bode_plot(frfr)
plt.show()

fig, ax = loop.nyquist_plot(frfr, which='G', logy=True, critical_point=True)
plt.show()
```

---

## 🧪 Specialized Subclasses

- `NPROLaserLock(LOOP)`: Model for laser frequency stabilization of an NPRO laser using PZT and temperature control loops with digital PLL (i.e., phasemeter) readout.

---

## 📚 Documentation

In-depth API documentation, tutorials, and example notebooks coming soon.

---

## 💡 Design Philosophy

- Favor explicit, graph-based design over implicit signal paths
- Maintain clarity between loop structure and numerical simulation
- Support rapid prototyping of complex control systems
- Blend symbolic (transfer function) and numeric (data-driven) components

---

## 👥 Contributing

Contributions are welcome! We are especially interested in:

- New component definitions (hardware models, filters, sensors)
- Visualization tools for loop inspection
- Test coverage and validation scripts
- Real-world loop configurations from physics labs

---

## 📜 License

GNU GENERAL PUBLIC LICENSE Version 3.

---

## 🛰 Authors & Acknowledgments

Developed by Miguel Dovale, based on an initial implementation by Kohei Yamamoto.

---
