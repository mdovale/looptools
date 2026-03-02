"""looptools: control loop modeling and analysis."""

import sys

# Backward compatibility: expose loops submodules at old paths
import looptools.loops.pll
import looptools.loops.mokulaserlock
import looptools.loops.nprolaserlock

sys.modules["looptools.pll"] = looptools.loops.pll
sys.modules["looptools.mokulaserlock"] = looptools.loops.mokulaserlock
sys.modules["looptools.nprolaserlock"] = looptools.loops.nprolaserlock
