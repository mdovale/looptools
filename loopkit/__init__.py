"""loopkit: control loop modeling and analysis."""

import sys

# Backward compatibility: expose loops submodules at old paths
import loopkit.loops.pll
import loopkit.loops.mokulaserlock
import loopkit.loops.nprolaserlock

sys.modules["loopkit.pll"] = loopkit.loops.pll
sys.modules["loopkit.mokulaserlock"] = loopkit.loops.mokulaserlock
sys.modules["loopkit.nprolaserlock"] = loopkit.loops.nprolaserlock
