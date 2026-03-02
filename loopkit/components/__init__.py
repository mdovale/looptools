# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Concrete Component implementations for control loop modeling."""

from loopkit.components._opamp import set_opamp_parameters, OpAmp_dict
from loopkit.components.gains import MultiplierComponent, RightBitShiftComponent
from loopkit.components.filters import (
    LPFComponent,
    ButterworthLPFComponent,
    TwoStageLPFComponent,
    IIRFilterComponent,
    iir_from_sos,
)
from loopkit.components.controllers import (
    PIControllerComponent,
    DoubleIntegratorComponent,
    PIIControllerComponent,
    MokuPIDSymbolicController,
    MokuPIDController,
)
from loopkit.components.delay import DSPDelayComponent
from loopkit.components.pll_components import PDComponent, PAComponent, LUTComponent
from loopkit.components.actuators import ActuatorComponent, ImplicitAccumulatorComponent, LeadLagComponent

__all__ = [
    "set_opamp_parameters",
    "OpAmp_dict",
    "PDComponent",
    "MultiplierComponent",
    "RightBitShiftComponent",
    "LPFComponent",
    "ButterworthLPFComponent",
    "TwoStageLPFComponent",
    "IIRFilterComponent",
    "iir_from_sos",
    "PIControllerComponent",
    "DoubleIntegratorComponent",
    "PIIControllerComponent",
    "MokuPIDSymbolicController",
    "MokuPIDController",
    "DSPDelayComponent",
    "PAComponent",
    "LUTComponent",
    "ActuatorComponent",
    "ImplicitAccumulatorComponent",
    "LeadLagComponent",
]
