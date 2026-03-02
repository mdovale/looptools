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
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
import numpy as np
from looptools.component import Component
from looptools.dimension import Dimension
import looptools.loopmath as lm


class ActuatorComponent(Component):
    """
    PZT actuator model with gain and cutoff frequency.

    Converts s-domain coefficients into z-domain using polynomial_conversion_s_to_z.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Ka_pzt : float
        Actuator gain.
    Fa_pzt : float
        Actuator cutoff frequency (Hz).
    unit : Dimension
        Dimensional unit of the actuator.

    Attributes
    ----------
    Ka_pzt : float
        Gain.
    Fa_pzt : float
        Cutoff frequency.
    """

    def __init__(self, name, sps, Ka_pzt, Fa_pzt, unit):
        self._Fa_pzt = Fa_pzt
        self._Ka_pzt = Ka_pzt
        nume = lm.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0 / (2.0 * np.pi * self._Fa_pzt), 1.0]), sps)
        super().__init__(name, sps, nume, deno, unit=unit)
        self.properties = {
            "Ka_pzt": (lambda: self.Ka_pzt, lambda value: setattr(self, "Ka_pzt", value)),
            "Fa_pzt": (lambda: self.Fa_pzt, lambda value: setattr(self, "Fa_pzt", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = ActuatorComponent.__new__(ActuatorComponent)
        new_obj.__init__(self.name, self.sps, self._Ka_pzt, self._Fa_pzt, self.unit)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Fa_pzt(self):
        return self._Fa_pzt

    @Fa_pzt.setter
    def Fa_pzt(self, value):
        self._Fa_pzt = float(value)
        self.update_component()

    @property
    def Ka_pzt(self):
        return self._Ka_pzt

    @Ka_pzt.setter
    def Ka_pzt(self, value):
        self._Ka_pzt = float(value)
        self.update_component()

    def update_component(self):
        nume = lm.polynomial_conversion_s_to_z(np.array([self._Ka_pzt]), self.sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0 / (2.0 * np.pi * self._Fa_pzt), 1.0]), self.sps)
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)


class ImplicitAccumulatorComponent(Component):
    """
    Continuous-time accumulator modeled in discrete-time domain.

    Approximates a pure integrator with scaling factor of 2pi.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    """

    def __init__(self, name, sps):
        nume = lm.polynomial_conversion_s_to_z(np.array([2.0 * np.pi]), sps)
        deno = lm.polynomial_conversion_s_to_z(np.array([1.0, 0.0]), sps)
        super().__init__(name, sps, nume, deno, unit=Dimension(["rad"], ["Hz"]))

    def __deepcopy__(self, memo):
        new_obj = ImplicitAccumulatorComponent.__new__(ImplicitAccumulatorComponent)
        new_obj.__init__(self.name, self.sps)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj


class LeadLagComponent(Component):
    """
    Lead-Lag controller component.

    Implements a compensator of the form:

        G(s) = K * (s + wz) / (s + wp)

    where `wz = 2π*fz` is the zero frequency and `wp = 2π*fp` is the pole frequency.

    This component is useful for phase compensation: when fz < fp, it behaves like
    a phase lead; when fz > fp, it's a lag; and when fz ≈ fp, it is a gain shaper.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    K : float
        Gain factor.
    fz : float
        Zero frequency in Hz.
    fp : float
        Pole frequency in Hz.
    unit : Dimension, optional
        Dimensional unit of the signal. Defaults to dimensionless.

    Attributes
    ----------
    K : float
        Gain factor.
    fz : float
        Zero frequency.
    fp : float
        Pole frequency.
    """

    def __init__(self, name, sps, K, fz, fp, unit=Dimension(dimensionless=True)):
        self._K = float(K)
        self._fz = float(fz)
        self._fp = float(fp)
        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(name, sps, nume, deno, unit=unit)
        self.properties = {
            "K": (lambda: self.K, lambda value: setattr(self, "K", value)),
            "fz": (lambda: self.fz, lambda value: setattr(self, "fz", value)),
            "fp": (lambda: self.fp, lambda value: setattr(self, "fp", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = LeadLagComponent.__new__(LeadLagComponent)
        new_obj.__init__(self.name, self.sps, self._K, self._fz, self._fp, self.unit)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = float(value)
        self.update_component()

    @property
    def fz(self):
        return self._fz

    @fz.setter
    def fz(self, value):
        self._fz = float(value)
        self.update_component()

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, value):
        self._fp = float(value)
        self.update_component()

    def update_component(self):
        w_z = 2 * np.pi * self._fz
        w_p = 2 * np.pi * self._fp
        nume = np.array([self._K, self._K * w_z])
        deno = np.array([1.0, w_p])
        super().__init__(self.name, self.sps, nume, deno, unit=self.unit)
