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
from functools import partial
from looptools.component import Component
from looptools.dimension import Dimension
import looptools.loopmath as lm


class PIControllerComponent(Component):
    """
    Proportional-Integral controller component, P+I.

    Combines proportional and integral actions into a PI controller with
    bit-shift-based tunable gain.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain as log2(Kp).
    Ki : float
        Integral gain as log2(Ki).

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    """

    def __init__(self, name, sps, Kp, Ki):
        self._Kp = 2 ** float(Kp)
        self._Ki = 2 ** float(Ki)
        P = Component("P", sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        PI = P + I
        super().__init__(name, sps, PI.nume, PI.deno, unit=PI.unit)
        self.properties = {
            "Kp": (lambda: self.Kp, lambda value: setattr(self, "Kp", value)),
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = PIControllerComponent.__new__(PIControllerComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Kp), np.log2(self._Ki))
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value):
        self._Kp = 2 ** float(value)
        self.update_component()

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2 ** float(value)
        self.update_component()

    def update_component(self):
        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        PI = P + I
        super().__init__(self.name, self.sps, PI.nume, PI.deno, unit=PI.unit)


class DoubleIntegratorComponent(Component):
    """
    Second-order integrator, I+II.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Ki : float
        Gain of first integrator (log2 scale).
    Kii : float
        Gain of second integrator (log2 scale).
    extrapolate : tuple(bool, float)
        (Enable extrapolation, transition frequency)

    Attributes
    ----------
    Ki : float
        Gain of the first integrator.
    Kii : float
        Gain of the second integrator.
    """

    def __init__(self, name, sps, Ki, Kii, extrapolate):
        self.extrapolate = extrapolate
        self._Ki = 2 ** float(Ki)
        self._Kii = 2 ** float(Kii)
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(dimensionless=True))
        II = Component("II", sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(dimensionless=True))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)
        DoubleI = I + II
        super().__init__(name, sps, DoubleI.nume, DoubleI.deno, unit=DoubleI.unit)
        self.TE = DoubleI.TE
        self.TE.name = name
        self.TF = partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF)
        self.properties = {
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
            "Kii": (lambda: self.Kii, lambda value: setattr(self, "Kii", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = DoubleIntegratorComponent.__new__(DoubleIntegratorComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Ki), np.log2(self._Kii), self.extrapolate)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2 ** (value)
        self.update_component()

    @property
    def Kii(self):
        return self._Kii

    @Kii.setter
    def Kii(self, value):
        self._Kii = 2 ** (value)
        self.update_component()

    def update_component(self):
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(dimensionless=True))
        II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(dimensionless=True))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)
        DoubleI = I + II
        super().__init__(self.name, self.sps, DoubleI.nume, DoubleI.deno, unit=DoubleI.unit)
        self.TE = DoubleI.TE
        self.TE.name = self.name
        self.TF = partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF)


class PIIControllerComponent(Component):
    """
    Proportional + Integrator + Double Integrator controller component, P+I+II.

    This component models a control law consisting of:
        - A proportional term (P)
        - A first-order integrator (I)
        - A second-order integrator (II)

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate in Hz.
    Kp : float
        Proportional gain (log₂ scale).
    Ki : float
        First integrator gain (log₂ scale).
    Kii : float
        Second integrator gain (log₂ scale).
    extrapolate : tuple(bool, float)
        Tuple (enable_extrapolation, transition_frequency) for the double integrator.

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        First integrator gain.
    Kii : float
        Second integrator gain.
    """

    def __init__(self, name, sps, Kp, Ki, Kii, extrapolate=(False, 1e2)):
        self.sps = sps
        self.extrapolate = extrapolate
        self._Kp = 2 ** float(Kp)
        self._Ki = 2 ** float(Ki)
        self._Kii = 2 ** float(Kii)

        P = Component("P", sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II = Component("II", sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)

        PII = P + I + II
        super().__init__(name, sps, PII.nume, PII.deno, unit=PII.unit)

        self.TE = PII.TE
        self.TE.name = name
        self.TF = partial(lm.add_transfer_function, tf1=P.TF, tf2=partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF))

        self.properties = {
            "Kp": (lambda: self.Kp, lambda value: setattr(self, "Kp", value)),
            "Ki": (lambda: self.Ki, lambda value: setattr(self, "Ki", value)),
            "Kii": (lambda: self.Kii, lambda value: setattr(self, "Kii", value)),
        }

    def __deepcopy__(self, memo):
        new_obj = PIIControllerComponent.__new__(PIIControllerComponent)
        new_obj.__init__(self.name, self.sps, np.log2(self._Kp), np.log2(self._Ki), np.log2(self._Kii), self.extrapolate)
        if getattr(self, "_loop", None) is not None:
            new_obj._loop = self._loop
        return new_obj

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value):
        self._Kp = 2 ** float(value)
        self.update_component()

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value):
        self._Ki = 2 ** float(value)
        self.update_component()

    @property
    def Kii(self):
        return self._Kii

    @Kii.setter
    def Kii(self, value):
        self._Kii = 2 ** float(value)
        self.update_component()

    def update_component(self):
        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        II.TF = partial(II.TF, extrapolate=self.extrapolate[0], f_trans=self.extrapolate[1], power=-2)
        PII = P + I + II
        super().__init__(self.name, self.sps, PII.nume, PII.deno, unit=PII.unit)
        self.TE = PII.TE
        self.TE.name = self.name
        self.TF = partial(lm.add_transfer_function, tf1=P.TF, tf2=partial(lm.add_transfer_function, tf1=I.TF, tf2=II.TF))


class MokuPIDSymbolicController(Component):
    """
    Moku-style symbolic PID controller using P, I, II, and D terms.

    WARNING: II-term causes numerical instabilities at low frequencies, use MokuPIDController instead.

    This component constructs the transfer function symbolically using known corner frequencies
    and proportional gain in dB, as implemented by Liquid Instruments Moku devices.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator (I) crossover frequency [Hz].
    Fc_ii : float or None
        Second integrator (II) crossover frequency [Hz].
    Fc_d : float or None
        Derivative (D) crossover frequency [Hz].
    f_trans : float or None
        Transition frequency for regularization [Hz]. Used to improve numerical behavior of double integrator.

    Properties
    ----------
    Kp_dB, Ki_dB, Kii_dB, Kd_dB : float
        Gains in decibels.
    Fc_i, Fc_ii, Fc_d : float
        Crossover frequencies in Hz.
    """

    def __init__(self, name, sps, Kp_dB, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        self.name = name
        self.sps = sps
        self.f_trans = f_trans

        self._Kp_dB = Kp_dB
        self._Fc_i = Fc_i
        self._Fc_ii = Fc_ii
        self._Fc_d = Fc_d

        self.update_component()

        self.properties = {
            "Kp_dB": (lambda: self.Kp_dB, lambda value: setattr(self, "Kp_dB", value)),
            "Ki_dB": (lambda: self.Ki_dB, lambda value: setattr(self, "Ki_dB", value)),
            "Kii_dB": (lambda: self.Kii_dB, lambda value: setattr(self, "Kii_dB", value)),
            "Kd_dB": (lambda: self.Kd_dB, lambda value: setattr(self, "Kd_dB", value)),
            "Fc_i": (lambda: self.Fc_i, lambda value: setattr(self, "Fc_i", value)),
            "Fc_ii": (lambda: self.Fc_ii, lambda value: setattr(self, "Fc_ii", value)),
            "Fc_d": (lambda: self.Fc_d, lambda value: setattr(self, "Fc_d", value)),
        }

    def update_component(self):
        tf_str = self.moku_pid_tf_string(
            self.sps,
            Kp_dB=self._Kp_dB,
            Ki_dB=None if self._Fc_i is not None else lm.log2_gain_to_db(np.log2(self._Ki)) if hasattr(self, "_Ki") else None,
            Kii_dB=None if self._Fc_ii is not None else lm.log2_gain_to_db(np.log2(self._Kii)) if hasattr(self, "_Kii") else None,
            Kd_dB=None if self._Fc_d is not None else lm.log2_gain_to_db(np.log2(self._Kd)) if hasattr(self, "_Kd") else None,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )

        super().__init__(self.name, tf=tf_str, sps=self.sps, unit=Dimension(["cycle"], ["s", "rad"]))

        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2
        self._Ki = 0.0 if self._Fc_i is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_i, kind="I")
        self._Kii = 0.0 if self._Fc_ii is None else 2 ** lm.gain_for_crossover_frequency(0.0, self.sps, self._Fc_ii, kind="II")

        if self._Fc_d is not None:
            omega_d = 2 * np.pi * self._Fc_d / self.sps
            mag = abs((1 - np.exp(-1j * omega_d)) / (1 + np.exp(-1j * omega_d)))
            self._Kd = self._Kp / mag
        elif hasattr(self, "_Kd"):
            pass
        else:
            self._Kd = 0.0

    @staticmethod
    def moku_pid_tf_string(sps, Kp_dB=0.0, Ki_dB=None, Kii_dB=None, Kd_dB=None, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        Kp_log2 = lm.db_to_log2_gain(Kp_dB)
        Kp = 2 ** Kp_log2

        if Fc_i is not None:
            Ki_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_i, kind="I")
        elif Ki_dB is not None:
            Ki_log2 = lm.db_to_log2_gain(Ki_dB)
        else:
            Ki_log2 = float("-inf")

        if Fc_ii is not None:
            Kii_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_ii, kind="II")
        elif Kii_dB is not None:
            Kii_log2 = lm.db_to_log2_gain(Kii_dB)
        else:
            Kii_log2 = float("-inf")

        if Fc_d is not None:
            Kd_log2 = lm.gain_for_crossover_frequency(0.0, sps, Fc_d, kind="D")
        elif Kd_dB is not None:
            Kd_log2 = lm.db_to_log2_gain(Kd_dB)
        else:
            Kd_log2 = float("-inf")

        Ki = 0.0 if not np.isfinite(Ki_log2) else 2 ** Ki_log2
        Kii = 0.0 if not np.isfinite(Kii_log2) else 2 ** Kii_log2
        Kd = 0.0 if not np.isfinite(Kd_log2) else 2 ** Kd_log2

        if f_trans is not None:
            delta = (2 * np.pi * f_trans / sps) ** 2
            Kii_term = f"({Kii:.6g})/((1 - z**-1)**2 + {delta:.3e})"
        else:
            Kii_term = f"({Kii:.6g})/(1 - z**-1)**2"

        return (
            f"{Kp:.6g} * (1"
            f" + ({Ki:.6g})/(1 - z**-1)"
            f" + {Kii_term}"
            f" + ({Kd:.6g})*(1 - z**-1)/(1 + z**-1))"
        )

    @property
    def Kp_dB(self):
        return self._Kp_dB

    @Kp_dB.setter
    def Kp_dB(self, value):
        self._Kp_dB = float(value)
        self.update_component()

    @property
    def Ki_dB(self):
        return None if self._Ki == 0.0 else lm.log2_gain_to_db(np.log2(self._Ki))

    @Ki_dB.setter
    def Ki_dB(self, value):
        self._Fc_i = None
        self._Ki = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Kii_dB(self):
        return None if self._Kii == 0.0 else lm.log2_gain_to_db(np.log2(self._Kii))

    @Kii_dB.setter
    def Kii_dB(self, value):
        self._Fc_ii = None
        self._Kii = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Kd_dB(self):
        return None if self._Kd == 0.0 else lm.log2_gain_to_db(np.log2(self._Kd))

    @Kd_dB.setter
    def Kd_dB(self, value):
        self._Fc_d = None
        self._Kd = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Fc_i(self):
        return self._Fc_i

    @Fc_i.setter
    def Fc_i(self, value):
        self._Fc_i = float(value)
        self.update_component()

    @property
    def Fc_ii(self):
        return self._Fc_ii

    @Fc_ii.setter
    def Fc_ii(self, value):
        self._Fc_ii = float(value)
        self.update_component()

    @property
    def Fc_d(self):
        return self._Fc_d

    @Fc_d.setter
    def Fc_d(self, value):
        self._Fc_d = float(value)
        self.update_component()

    def __deepcopy__(self, memo):
        return MokuPIDSymbolicController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )


class MokuPIDController(Component):
    """
    Moku-style PID controller with P, optional I, II, and D terms using symbolic structure
    and extrapolated low-frequency behavior for numerical stability.

    Parameters
    ----------
    name : str
        Component name.
    sps : float
        Sample rate [Hz].
    Kp_dB : float
        Proportional gain in dB.
    Fc_i : float or None
        First integrator crossover frequency [Hz]. If None, I is omitted.
    Fc_ii : float or None
        Second integrator crossover frequency [Hz]. If None, II is omitted.
    Fc_d : float or None
        Derivative crossover frequency [Hz]. If None, D is omitted.
    f_trans : float
        Transition frequency below which extrapolation is applied [Hz].
    """

    def __init__(self, name, sps, Kp_dB, Fc_i=None, Fc_ii=None, Fc_d=None, f_trans=None):
        self.name = name
        self.sps = sps
        self.f_trans = f_trans
        self._Kp_dB = Kp_dB
        self._Fc_i = Fc_i
        self._Fc_ii = Fc_ii
        self._Fc_d = Fc_d

        self.update_component()

        self.properties = {
            "Kp_dB": (lambda: self.Kp_dB, lambda value: setattr(self, "Kp_dB", value)),
            "Ki_dB": (lambda: self.Ki_dB, lambda value: setattr(self, "Ki_dB", value)),
            "Kii_dB": (lambda: self.Kii_dB, lambda value: setattr(self, "Kii_dB", value)),
            "Kd_dB": (lambda: self.Kd_dB, lambda value: setattr(self, "Kd_dB", value)),
            "Fc_i": (lambda: self.Fc_i, lambda value: setattr(self, "Fc_i", value)),
            "Fc_ii": (lambda: self.Fc_ii, lambda value: setattr(self, "Fc_ii", value)),
            "Fc_d": (lambda: self.Fc_d, lambda value: setattr(self, "Fc_d", value)),
        }

    def update_component(self):
        Kp_log2 = lm.db_to_log2_gain(self._Kp_dB)
        self._Kp = 2 ** Kp_log2

        P = Component("P", self.sps, np.array([self._Kp]), np.array([1.0]), unit=Dimension(["cycle"], ["s", "rad"]))
        components = [P]

        if self._Fc_i is not None and self._Fc_ii is None:
            self._Ki = 2 ** lm.gain_for_crossover_frequency(Kp_log2, self.sps, self._Fc_i, kind="I")
            I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=P.unit)
            components.append(I)
        else:
            self._Ki = None

        if self._Fc_ii is not None and self._Fc_i is not None:
            i_log2, ii_log2 = lm.gain_for_crossover_frequency(Kp_log2, self.sps, [self._Fc_i, self._Fc_ii], kind="II")
            self._Ki, self._Kii = 2 ** i_log2, 2 ** ii_log2
            I = Component("I", self.sps, np.array([self._Ki]), np.array([1.0, -1.0]), unit=P.unit)
            components.append(I)

            II = Component("II", self.sps, np.array([self._Kii]), np.array([1.0, -2.0, 1.0]), unit=P.unit)
            if self.f_trans is not None:
                II.TF = partial(II.TF, extrapolate=True, f_trans=self.f_trans, power=-2)
            components.append(II)
        else:
            self._Ki = None
            self._Kii = None

        if self._Fc_d is not None:
            self._Kd = 2 ** lm.gain_for_crossover_frequency(Kp_log2, self.sps, self._Fc_d, kind="D")
            D = Component("D", self.sps, np.array([self._Kd, -self._Kd]), np.array([1.0, 0.0, 1.0]), unit=P.unit)
            components.append(D)
        else:
            self._Kd = None

        PID = components[0]
        for comp in components[1:]:
            PID = PID + comp

        super().__init__(self.name, self.sps, PID.nume, PID.deno, unit=PID.unit)
        self.TF = PID.TF
        self.TE = PID.TE
        self.TE.name = self.name

    def __deepcopy__(self, memo):
        return MokuPIDController(
            name=self.name,
            sps=self.sps,
            Kp_dB=self._Kp_dB,
            Fc_i=self._Fc_i,
            Fc_ii=self._Fc_ii,
            Fc_d=self._Fc_d,
            f_trans=self.f_trans,
        )

    @property
    def Kp_dB(self):
        return self._Kp_dB

    @Kp_dB.setter
    def Kp_dB(self, value):
        self._Kp_dB = float(value)
        self.update_component()

    @property
    def Ki_dB(self):
        return None if self._Ki is None else lm.log2_gain_to_db(np.log2(self._Ki))

    @Ki_dB.setter
    def Ki_dB(self, value):
        self._Fc_i = None
        self._Ki = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Kii_dB(self):
        return None if self._Kii is None else lm.log2_gain_to_db(np.log2(self._Kii))

    @Kii_dB.setter
    def Kii_dB(self, value):
        self._Fc_ii = None
        self._Kii = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Kd_dB(self):
        return None if self._Kd is None else lm.log2_gain_to_db(np.log2(self._Kd))

    @Kd_dB.setter
    def Kd_dB(self, value):
        self._Fc_d = None
        self._Kd = 2 ** lm.db_to_log2_gain(value)
        self.update_component()

    @property
    def Fc_i(self):
        return self._Fc_i

    @Fc_i.setter
    def Fc_i(self, value):
        self._Fc_i = float(value) if value is not None else None
        self.update_component()

    @property
    def Fc_ii(self):
        return self._Fc_ii

    @Fc_ii.setter
    def Fc_ii(self, value):
        self._Fc_ii = float(value) if value is not None else None
        self.update_component()

    @property
    def Fc_d(self):
        return self._Fc_d

    @Fc_d.setter
    def Fc_d(self, value):
        self._Fc_d = float(value) if value is not None else None
        self.update_component()
