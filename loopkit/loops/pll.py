# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.

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

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting this software to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import loopkit.components as lc
from loopkit.component import Component
from loopkit.loop import LOOP

logger = logging.getLogger(__name__)

# Sentinel for "no exclusions" when using the but parameter.
_BUT_NONE: Tuple[None, ...] = (None,)


def _validate_pll_params(
    sps: float,
    amp: float,
    cshift: int,
    klf: float,
    kp: int,
    ki: int,
    n_reg: int,
    but: Tuple[Union[str, None], ...],
) -> None:
    """Validate PLL constructor parameters. Raises ValueError or TypeError on invalid input."""
    if not isinstance(sps, (int, float)) or sps <= 0:
        raise ValueError(f"sps must be a positive number, got {sps!r}")
    if not isinstance(amp, (int, float)) or amp <= 0:
        raise ValueError(f"Amp must be positive, got {amp!r}")
    if not isinstance(cshift, int) or cshift < 0:
        raise ValueError(f"Cshift must be a non-negative int, got {cshift!r}")
    if not isinstance(klf, (int, float)) or klf <= 0:
        raise ValueError(f"Klf must be positive, got {klf!r}")
    if not isinstance(kp, int) or kp < 0:
        raise ValueError(f"Kp must be a non-negative int, got {kp!r}")
    if not isinstance(ki, int) or ki < 0:
        raise ValueError(f"Ki must be a non-negative int, got {ki!r}")
    if not isinstance(n_reg, int) or n_reg < 0:
        raise ValueError(f"n_reg must be a non-negative int, got {n_reg!r}")
    if isinstance(but, str):
        raise TypeError("but must be a sequence of component names, not a string")
    if not isinstance(but, (list, tuple)):
        raise TypeError(f"but must be a sequence, got {type(but).__name__}")
    valid_names = frozenset(("PD", "LPF", "Gain", "PI", "PA", "LUT", "DSP"))
    for item in but:
        if item is not None and item not in valid_names:
            raise ValueError(
                f"but must contain only {sorted(valid_names)} or None, got {item!r}"
            )


class PLL(LOOP):
    """
    Phase-Locked Loop (PLL) simulation loop subclass of LOOP.

    This class models a digital PLL system using a chain of configurable
    components such as a phase detector, low-pass filter, gain, PI controller,
    and output driver stages. This model is designed to match common
    FPGA/DSP-based implementations.
    """

    def __init__(
        self,
        sps: float,
        Amp: float,
        Cshift: int,
        Klf: float,
        Kp: int,
        Ki: int,
        *,
        twostages: bool = True,
        n_reg: int = 10,
        but: Optional[Sequence[Union[str, None]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a PLL loop.

        Parameters
        ----------
        sps : float
            System sampling frequency in Hz.
        Amp : float
            Normalized input signal amplitude, defined as Vpk / Vpp_adc.
        Cshift : int
            Number of bit shifts applied in the gain stage (left shift).
        Klf : float
            Loop filter gain (typically normalized).
        Kp : int
            Proportional gain of PI controller, interpreted as bit shifts.
        Ki : int
            Integral gain of PI controller, interpreted as bit shifts.
        twostages : bool, optional
            If True, use a two-stage IIR loop filter (default is True).
        n_reg : int, optional
            Number of DSP delay registers to insert (default is 10).
        but : sequence of str or None, optional
            Component names to exclude from the loop. Use (None,) or [None]
            for no exclusions. Default is no exclusions.
        name : str, optional
            Loop name. Default is 'PLL'.
        """
        _validate_pll_params(
            sps, Amp, Cshift, Klf, Kp, Ki, n_reg,
            _BUT_NONE if but is None else but,
        )
        _but: Tuple[Union[str, None], ...] = (
            _BUT_NONE if but is None else tuple(x for x in but)
        )

        super().__init__(sps, name=name or "PLL")

        self._but = _but
        self._twostages = bool(twostages)
        self._amp = float(Amp)
        self._cshift = int(Cshift)
        self._klf = float(Klf)
        self._kp = int(Kp)
        self._ki = int(Ki)
        self._n_reg = int(n_reg)

        # Phase detector
        if "PD" not in _but:
            self.add_component(lc.PDComponent("PD", self.sps, self._amp))

        # IIR LF
        if "LPF" not in _but:
            if self._twostages:
                self.add_component(
                    lc.TwoStageLPFComponent("LPF", self.sps, self._klf)
                )
            else:
                self.add_component(lc.LPFComponent("LPF", self.sps, self._klf))

        # Gain
        if "Gain" not in _but:
            self.add_component(
                lc.RightBitShiftComponent("Gain", self.sps, self._cshift)
            )

        # PI controller
        if "PI" not in _but:
            self.add_component(
                lc.PIControllerComponent("PI", self.sps, self._kp, self._ki)
            )

        # PA
        if "PA" not in _but:
            self.add_component(lc.PAComponent("PA", self.sps))

        # LUT
        if "LUT" not in _but:
            self.add_component(lc.LUTComponent("LUT", self.sps))

        # DSP delay (= register delay)
        if "DSP" not in _but:
            self.add_component(
                lc.DSPDelayComponent("DSP", self.sps, self._n_reg)
            )

        if _but != _BUT_NONE:
            logger.warning(
                "The following components are not included in the loop: %s",
                _but,
            )

        self.update()
        self.register_component_properties()

    @property
    def but(self) -> Tuple[Union[str, None], ...]:
        """Component names excluded from the loop (immutable)."""
        return self._but

    @property
    def twostages(self) -> bool:
        """Whether a two-stage IIR loop filter is used."""
        return self._twostages

    @property
    def Amp(self) -> float:
        """Normalized input signal amplitude."""
        return self._amp

    @property
    def Cshift(self) -> int:
        """Number of bit shifts in the gain stage."""
        return self._cshift

    @property
    def Klf(self) -> float:
        """Loop filter gain."""
        return self._klf

    @property
    def Kp(self) -> int:
        """Proportional gain of PI controller (bit shifts)."""
        return self._kp

    @property
    def Ki(self) -> int:
        """Integral gain of PI controller (bit shifts)."""
        return self._ki

    @property
    def n_reg(self) -> int:
        """Number of DSP delay registers."""
        return self._n_reg

    def __deepcopy__(self, memo: dict) -> PLL:
        new_obj: PLL = PLL.__new__(PLL)
        new_obj.__init__(
            self.sps,
            self._amp,
            self._cshift,
            self._klf,
            self._kp,
            self._ki,
            twostages=self._twostages,
            n_reg=self._n_reg,
            but=self._but,
        )
        new_obj.callbacks = self.callbacks
        return new_obj

    def show_all_te(self) -> None:
        """
        Display the transfer elements of all components in the loop.

        Prints the internal transfer element (TE) representation of each
        component, as well as the overall open-loop forward path (Gc),
        feedback path (Hc), and error function (Ec).
        """
        for comp in self.components_dict.values():
            print(
                f"=== transfer function of {comp.name} === {comp.TE}"
            )
        print(f"=== transfer function of G === {self.Gc.TE}")
        print(f"=== transfer function of H === {self.Hc.TE}")
        print(f"=== transfer function of E === {self.Ec.TE}")

    def point_to_point_component(
        self,
        _from: Optional[str] = None,
        _to: Optional[str] = None,
        suppression: bool = False,
        view: bool = False,
    ) -> Component:
        """
        Compute the transfer element between two components in the loop.

        Parameters
        ----------
        _from : str, optional
            Name of the starting component.
        _to : str, optional
            Name of the stopping component. This component is *not* included.
        suppression : bool, optional
            If True, apply loop suppression factor 1 / (1 + G). Default is False.
        view : bool, optional
            If True, print details about the path and resulting transfer element.

        Returns
        -------
        Component
            Transfer element between the specified components.
        """
        return super().point_to_point_component(
            _from, _to, suppression=suppression, view=view
        )

    def point_to_point_tf(
        self,
        f: ArrayLike,
        _from: str,
        _to: Optional[str] = None,
        suppression: bool = False,
        view: bool = False,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """
        Compute the frequency response (transfer function) between two loop components.

        Parameters
        ----------
        f : array_like
            Array of Fourier frequencies in Hz.
        _from : str
            Name of the starting component.
        _to : str, optional
            Name of the stopping component. This component is *not* included.
        suppression : bool, optional
            If True, apply loop suppression factor 1 / (1 + G). Default is False.
        view : bool, optional
            If True, print details about the path and resulting transfer function.

        Returns
        -------
        ndarray
            Complex frequency response of the transfer function between components.
        """
        return super().point_to_point_tf(
            f, _from, _to=_to, suppression=suppression, view=view
        )
