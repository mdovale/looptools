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
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

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
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

from typing import Iterable, Optional


def _validate_units(units: Iterable, name: str) -> None:
    """Raise TypeError if any unit is not a string."""
    for u in units:
        if not isinstance(u, str):
            raise TypeError(f"{name} must contain strings, got {type(u).__name__}: {u!r}")


class Dimension:
    """
    Represents a symbolic physical dimension using numerator and denominator units.

    This class allows basic algebraic manipulation of unit-like structures,
    such as multiplying and reducing dimensional expressions symbolically.
    It is not tied to a specific unit system, but works with symbolic strings.
    """

    def __init__(
        self,
        numes: Optional[Iterable[str]] = None,
        denos: Optional[Iterable[str]] = None,
        dimensionless: bool = False,
    ) -> None:
        """
        Initialize a Dimension object.

        Parameters
        ----------
        numes : list of str, optional
            List of numerator units (e.g., ['m', 's']). Default is empty.
        denos : list of str, optional
            List of denominator units (e.g., ['s', 'kg']). Default is empty.
        dimensionless : bool, optional
            If True, initializes as a dimensionless unit (i.e., 1). Default is False.
        """
        if numes is None:
            numes = []
        if denos is None:
            denos = []

        if dimensionless:
            self._numes: tuple[str, ...] = ()
            self._denos: tuple[str, ...] = ()
        else:
            numes_list = list(numes)
            denos_list = list(denos)
            _validate_units(numes_list, "numes")
            _validate_units(denos_list, "denos")
            self._reduction(numes_list, denos_list)
            self._numes = tuple(numes_list)
            self._denos = tuple(denos_list)

    @property
    def numes(self) -> tuple[str, ...]:
        """Numerator units (immutable)."""
        return self._numes

    @property
    def denos(self) -> tuple[str, ...]:
        """Denominator units (immutable)."""
        return self._denos

    def __mul__(self, other: Dimension) -> Dimension:
        """
        Multiply two Dimension objects.

        Parameters
        ----------
        other : Dimension
            Another Dimension instance to multiply with this one.

        Returns
        -------
        Dimension
            A new Dimension object representing the product of the two.

        Notes
        -----
        The multiplication follows symbolic dimensional algebra and
        automatically reduces common units between numerators and denominators.
        """
        self_numes = list(self.numes)
        self_denos = list(self.denos)
        other_numes = list(other.numes)
        other_denos = list(other.denos)

        self._reduction(self_numes, other_denos)
        self._reduction(self_denos, other_numes)

        self_numes.extend(other_numes)
        self_denos.extend(other_denos)

        return Dimension(self_numes, self_denos)

    def __truediv__(self, other: Dimension) -> Dimension:
        """
        Divide this Dimension by another (self / other).

        Returns
        -------
        Dimension
            A new Dimension representing the quotient.
        """
        return self * Dimension(other.denos, other.numes)

    def __eq__(self, other: object) -> bool:
        """Return True if self and other represent the same dimension."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return sorted(self.numes) == sorted(other.numes) and sorted(self.denos) == sorted(
            other.denos
        )

    def __hash__(self) -> int:
        """Hash based on canonical (sorted) form for consistency with __eq__."""
        return hash((tuple(sorted(self.numes)), tuple(sorted(self.denos))))

    def __repr__(self) -> str:
        s = self.unit_string()
        return f"Dimension({s!r})" if s else "Dimension(dimensionless=True)"

    def _reduction(self, numes: list[str], denos: list[str]) -> None:
        """
        Simplify numerator and denominator unit lists by cancelling common terms.

        Parameters
        ----------
        numes : list of str
            Numerator units (modified in-place).
        denos : list of str
            Denominator units (modified in-place).

        Notes
        -----
        This method reduces the symbolic expression by cancelling equal terms from
        both numerator and denominator.
        """
        commons = list(set(numes) & set(denos))
        for common in commons:
            n_common_numes = numes.count(common)
            n_common_denos = denos.count(common)
            n_common = min(n_common_numes, n_common_denos)
            for _ in range(n_common):
                numes.remove(common)
                denos.remove(common)

    def unit_string(self) -> str:
        """
        Generate a string representation of the dimension.

        Returns
        -------
        str
            A symbolic unit string in the form 'unit1*unit2/...', or empty string if dimensionless.
        """
        numerator = "*".join(self.numes) if self.numes else ""
        denominator = "*".join(self.denos) if self.denos else ""
        if numerator and denominator:
            return numerator + "/" + denominator
        elif numerator:
            return numerator
        elif denominator:
            return "1/" + denominator
        else:
            return ""

    def show(self) -> None:
        """
        Print the dimension as a symbolic unit string.

        Examples
        --------
        >>> d = Dimension(['m'], ['s'])
        >>> d.show()
        [m/s]
        """
        print(f"[{self.unit_string()}]")