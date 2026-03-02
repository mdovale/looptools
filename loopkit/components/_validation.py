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
"""Shared validation helpers for component parameters."""
from __future__ import annotations

import numbers
from typing import Tuple


def _validate_positive(name: str, value: float, *, strict: bool = True) -> float:
    """Validate and coerce to positive float. Raises TypeError or ValueError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if strict and v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _validate_non_negative(name: str, value: float) -> float:
    """Validate and coerce to non-negative float. Raises TypeError or ValueError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


def _validate_int_non_negative(name: str, value: int | float) -> int:
    """Validate and coerce to non-negative integer. Raises TypeError or ValueError."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool")
    if not isinstance(value, (int, numbers.Integral, numbers.Real)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = int(value)
    if v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


def _validate_int_positive(name: str, value: int | float) -> int:
    """Validate and coerce to positive integer. Raises TypeError or ValueError."""
    v = _validate_int_non_negative(name, value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {v}")
    return v


def _validate_str_non_empty(name: str, value: str) -> str:
    """Validate non-empty string. Raises TypeError or ValueError."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError(f"{name} must be non-empty")
    return s


def _validate_numeric(name: str, value: float) -> float:
    """Validate and coerce to float. Raises TypeError."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    return float(value)


def _validate_optional_positive(name: str, value: float | None) -> float | None:
    """Validate optional positive float. Returns None if value is None."""
    if value is None:
        return None
    return _validate_positive(name, value)


def _validate_extrapolate(
    value: Tuple[bool, float],
    *,
    name: str = "extrapolate",
) -> Tuple[bool, float]:
    """Validate extrapolate tuple (enable: bool, f_trans: float > 0). Returns immutable tuple."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            f"{name} must be (enable: bool, f_trans: float), got {type(value).__name__}"
        )
    enable, f_trans = value
    if not isinstance(enable, bool):
        raise TypeError(f"{name}[0] must be bool, got {type(enable).__name__}")
    f_trans_f = _validate_positive(f"{name}[1] (f_trans)", f_trans)
    return (enable, f_trans_f)
