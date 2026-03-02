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
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
from __future__ import annotations

import logging
import re
from typing import Match

# Module-level compiled regex patterns (immutable, compiled once)
_SCI_PATTERN = re.compile(r"\b\d+\.?\d*(e[+-]?\d+)\b", re.IGNORECASE)
_PLACEHOLDER_PATTERN = re.compile(r"<<__SCI_\d+__>>")
_MUL_DIGIT_BEFORE_LETTER = re.compile(r"(?<=\d)(?=[a-df-zA-DF-Z])")
_MUL_LETTER_BEFORE_DIGIT = re.compile(r"(?<=[a-zA-Z])(?=\d)")
_MUL_LETTER_BEFORE_LETTER = re.compile(r"(?<=[a-zA-Z])(?=[a-zA-Z])")

_logger = logging.getLogger(__name__)


def _insert_multiplication(chunk: str) -> str:
    """Insert explicit multiplication operators in a chunk of TF string."""
    result = chunk.replace("^", "**")
    result = _MUL_DIGIT_BEFORE_LETTER.sub("*", result)
    result = _MUL_LETTER_BEFORE_DIGIT.sub("*", result)
    result = _MUL_LETTER_BEFORE_LETTER.sub("*", result)
    return result


def normalize_tf_string(tf_str: str, debug: bool = False) -> str:
    """
    Normalize a symbolic transfer function string for parsing with SymPy.

    This function performs the following transformations:
    1. Protects scientific notation literals (e.g., "1e-05") to prevent them
       from being misinterpreted as symbolic expressions (e.g., "1 * e - 5").
    2. Inserts explicit multiplication operators where needed (e.g., "2z" →
       "2*z").
    3. Restores scientific notation as decimal floats (e.g., "1e-05" →
       "0.00001").

    The normalization ensures that the resulting string is safe for parsing via
    `sympy.parse_expr`, especially in control loop transfer function definitions.

    Parameters
    ----------
    tf_str : str
        The raw string representation of the transfer function to be normalized.
    debug : bool, optional
        If True, logs intermediate processing steps at DEBUG level.

    Returns
    -------
    str
        A sanitized, parseable transfer function string with proper syntax.

    Raises
    ------
    TypeError
        If tf_str is not a string.
    ValueError
        If tf_str is empty.

    Examples
    --------
    >>> normalize_tf_string('1 + 0.01/(1 - z**-1) + 1e-05/(1 - z**-1)**2')
    '1 + 0.01/(1 - z**-1) + 0.0000100000000000/(1 - z**-1)**2'
    """
    if not isinstance(tf_str, str):
        raise TypeError(f"tf_str must be str, got {type(tf_str).__name__}")
    if not tf_str:
        raise ValueError("tf_str cannot be empty")
    if not isinstance(debug, bool):
        raise TypeError(f"debug must be bool, got {type(debug).__name__}")

    protected_literals: dict[str, str] = {}

    def protect_sci(match: Match[str]) -> str:
        raw = tf_str[match.start() : match.end()]
        val = format(float(raw), ".16f")
        key = f"<<__SCI_{len(protected_literals)}__>>"
        protected_literals[key] = val
        if debug:
            _logger.debug("Protected: %r → %s as %s", raw, val, key)
        return key

    # Step 1: Protect scientific notation
    tf_protected = _SCI_PATTERN.sub(protect_sci, tf_str)
    if debug:
        _logger.debug("After protecting scientific notation: %r", tf_protected)

    # Step 2: Insert multiplication safely, ignoring placeholders
    segments: list[str] = []
    last = 0
    for match in _PLACEHOLDER_PATTERN.finditer(tf_protected):
        start, end = match.span()
        safe_chunk = _insert_multiplication(tf_protected[last:start])
        segments.append(safe_chunk)
        segments.append(tf_protected[start:end])
        last = end
    segments.append(_insert_multiplication(tf_protected[last:]))

    tf_final = "".join(segments)
    if debug:
        _logger.debug("After inserting multiplication: %r", tf_final)

    # Step 3: Restore protected float values
    for key, val in protected_literals.items():
        tf_final = tf_final.replace(key, val)
    if debug:
        _logger.debug("After restoring floats: %r", tf_final)

    return tf_final
