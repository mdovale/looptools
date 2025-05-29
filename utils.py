import re

def normalize_tf_string(tf_str: str, debug: bool = False) -> str:
    """
    Normalize a symbolic transfer function string for parsing with SymPy.

    This function performs the following transformations:
    1. Protects scientific notation literals (e.g., "1e-05") to prevent them from
       being misinterpreted as symbolic expressions (e.g., "1 * e - 5").
    2. Inserts explicit multiplication operators where needed (e.g., "2z" → "2*z").
    3. Restores scientific notation as decimal floats (e.g., "1e-05" → "0.00001").

    The normalization ensures that the resulting string is safe for parsing via
    `sympy.parse_expr`, especially in control loop transfer function definitions.

    Parameters
    ----------
    tf_str : str
        The raw string representation of the transfer function to be normalized.
    debug : bool, optional
        If True, prints intermediate processing steps for debugging.

    Returns
    -------
    str
        A sanitized, parseable transfer function string with proper syntax.

    Examples
    --------
    >>> normalize_tf_string('1 + 0.01/(1 - z**-1) + 1e-05/(1 - z**-1)**2')
    '1 + 0.01/(1 - z**-1) + 0.0000100000000000/(1 - z**-1)**2'
    """
    sci_pattern = re.compile(r'\b\d+\.?\d*(e[+-]?\d+)\b', re.IGNORECASE)
    protected_literals = {}

    def protect_sci(match):
        raw = tf_str[match.start():match.end()]
        val = format(float(raw), '.16f')  # Convert to base-10 decimal
        key = f"<<__SCI_{len(protected_literals)}__>>"
        protected_literals[key] = val
        if debug:
            print(f"[DEBUG] Protected: {repr(raw)} → {val} as {key}")
        return key

    # Step 1: Protect scientific notation
    tf_protected = sci_pattern.sub(protect_sci, tf_str)
    if debug:
        print("[DEBUG] After protecting scientific notation:", repr(tf_protected))

    # Step 2: Insert multiplication safely, ignoring placeholders
    placeholder_pattern = re.compile(r'<<__SCI_\d+__>>')
    segments = []
    last = 0
    for match in placeholder_pattern.finditer(tf_protected):
        start, end = match.span()
        safe_chunk = tf_protected[last:start]
        safe_chunk = safe_chunk.replace('^', '**')
        safe_chunk = re.sub(r'(?<=\d)(?=[a-df-zA-DF-Z])', '*', safe_chunk)
        safe_chunk = re.sub(r'(?<=[a-zA-Z])(?=\d)', '*', safe_chunk)
        safe_chunk = re.sub(r'(?<=[a-zA-Z])(?=[a-zA-Z])', '*', safe_chunk)
        segments.append(safe_chunk)
        segments.append(tf_protected[start:end])  # placeholder untouched
        last = end
    # Final tail
    tail = tf_protected[last:]
    tail = tail.replace('^', '**')
    tail = re.sub(r'(?<=\d)(?=[a-df-zA-DF-Z])', '*', tail)
    tail = re.sub(r'(?<=[a-zA-Z])(?=\d)', '*', tail)
    tail = re.sub(r'(?<=[a-zA-Z])(?=[a-zA-Z])', '*', tail)
    segments.append(tail)

    tf_final = ''.join(segments)
    if debug:
        print("[DEBUG] After inserting multiplication:", repr(tf_final))

    # Step 3: Restore protected float values
    for key, val in protected_literals.items():
        tf_final = tf_final.replace(key, val)
    if debug:
        print("[DEBUG] After restoring floats:", repr(tf_final))

    return tf_final