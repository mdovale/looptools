import copy

class Dimension:
    """
    Represents a symbolic physical dimension using numerator and denominator units.

    This class allows basic algebraic manipulation of unit-like structures,
    such as multiplying and reducing dimensional expressions symbolically.
    It is not tied to a specific unit system, but works with symbolic strings.
    """

    def __init__(self, numes=None, denos=None, dimensionless=False):
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
            self.numes = []
            self.denos = []
        else:
            self.numes = numes
            self.denos = denos

        self.reduction(self.numes, self.denos)

    def __mul__(self, other):
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
        self_numes = copy.copy(self.numes)
        self_denos = copy.copy(self.denos)
        other_numes = copy.copy(other.numes)
        other_denos = copy.copy(other.denos)

        self.reduction(self_numes, other_denos)
        self.reduction(self_denos, other_numes)

        self_numes.extend(other_numes)
        self_denos.extend(other_denos)

        return Dimension(self_numes, self_denos)

    def reduction(self, numes, denos):
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

    def unit_string(self):
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

    def show(self):
        """
        Print the dimension as a symbolic unit string.

        Examples
        --------
        >>> d = Dimension(['m'], ['s'])
        >>> d.show()
        [m/s]
        """
        print(f"[{self.unit_string()}]")