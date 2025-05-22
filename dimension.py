import copy

class Dimension:
	def __init__(self, numes=[], denos=[], dimensionless=False):
		""" class for a dimension
		Args:
			numes: numerators
			denos: denominators
		"""

		if dimensionless:
			self.numes = []
			self.denos = []
		else:
			self.numes = numes
			self.denos = denos
		self.reduction(self.numes, self.denos)

	def __mul__(self, other):
		""" define * operator
		Args:
			numes: numerators
			denos: denominators
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
		"""　dimension reduction
		Args:
			numes: numerators
			denos: denominators
		"""

		commons = list(set(numes) & set(denos))
		for common in commons:
			n_common_nummes = numes.count(common)
			n_common_denos = denos.count(common)
			n_common = min(n_common_nummes, n_common_denos)
			for i in range(n_common):
				numes.remove(common)
				denos.remove(common)

	def unit_string(self):
		"""　generate unit string
		Args:
		"""

		# : numerator
		if not self.numes:
			numerator = ""
		else:
			numerator = self.numes[0]
			for i in range(1,len(self.numes)):
				numerator += "*" + self.numes[i]
		# : denominator
		if not self.denos:
			denominator = ""
		else:
			denominator = self.denos[0]
			for i in range(1,len(self.denos)):
				denominator += "*" + self.denos[i]

		return numerator+"/"+denominator

	def show(self):
		"""　disply an unit
		Args:
		"""

		unit = self.unit_string()
		# : print
		print("["+unit+"]")
		#print(r"$\frac{}{}$".format(numerator,denominator))
		