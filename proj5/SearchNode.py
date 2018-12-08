import math

class SearchNode:

	city = None
	cities = []
	unvisited = []
	lowerBound = math.inf
	index = -1
	lowerBoundMatrix = None

	def __init__(self, city, cities, unvisited, index):
		self.cities = cities
		self.city = city
		self.unvisited = unvisited
		self.index = index

	def __lt__(self, other):
		return self.lowerBound-(len(self.cities))**4  < other.lowerBound-(len(other.cities))**4