#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
import copy
from TSPClasses import *
from random import *
import heapq
from SearchNode import *
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	results = {}
	bssf = -1
	solution = None

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
	# TIME COMPLEXITY: Worst case we get a time complexity of O(n!) where we have to check
	# every possible permutation.
	# SPACE COMPLEXITY: We need to store a list of cities of size n for each permutation
	# attempt as well as for the final answer
	# As we go through each iteration, perm gets overwritten so we don't need to worry about
	# it allocating more space per iteration thus we get a O(n). All other variables are constant in
	# regard to n
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		allGreedySolutions = []
		start_time = time.time()
		count = 0
		bssf = None
		bestCost = np.Infinity
		foundTour = False

		while not foundTour and time.time() - start_time < time_allowance:
			for i in range(len(cities)):
				greedyList = []
				restOfCities = []
				start = cities[i]
				greedyList.append(start)

				if i != len(cities) - 1:
					restOfCities = cities[i+1:]
				if i != 0:
					restOfCities.extend(cities[0:i])

				cityToExplore = start

				while len(restOfCities) > 0:
					cheapestCost = np.Infinity
					nextBestCity = None
					for nextCity in restOfCities:
						#find the cheapest city
						if cityToExplore.costTo(nextCity) < cheapestCost:
							cheapestCost = cityToExplore.costTo(nextCity)
							nextBestCity = nextCity

					#If next best city is none, we don't add this to the master list
					if nextBestCity is None:
						break
					cityToExplore = nextBestCity
					greedyList.append(nextBestCity)
					restOfCities.remove(nextBestCity)

				#make sure it's possible to get back to start city
				if cityToExplore.costTo(start) < np.Infinity and len(greedyList) == len(cities):
					allGreedySolutions.append(TSPSolution(greedyList))


			#Now go through all greedy solutions, find best one
			for sol in allGreedySolutions:
				count += 1
				if sol._costOfRoute() < bestCost:
					bssf = sol
					bestCost = sol._costOfRoute()
			foundTour = True

		end_time = time.time()
		results['time'] = end_time - start_time
		results['count'] = count
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		if bssf is None:
			results['cost'] = np.Infinity
			results['soln'] = None
		else:
			results['cost'] = bestCost
			results['soln'] = bssf
		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def initializeMatrix(self, matrix, size, cities):
		# TIME COMPLEXITY: we have a double for loop iterating through from 0 to size of input (n) thus
		# we get O(n^2)
		# SPACE COMPLEXITY: because we are creating a matrix of n columns by n rows we also get a space
		# complexity of O(n^2)
		for i in range(0,size):
			for j in range(0, size):
				if i == j:
					matrix[i][j] = math.inf
				matrix[i][j] = cities[i].costTo(cities[j])

	def rowReduce(self, matrix, ncities):
		# TIME COMPLEXITY: To get the rowMins, columnMins, and apply these to their respective rows and columns each
		# takes O(n^2). (bc we iterate through each column by each row)
		# While we have multiple O(n) operations, this only adds a constant so the rowReduce method takes
		# O(n^2)
		# SPACE COMPLEXITY: We pass in a matrix of space O(n^2) however this was created ahead of time and none of the
		# remaining variables sizes change in respect to n so we get O(1)

		# row reduce
		rowMins = np.amin(matrix, axis=1)
		for i in range(0, ncities):
			if rowMins[i] != math.inf:
				matrix[i:i+1, :] = matrix[i:i+1, :]-rowMins[i]

		# column reduce
		columnMins = np.amin(matrix, axis=0)
		for i in range(0, ncities):
			if columnMins[i] != math.inf:
				matrix[:, i:i+1] = matrix[:, i:i+1]-columnMins[i]

		# get change
		sum = np.sum(columnMins[columnMins != math.inf]) + np.sum(rowMins[rowMins != math.inf])
		return sum

	def calcLowerBound(self, start, end, matrix, ncities):
		# TIME COMPLEXITY: This function calls rowReduce which is a time complexity of O(n^2). We also change a single column
		# and row to infinity where the rows and columns are of length n which gives a time complexity of O(n). Thus our
		# overall time complexity of calcLowerBound is O(n^2)
		# SPACE COMPLEXITY: We pass in a matrix of space O(n^2) however this was created ahead of time and none of the
		# remaining variables sizes change in respect to n so we get O(1)
		if start is None and end is None:  # initial matrix
			lowerBound = self.rowReduce(matrix, ncities)
		else:  # set necessary row, column, vals to inf
			matrixVal = matrix[start._index, end._index]
			matrix[start._index:start._index+1, :] = math.inf
			matrix[:, end._index: end._index+1] = math.inf
			matrix[end._index: end._index+1, start._index:start._index+1] = math.inf
			lowerBound = self.rowReduce(matrix, ncities) + matrixVal

		return lowerBound

	def convertIndicesToRoute(self, indices, cities):
		# TIME COMPLEXITY: This function loops through cities of length n thus giving a time complexity of O(n)
		# SPACE COMPLEXITY: We create a TSPSolution route object which stores a list of cities of size n and
		# thus get a space complexity of O(n)

		# gets route from city indices stored in node
		routes = []
		for i in range(0, len(indices)):
			routes.append(cities[indices[i]])

		route = TSPSolution(routes)
		return route

	def branchAndBound( self, time_allowance=60.0 ):
		# TIME COMPLEXITY: This function calls various methods: initalizeMatrix,
		# defaultRandomTour, and calcLowerBound. All of these are called before we start creating
		# states. As we create
		# states, we call calcLowerBound which gives a time of O(n^2) and worst case, we do this
		# for (n-1)! states. This
		# dominates everything outside the loop so we get a time complexity of O(n^2)((n-1)!)
		# SPACE COMPLEXITY: As we create nodes, we store a matrix of size O(n^2) and worst case
		# we do this for (n-1)! nodes.
		# Worst case we will also be storing (n-1)! nodes in the priority queue.
		# Thus for space complexity we get O(n^2)((n-1)!)

		# initialize variables
		start_time = time.time()
		priorityQueue = []
		heapq.heapify(priorityQueue)
		cities = self._scenario.getCities()
		ncities = len(cities)

		# create matrix
		matrix = np.zeros(shape=(ncities,ncities))
		self.initializeMatrix(matrix, ncities, cities)

		# setup starting Node
		startIndex = randint(0, ncities-1)
		startCity = cities[startIndex]
		citiesContained = [startCity._index]
		startNode = SearchNode(startCity, citiesContained, None, startIndex)
		startNode.lowerBound = self.calcLowerBound(None, None, matrix, ncities)
		startNode.lowerBoundMatrix = matrix
		heapq.heappush(priorityQueue, (startNode.lowerBound - 1, startNode))

		# get bssf
		initialResult = self.defaultRandomTour()
		self.bssf = initialResult['cost']
		# setup variables to report
		maxQueueSize = 1
		statesCreated = 1
		prunedSize = 0
		solutionsFound = 0

		# begin algorithm
		while len(priorityQueue) != 0 and time.time()-start_time < time_allowance:

			node = heapq.heappop(priorityQueue)[1]  # get second element of tuple (node to expand) from top of queue
			if node.lowerBound < self.bssf:  # do we prune?
				for i in range(0, len(cities)):
					# calc lower bound
					cityMatrix = copy.deepcopy(node.lowerBoundMatrix)
					cityLowerBound = self.calcLowerBound(node.city, cities[i], cityMatrix, ncities)
					if cityLowerBound != math.inf:  # if dest is reachable from current city

						# create node
						cityLowerBound = cityLowerBound + node.lowerBound
						citiesIncluded = copy.deepcopy(node.cities)
						citiesIncluded.append(cities[i]._index)
						cityNode = SearchNode(cities[i], citiesIncluded, None, i)
						cityNode.lowerBound = cityLowerBound
						cityNode.lowerBoundMatrix = cityMatrix
						statesCreated += 1

						if len(citiesIncluded) == ncities:  # we have a possible solution
							if cityLowerBound < self.bssf:  # possibly better than current!
								if cities[i].costTo(cities[citiesIncluded[0]]) != np.inf:  # gets back to start?
									solutionsFound += 1
									#print("solution found")
									self.bssf = cityLowerBound
									self.solution = cityNode
								else:
									prunedSize += 1
						elif cityLowerBound < self.bssf:  # we need to put the node in the queue
							heuristic = cityNode.lowerBound - (len(cityNode.cities)) ** 4
							heapq.heappush(priorityQueue, (heuristic, cityNode))
							if len(priorityQueue) > maxQueueSize:  # updating max queue size
								maxQueueSize = len(priorityQueue)
						else:
							prunedSize += 1
			else:
				prunedSize += 1

		end_time = time.time()
		route = None
		if self.solution is not None:
			route = self.convertIndicesToRoute(self.solution.cities, cities)
			self.results['cost'] = route.cost
			self.results['soln'] = route
		else:
			self.results['cost'] = initialResult['cost']
			self.results['soln'] = initialResult['soln']

		self.results['time'] = end_time - start_time
		self.results['count'] = solutionsFound
		self.results['max'] = maxQueueSize
		self.results['total'] = statesCreated
		self.results['pruned'] = prunedSize

		print(self.results['time'])
		return self.results




	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		start_time = time.time()
		cities = self._scenario.getCities()
		ncities = len(cities)
		matrix = np.zeros(shape=(ncities,ncities))
		#matrix = [[0, 1, 15, 6],[2, 0, 7, 3], [9, 6, 0, 12], [10, 4, 8, 0]]
		self.initializeMatrix(matrix, ncities, cities)
		heldKarp = HeldKarpSolver(matrix, ncities)
		cost, route = heldKarp.solve(start_time, time_allowance)
		end_time = time.time()
		if cost is not None and route is not None:
			self.results['cost'] = cost
			self.results['soln'] = self.convertIndicesToRoute(route[:-1], cities)
		else:
			self.results['cost'] = math.inf
			self.results['soln'] = None
		self.results['time'] = end_time - start_time
		self.results['count'] = -1
		return self.results
