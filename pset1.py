# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 1: Python Coding Questions - Fall 2022
Due September 27, 2022 at 11:59pm
"""

### Package Imports ###
import heapq
import abc
from typing import List, Optional, Tuple
### Package Imports ###


#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.6 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset1.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


class Stack:
    """A container with a last-in-first-out (LIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Push 'item' onto the stack"""
        self.list.append(item)

    def pop(self):
        """Pop the most recently pushed item from the stack"""
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the stack is empty"""
        return len(self.list) == 0

class Queue:
    """A container with a first-in-first-out (FIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Enqueue the 'item' into the queue"""
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the queue is empty"""
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class SearchProblem(abc.ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abc.abstractmethod
    def getStartState(self) -> "State":
        """
        Returns the start state for the search problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def isGoalState(self, state: "State") -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getSuccessors(self, state: "State") -> List[Tuple["State", str, int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getCostOfActions(self, actions) -> int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raise NotImplementedError


ACTION_LIST = ["UP", "DOWN", "LEFT", "RIGHT"]

class State():
    def __init__(self, pos: Tuple[int, int], cost: int, targets: List[Tuple[int, int]], actions : List[str]):
        self.pos = pos
        self.posy = pos[0]
        self.posx = pos[1]
        self.cost = cost
        self.targets = targets
        self.actions = actions
    
    def isGoal(self):
        return not(self.targets)

class GridworldSearchProblem(SearchProblem):
    """
    Fill in these methods to define the grid world search as a search problem.
    Actions are of type `str`. Feel free to use any data type/structure to define your states though.
    In the type hints, we use "State" to denote a data structure that keeps track of the state, and you can use
    any implementation of a "State" you want.
    """
    def __init__(self, file_name):
        """Read the text file and initialize all necessary variables for the search problem"""
        "*** YOUR CODE HERE ***"
        file = open(file_name)
        data = [i.split() for i in file.readlines()]
        self.height = int(data[0][0])
        self.width = int(data[0][1])
        self.map = []
        self.toVisit = []
        for i in range(1, 1 + self.height):
            row = []
            for j in range(0, self.width):
                pos = int(data[i][j])
                row.append(int(data[i][j]))
                if pos == 1:
                    self.toVisit.append((i - 1, j))
            self.map.append(row)
        self.start = State((int(data[-1][1]), int(data[-1][0])), 0, self.toVisit, [])
        

    def getStartState(self) -> State:
        "*** YOUR CODE HERE ***"
        return self.start

    def isGoalState(self, state: State) -> bool:
        "*** YOUR CODE HERE ***"
        return state.isGoal()

    def getSuccessors(self, state: State) -> List[Tuple[State, str, int]]:
        # This return a list of succesors, each of which is a tuple of a state, the action taken to get there, and the cost?
        "*** YOUR CODE HERE ***"
        successors = []
        # Check if valid to move in each direction if so add successor to array
        # This is a list of tuples, where each tuple has a state, the move to get to that state,
        # And the total cost of getting to that state from the start state
        if (state.posy + 1 < self.height and self.map[state.posy + 1][state.posx] != -1):
            succesorPos = (state.posy + 1, state.posx)
            successors.append((State(succesorPos,
                                     state.cost + 1, 
                                     list(filter(lambda x: x != (succesorPos), state.targets)),
                                     state.actions + [ACTION_LIST[1]]), 
                               ACTION_LIST[1],
                               state.cost + 1))
                              
        if (state.posx + 1 < self.width and self.map[state.posy][state.posx + 1] != -1):
            succesorPos = (state.posy, state.posx + 1)
            successors.append((State(succesorPos,
                                     state.cost + 1,
                                     list(filter(lambda x: x != (succesorPos), state.targets)),
                                     state.actions + [ACTION_LIST[3]]),
                               ACTION_LIST[3],
                               state.cost + 1))

        if (state.posy - 1 >= 0 and self.map[state.posy - 1][state.posx] != -1):
            succesorPos = (state.posy - 1, state.posx)
            successors.append((State(succesorPos,
                                     state.cost + 1,
                                     list(filter(lambda x: x != (succesorPos), state.targets)),
                                     state.actions + [ACTION_LIST[0]]),
                               ACTION_LIST[0],
                               state.cost + 1))

        if (state.posx - 1 >= 0 and self.map[state.posy][state.posx - 1] != -1):
            succesorPos = (state.posy, state.posx - 1)
            successors.append((State(succesorPos,
                                     state.cost + 1,
                                     list(filter(lambda x: x != (succesorPos), state.targets)),
                                     state.actions + [ACTION_LIST[2]]),
                               ACTION_LIST[2],
                               state.cost + 1))
        return successors

    def getCostOfActions(self, actions: List[str]) -> int:
        "*** YOUR CODE HERE ***"
        return len(actions)


def depthFirstSearch(problem: SearchProblem) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState().pos)
    visited = set()
    stack = Stack()
    stack.push(problem.getStartState())
    while(not(stack.isEmpty())):
        state = stack.pop()
        # print(state.pos)
        # print(state)
        # What if we have visited before but it's taken us longer to get there?
        # This is also not tracking actions correctly, if we have been there and return we should remove that 
        if (state.pos, tuple(state.targets)) in visited:
            continue
        visited.add((state.pos, tuple(state.targets)))
        if state.isGoal():
            return state.actions
        for suc in problem.getSuccessors(state):
            stack.push(suc[0])
    return None


def breadthFirstSearch(problem: SearchProblem) -> List[str]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    raise NotImplementedError


def nullHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def simpleHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    This heuristic returns the number of residences that you have not yet visited.
    """
    raise NotImplementedError


def customHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    Create your own heurstic. The heuristic should
        (1) reduce the number of states that we need to search (tested by the autograder by counting the number of
            calls to GridworldSearchProblem.getSuccessors)
        (2) be admissible and consistent
    """
    raise NotImplementedError


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[str]:
    """Search the node that has the lowest combined cost and heuristic first.
    This function takes in an arbitrary heuristic (which itself is a function) as an input."""
    "*** YOUR CODE HERE ***"
    raise NotImplementedError


if __name__ == "__main__":
    ### Sample Test Cases ###
    # Run the following statements below to test the running of your program
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case1.txt")
    print(depthFirstSearch(gridworld_search_problem))
    exit()       
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case1.txt") # Test Case 1
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case2.txt") # Test Case 2
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case3.txt") # Test Case 3
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))
