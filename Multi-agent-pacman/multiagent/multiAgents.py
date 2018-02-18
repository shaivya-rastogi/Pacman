# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import pdb

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentFoodAsList = (currentGameState.getFood()).asList()
        foodDistance = 0
        ghostDistance = 0
        heuristic = 0

        for currentFood in currentFoodAsList:
            foodDistance = (manhattanDistance(newPos, currentFood))
            if foodDistance == 0:
                heuristic += 100
            else:
                heuristic += 10/foodDistance

        ghostPosition = [ghostState.getPosition() for ghostState in newGhostStates]
        for ghostPos in ghostPosition:
            ghostDistance = (manhattanDistance(newPos, ghostPos))
            if ghostDistance <= 1:
                if ghostState.scaredTimer == 0:
                    heuristic -= 200
                else:
                    heuristic += 100
            else:
                heuristic += 2

        return heuristic

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.MinimaxSearch(gameState, 1, 0)

    def MinimaxSearch(self, gameState, currDepth, agentIndex):
        if currDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        validActions = list()
        moves = list()
        winnerIndex = list()
        for action in gameState.getLegalActions(agentIndex):
            if action != 'Stop' or action != 'STOP':
                validActions.append(action)

        newIndex = agentIndex + 1
        newDepth = currDepth
        if newIndex >= gameState.getNumAgents():
            newIndex = 0
            newDepth = newDepth + 1
        for action in validActions:
            val = self.MinimaxSearch(gameState.generateSuccessor(agentIndex, action), newDepth, newIndex)
            moves.append(val)

        if agentIndex == 0 and currDepth == 1:
            winningMove = max(moves)
            for idx in range(len(moves)):
                if moves[idx] == winningMove:
                    winnerIndex = idx
            return validActions[winnerIndex]

        if agentIndex == 0:
            return max(moves)
        else:
            return min(moves)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        result = float("-inf")
        for action in gameState.getLegalActions(0):
            #nextState = gameState.generateSuccessor(0, action)
            result1 = result
            result = max(result, self.minimumValue(gameState.generateSuccessor(0, action), 1,
                                                 1, alpha, beta))
            if result > result1:
                optimalAction = action
            if result >= beta:
                return optimalAction
            alpha = max(alpha, result)
        return optimalAction

    def minimumValue(self, gameState, agentIndex, currDepth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or currDepth > self.depth:
            return self.evaluationFunction(gameState)
        v = float("inf")
        validActions = gameState.getLegalActions(agentIndex)

        if agentIndex == gameState.getNumAgents() - 1:
            for action in validActions:
                v = min(v, self.maximumValue(gameState.generateSuccessor(agentIndex, action),
                                             0, currDepth + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
        else:
            for action in validActions:
                v = min(v, self.minimumValue(gameState.generateSuccessor(agentIndex, action),
                                             agentIndex + 1, currDepth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
        return v

    def maximumValue(self, gameState, agentIndex, currDepth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or currDepth > self.depth:
            return self.evaluationFunction(gameState)

        validActions = gameState.getLegalActions(0)
        v = float("-inf")
        for action in validActions:
            v = max(v, self.minimumValue(gameState.generateSuccessor(0, action),
                                         1, currDepth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectiMax(gameState, 1, 0)

    def ExpectiMax(self, gameState, currDepth, agentIndex):
        if currDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        validActions = list()
        moves = list()
        for action in gameState.getLegalActions(agentIndex):
            if action != 'Stop' or action != 'STOP':
                validActions.append(action)

        newIndex = agentIndex + 1
        newDepth = currDepth
        if newIndex >= gameState.getNumAgents():
            newIndex = 0
            newDepth = newDepth + 1
        for action in validActions:
            val = self.ExpectiMax(gameState.generateSuccessor(agentIndex, action), newDepth, newIndex)
            moves.append(val)

        if agentIndex == 0 and currDepth == 1:
            winningMove = max(moves)
            for idx in range(len(moves)):
                if moves[idx] == winningMove:
                    winnerIndex = idx
            return validActions[winnerIndex]

        if agentIndex == 0:
            return max(moves)
        else:
            avgMove = sum(moves) / len(moves)
            return avgMove

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    ghostScore = 0
    for ghost in currGhostStates:
        distance = manhattanDistance(currPos, currGhostStates[0].getPosition())
        if distance > 0:
            ghostScore += 10 / distance
    score += ghostScore
    return score


# Abbreviation
better = betterEvaluationFunction

