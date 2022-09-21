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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        stayawayconstant = 3
        getfoodconstant = 5
        deductions = 0
        bonus = 0
        
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                deductions -= (stayawayconstant/max(manhattanDistance(ghost.getPosition(), newPos), 0.1))**2
            else:
                bonus += (stayawayconstant/max(manhattanDistance(ghost.getPosition(), newPos), 0.1))**2
        
        foodList = newFood.asList()
        if len(foodList) == 0:
            return 10000
            
        closestFoodDistance = manhattanDistance(foodList[0], newPos)
        for currFood in foodList:
            currDistance = manhattanDistance(currFood, newPos)
            if(currDistance < closestFoodDistance):
                closestFoodDistance = currDistance
        bonus += getfoodconstant/max(closestFoodDistance, 0.1)
        
        return successorGameState.getScore() + deductions + bonus

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        
        def minimax(state, depth, agent):
            if agent >= state.getNumAgents():
                agent = 0
                
            bestMove = None
            if agent == 0:
                if depth <= 0 or state.isLose() or state.isWin():
                    return self.evaluationFunction(state), bestMove
                bestSoFar = float('-inf')
                for action in state.getLegalActions(agent):
                    nextState = state.generateSuccessor(agent, action)
                    response = minimax(nextState, depth - 1, agent + 1)[0]
                    if response > bestSoFar:
                        bestSoFar, bestMove = response, action
            else:
                if depth < 0 or state.isLose() or state.isWin():
                    return self.evaluationFunction(state), bestMove
                bestSoFar = float('inf')
                for action in state.getLegalActions(agent):
                    nextState = state.generateSuccessor(agent, action)
                    response = minimax(nextState, depth, agent + 1)[0]
                    if response < bestSoFar:
                        bestSoFar, bestMove = response, action
            return bestSoFar, bestMove
            
        return minimax(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agent, alpha, beta):
            if agent >= state.getNumAgents():
                agent = 0
                
            bestMove = None
            if agent == 0:
                if depth <= 0 or state.isLose() or state.isWin():
                    return self.evaluationFunction(state), bestMove
                bestSoFar = float('-inf')
                for action in state.getLegalActions(agent):
                    nextState = state.generateSuccessor(agent, action)
                    response = minimax(nextState, depth - 1, agent + 1, alpha, beta)[0]
                    if response > bestSoFar:
                        bestSoFar, bestMove = response, action
                        alpha = max(alpha, bestSoFar)
                        if alpha > beta:
                            return bestSoFar, bestMove
            else:
                if depth < 0 or state.isLose() or state.isWin():
                    return self.evaluationFunction(state), bestMove
                bestSoFar = float('inf')
                for action in state.getLegalActions(agent):
                    nextState = state.generateSuccessor(agent, action)
                    response = minimax(nextState, depth, agent + 1, alpha, beta)[0]
                    if response < bestSoFar:
                        bestSoFar, bestMove = response, action
                        beta = min(beta, bestSoFar)
                        if alpha > beta:
                            return bestSoFar, bestMove
            return bestSoFar, bestMove
            
        return minimax(gameState, self.depth, 0, float('-inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectHelper(0, gameState, 0)[1]

    def expectHelper(self, currDepth, gameState, agent):
        if agent >= gameState.getNumAgents():
            agent = 0

        if agent == 0:
            if currDepth >= self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), None

            return self.maxAgentDecision(currDepth, gameState, agent)
        else:
            if currDepth > self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), None

            return self.minAgentDecision(currDepth, gameState, agent)

    def maxAgentDecision(self, currDepth, gameState, agent):
        currScore = 0
        actionToReturn = 0
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            scoreReturned = self.expectHelper(currDepth + 1, gameState.generateSuccessor(agent, action), agent + 1)[0]
            if scoreReturned >= currScore:
                actionToReturn = action
                currScore = scoreReturned
        return currScore, actionToReturn

    def minAgentDecision(self, currDepth, gameState, agent):
        legalActions = gameState.getLegalActions(agent)
        average = 0
        for action in legalActions:
            nextState = gameState.generateSuccessor(agent, action)
            average += self.expectHelper(currDepth, nextState, agent+1)[0] / len(legalActions)

        randomMove = random.randint(0, len(legalActions) - 1)
        return average, legalActions[randomMove]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    deductions = 0
    bonus = 0

    foodList = newFood.asList()
    if len(foodList) == 0:
        return 10000
    currentGameState.getGhostStates()
    for ghost in newGhostStates:
        distanceToGhost = max((manhattanDistance(ghost.getPosition(), newPos)), 0.1)
        if ghost.scaredTimer == 0:
            if distanceToGhost < 2:
                deductions -= (1 / max(distanceToGhost, 0.01))
        else:
            bonus += (1/max((manhattanDistance(ghost.getPosition(), newPos)), 0.001)) ** 5

    closest = 100000
    for food in foodList:
        if manhattanDistance(food, newPos) < closest:
            closest = manhattanDistance(food, newPos)

    bonus += (1/(max(closest, 0.01) + len(foodList))) ** 3
    if distanceToGhost < 1:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(newGhostStates[0].scaredTimer)
    #print("Pacman position: ", newPos, " Ghost position: ", newGhostStates[0].getPosition(), " Distance to ghost: ",
          #distanceToGhost, " Closest to food: ", closest, " Bonus: ", bonus,
          #" Deductions: ", deductions, " Value: ", bonus + deductions)
    return bonus + deductions


# Abbreviation
better = betterEvaluationFunction
