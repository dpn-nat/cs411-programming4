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

        score = successorGameState.getScore() # i started with the base game score of the successor state 
        foodList = newFood.asList() # taking the food grid and turned it into a list of positions 

        if len(foodList) > 0: # if any food remanins on the board 
            distances = [manhattanDistance(newPos, food) for food in foodList] # compute the manhattan distance from the pacman to each food pellet 
            minFoodDist = min(distances) # find the distance to the closest food on the board 
            score += 1.0 / (minFoodDist + 1) # the reward will go up as pacman reach or goes toward the food 

        for i, ghost in enumerate(newGhostStates): # looping through ghost to evalute its effect
            ghostPos = ghost.getPosition() # getting position of ghost by using getposition 
            dist = manhattanDistance(newPos, ghostPos) # using manhattandistance i computed the distance from pacman to ghost 

            if newScaredTimes[i] > 0: #  if ghost can be eaten
                score += 2.0 / (dist + 1) # then move toward the ghost and increase the score so pacman knows that he should that more 
            else:
                if dist < 2: # if the ghost is close and you can't eat it then 
                    score -= 100  # ran away so score is penalized to move toward the ghost 
                else:
                    score -= 1.0 / (dist + 1) # but if is in near distance or avoid moving toward the ghost then lower penalized

        return score

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
        totalAgents = gameState.getNumAgents() # this gets total number of agents in game board

        def minimax(state, agentIndex, depth):  # using recursive 
            if depth == self.depth or state.isWin() or state.isLose(): # if max depth has been reach or terminal state then 
                return self.evaluationFunction(state) # return evaluaiton score
            
            if agentIndex == 0: # when pacman is in control 
                actions = state.getLegalActions(agentIndex) # get all actions that pacman can make 
                value = minimax(state.generateSuccessor(agentIndex, actions[0]), 1, depth) # I use the first result of the possible move as begining movve 

                for action in actions: # other remaning actions 
                    success = state.generateSuccessor(agentIndex, action) 
                    value = max(value, minimax(success, 1, depth)) # taking the max value 

                return value # return the best value for pacman 

            else: # now for the ghost I revversed it 
                actions = state.getLegalActions(agentIndex) # once again all legal action that we take 

            nextAgent = agentIndex + 1 # move to next agent 
            nextDepth = depth # but keep the same depth 

            if nextAgent == totalAgents: # if this agent was last one  go back to pacman 
                nextAgent = 0 
                nextDepth += 1 # and increase depth 

            oneAction = actions[0] # once again get the first other remaning actions 
            oneSucc = state.generateSuccessor(agentIndex, oneAction) # generate the game state after getting the first action 
            value = minimax(oneSucc, nextAgent, nextDepth) # and evaluate the first succesor using minimax function 

            for action in actions[1:]: # once again go through all remaning action 
                successor = state.generateSuccessor(agentIndex, action) # generating the successor state 
                value = min(value, minimax(successor, nextAgent, nextDepth)) # compute the min value for this successor 

            return value # return the minimum value found 

        actions = gameState.getLegalActions(0) # getting all legal moves for pacman 

        oneAction = actions[0] # take the first action as begining move 
        oneSucc = gameState.generateSuccessor(0, oneAction) # generate the next game 
        bestScore = minimax(oneSucc, 1, 0) # using minimax calculate the bestscore 
        bestAction = oneAction # so first action replaced with bestaction 

        for action in actions[1:]: # once again go through remaning action 
            successor = gameState.generateSuccessor(0, action) # generate the successor state 
            score = minimax(successor, 1, 0) # using minimax compute the score for this action 

            if score > bestScore: # if this action is better than then current best 
                bestScore = score # update the best score
                bestAction = action  # update the best action 

        return bestAction # return the best action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #  this is for the total number of agents (pacman + ghosts)
        numAgents = gameState.getNumAgents()

        # this is the recursive helper function
        def helper(stat, agenInd, dept, alph, bet):
            # the base case is to stop if we hit depth limit or win/lose state
            if dept == self.depth or stat.isWin() or stat.isLose():
                return self.evaluationFunction(stat)

            # this is to get all the possible actions for the agent
            acts = stat.getLegalActions(agenInd)

            if agenInd == 0:
                # this is to start with a very low value
                valu = -1000000000

                
                for act in acts:    
                    # this is to generate the next state
                    successor = stat.generateSuccessor(agenInd, act)
                    # this is to call the helper for the next agent (ghotst)
                    valu = max(valu, helper(successor, 1, dept, alph, bet))

                    if valu > bet:
                        return valu
                    
                    # this updates alpha
                    alph = max(alph, valu)

                return valu

            else:
                # this starts with a very high value
                valu = 1000000000

                # this moves to the next agent
                nexAget = agenInd + 1
                nexDept = dept

                # if it is the last ghost then it would go back to pacman and increase the depth
                if nexAget == numAgents:
                    nexAget = 0
                    nexDept += 1

                for act in acts:
                    # this generates the next state
                    successor = stat.generateSuccessor(agenInd, act)
                    # this is the recursive call
                    valu = min(valu, helper(successor, nexAget, nexDept, alph, bet))

                    # if the value is already worse than alpha, then it would keep exploring
                    if valu < alph:
                        return valu

                    # this updates the beta
                    bet = min(bet, valu)

                return valu

        alph = -1000000000
        bet = 1000000000
        # thsi tracks the best score and the best action
        besScor = -1000000000
        besAction = None
        
        # this is to loop through all the pacman actions
        for act in gameState.getLegalActions(0):
            # this generates the next state
            successor = gameState.generateSuccessor(0, act)
            scor = helper(successor, 1, 0, alph, bet)

            # updates the best action if a better score is found
            if scor > besScor:
                besScor = scor
                besAction = act

            # this updates the alpha at the root
            alph = max(alph, besScor)

        # this returns the best move
        return besAction

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
