# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            V_new = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    V_values = []
                    for action in self.mdp.getPossibleActions(state):
                        V_i = self.computeQValueFromValues(state, action)
                        V_values.append(V_i)
                    V_new[state] = max(V_values)
                else:
                    V_new[state] = 0
            self.values = V_new

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_value = 0
        for nextStates in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState, probability = nextStates
            reward = self.mdp.getReward(state, action, nextState)
            Q_value += probability * (reward + (self.discount * self.values[nextState]))
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        Q_value = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            q_values = self.computeQValueFromValues(state, action)
            if q_values > Q_value:
                Q_value = q_values
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            #pdb.set_trace()
            V_new = util.Counter()
            states = self.mdp.getStates()
            numberOfStates = len(self.mdp.getStates())
            if i == 0 or l == numberOfStates:
                l = 0
            if not self.mdp.isTerminal(states[l]):
                V_values = []
                for action in self.mdp.getPossibleActions(states[l]):
                    V_i = self.computeQValueFromValues(states[l], action)
                    V_values.append(V_i)
                V_new[states[l]] = max(V_values)
                self.values[states[l]] = V_new[states[l]]
            l += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}

        def maximumQValue(state):
            Q_value = float("-inf")
            for action in self.mdp.getPossibleActions(state):
                q_values = self.computeQValueFromValues(state, action)
                if q_values > Q_value:
                    Q_value = q_values
            return Q_value

        for state in states:
            predecessors[state] = set()
            if not self.mdp.isTerminal(state):
                for state1 in states:
                    for action in self.mdp.getPossibleActions(state1):
                         for nextStates in self.mdp.getTransitionStatesAndProbs(state1, action):
                              if nextStates[0] == state and nextStates[1] > 0:
                                  predecessors[state].add(state1)

        queue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                difference = abs(self.values[state] - maximumQValue(state))
                queue.push(state, -difference)

        for i in range(0, self.iterations):
            if queue.isEmpty():
                return
            priorityState = queue.pop()
            if not self.mdp.isTerminal(priorityState):
                self.values[priorityState] = maximumQValue(priorityState)
            for predecessor in predecessors[priorityState]:
                diff = abs(self.values[predecessor] - maximumQValue(predecessor))
                if diff > self.theta:
                    queue.update(predecessor, -diff)