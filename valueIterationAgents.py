

import mdp, util

from learningAgents import ValueEstimationAgent

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
        # Write value iteration code here
        V=util.Counter()
        for j in range(iterations):
           lastv=V.copy()
           for posi in mdp.getStates():
	       if mdp.isTerminal(posi) == False:
                 lin=-9999999999
                 for act in mdp.getPossibleActions(posi):
		    temprew=0
                    for losi in mdp.getTransitionStatesAndProbs(posi,act):
                        temprew=temprew+losi[1]*(mdp.getReward(posi,act,losi[0])+discount*lastv[losi[0]])
                    if temprew > lin:
                        lin=temprew                      
                 V[posi]=lin
	       else:
		    for act in mdp.getPossibleActions(posi):
		     temprew=0 
                     for losi in mdp.getTransitionStatesAndProbs(posi,act):
                        temprew=temprew+losi[1]*(mdp.getReward(posi,act,losi[0])+discount*lastv[losi[0]])
	             V[posi]=temprew
	self.values=V
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
	ans=0
	for j in self.mdp.getTransitionStatesAndProbs(state,action):
            ans= ans+j[1]*(self.mdp.getReward(state,action,j[0])+self.discount*self.values[j[0]])
        return ans
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
	rew=-99999999999
        if self.mdp.isTerminal(state)== True:
            return None
        maxq=-99999999999
	for i in self.mdp.getPossibleActions(state):
	    if self.getQValue(state,i)>maxq:
               maxq=self.getQValue(state,i)
               maxact=i
        return maxact
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
