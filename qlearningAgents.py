
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
	self.Qvalue=util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Qvalue[(state,action)]
  

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        act=self.getLegalActions(state)
	if act:
	 maxi=-9999999999
	 for i in self.getLegalActions(state):
		k=self.getQValue(state,i)
		if k >= maxi:
    		  maxi=k
	 return maxi
	return 0.0	

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        act=self.getLegalActions(state)
	if act:
	  maxi=-99999999
	  bestact=None
          for i in self.getLegalActions(state):
	        k=self.getQValue(state,i)
		if k >= maxi:
    		  maxi=k
		  bestact=i
	  return bestact
	return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        action = None
	if legalActions==None:
	 return None
	if util.flipCoin(self.epsilon) == True:
         act=self.getLegalActions(state)
	# opti=self.getPolicy(state)
	 action=random.choice(act)
	# while action == opti:
	#	action=random.choice(act)
	 return action
        else:
	 return self.getPolicy(state)	
        
	return action
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
       	if self.getLegalActions(nextState):
	 maxi=-999999999	
	 for i in self.getLegalActions(nextState):
	   	if self.getQValue(nextState,i) >= maxi:
		  maxi=self.getQValue(nextState,i)
	 rew=reward+self.discount*maxi
	else:
	 rew=reward
	self.Qvalue[(state,action)]=self.getQValue(state,action)+self.alpha*(rew- self.getQValue(state,action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
	"""
        ans=0
	features= self.featExtractor.getFeatures(state,action)
	for j in features.keys():
		ans += self.weights[j]*features[j]
	
	return ans 
  

    def update(self, state, action, nextState, reward):
	features= self.featExtractor.getFeatures(state,action)
	bestq=-9999999999
        for act in self.getLegalActions(nextState):
            if self.getQValue(nextState,act) > bestq:
                bestq = self.getQValue(nextState,act)
        if bestq == -9999999999:
            bestq = 0
        res = (reward + (self.discount * bestq)) - self.getQValue(state,action)
      	res*=self.alpha
        self.Qvalue[(state,action)] += res 
        for feature in features.keys():
            self.weights[feature] += res* features[feature]
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
