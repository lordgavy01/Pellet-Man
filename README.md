# License:
Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and Pieter Abbeel (pabbeel@cs.berkeley.edu).
# Project Details:

This project is one of the Course Project of Course CS-188(Intro to AI) of Berkeley University.

**Note: Run the code on Python 2.6 or Python 2.7 else it won't work**

I have implemented an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. The implementation is in ApproximateQAgent class in qlearningAgents.py, which is a subclass of PacmanQAgent.
Features I have used for training are:
1) whether food will be eaten
2) how far away the next food is
3) whether a ghost collision is imminent
4) whether a ghost is one step away

Pacman can trained with the following command:

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid 

To try on much bigger layouts, Write the following command: (warning: this may take a few minutes to train)

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 

Pacman.py supports various other options like to change layout, number of ghosts , number of episodes.You can see the list of all options and their default values via:

python pacman.py -h

You can train your agent on various other layouts by writing -l [Layout Name], Other Layouts are as follows:
1) capsuleClassic
2) contestClassic
3) mediumClassic
4) mediumGrid
5) minimaxClassic
6) openClassic
7) originalClassic
8) smallClassic
9) smallGrid
10) testClassic
11) trappedClassic
12) trickyClassic

Q-learning agent should win almost every time with these simple features, even with only 50 training games.

Note: If you want to experiment with learning parameters, you can use the option -a, for example -a epsilon=0.1,alpha=0.3,gamma=0.7. These values will then be accessible as self.epsilon, self.gamma and self.alpha inside the agent.

Note: While a total of 2010 games will be played, the first 2000 games will not be displayed because of the option -x 2000, which designates the first 2000 games for training (no output). Thus, you will only see Pacman play the last 10 of these games. The number of training games is also passed to your agent as the option numTraining.

Note: If you want to watch 10 training games to see what's going on, use the command:

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.
