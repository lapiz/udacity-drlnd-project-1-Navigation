from env import *
from dqn_agent import Agent
from itertools import product
from scores import Scores
import sys

def make_param( gamma, hidden ):
    return {
        'buffer_size': int(1e5),    # replay buffer size
        'batch_size': 64,           # minibatch size
        'gamma': gamma,              # discount factor
        'tau': 1e-3,                # for soft update of target parameters
        'LR': 5e-4,                 # learning rate 
        'update_interval': 4,       # how often to update the network
        'hidden_layer': (hidden, hidden)  # hidden layer info
    }

gamma_set = [ 0.95 ]
hidden_set = [128]
#gamma_set = [ 0.99, 0.95, 0.9 ]
#hidden_set = [64, 128, 256]

EXPECT=13

env = Env()

def get_prefix(g, h):
    return F'result/{g}_{h}'


def Train():
    for arg in product(gamma_set, hidden_set):
        params = make_param(*arg)
        agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0, params = params ) 
        prefix = get_prefix(*arg)
        print(params)
        print(prefix)
        env.TrainAgent(agent, EXPECT, prefix, 10000 )
 
#    for arg in test_set:
#        print( F'Test for {arg[0]} - {arg[1]} started.')
#        Run(arg[0], arg[1], 50)

def Test(count):
     for arg in product(gamma_set, hidden_set):
        print( F'Test for {arg[0]} - {arg[1]} started.')
        scores = Scores(EXPECT, 10, False)
        prefix = get_prefix(*arg)
        for i_episode in range(1, count+1):
            params = make_param(*arg)
            agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0, params = params ) 
            agent.Load(F'{prefix}_checkpoint.pth')
            score = env.RunAgent(agent, False)
            scores.AddScore(score)
            sys.stdout.flush()

        scores.FlushLog(F'test_{get_prefix(*arg)}')

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        Train()
    else:
        Test(int(sys.argv[1]))
