import platform

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from unityagents import UnityEnvironment
from scores import Scores

class Env:
    def __init__(self):
        fn = 'Banana.app'
        if platform.system() == 'Linux':
            fn = 'Banana_Linux_NoVis/Banana.x86_64'
        self.env = UnityEnvironment(file_name=fn)
        self.brain_name = self.env.brain_names[0]
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        brain = self.env.brains[self.brain_name]
        state = env_info.vector_observations[0]

        self.action_size = brain.vector_action_space_size
        self.state_size = len(state)

        print('Number of agents:', len(env_info.agents))
        print('Number of actions:', self.action_size)
        
        print('States look like:', state)
        print('States have length:', self.state_size)

    def RunRandomly(self, train_mode = True ):
        return self._run(train_mode, lambda s: np.random.randint(self.action_size), lambda s,a,r,ns,d : None)

    def RunAgent(self, agent, train_mode = True ):
        return self._run(train_mode, lambda s: agent.act(s, 0.), lambda s,a,r,ns,d: None)

    def TrainAgent(self, agent, expect, prefix, episodes=1000, window_size=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = Scores(expect, window_size)
        success = False
        eps = eps_start
        for i_episode in range(1, episodes+1):
            score = self._run(True, lambda s: agent.act(s, eps), lambda s,a,r,ns,d : agent.step(s,a,r,ns,d) )
            eps = max(eps_end, eps_decay*eps)
            if scores.AddScore(score) == True:
                agent.Save(F'{prefix}_checkpoint.pth')
                success = True
                break

        scores.FlushLog(prefix)
        return success

    def _run(self, train_mode, get_action, update_agent):
        score = 0
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        done = False
        while not done:
            action = get_action(state)
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            update_agent(state, action, reward, next_state, done)
            score += reward
            state = next_state
        return score
        
