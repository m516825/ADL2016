"""
### NOTICE ###

You need to upload this file.
You can add any function you want in this file.

"""
import importlib
import argparse
import sys
import numpy as np
import random
sys.path.append('./dqn-atari')
import dqn
from state import State

class Argument(object):
    def __init__(self):
        self.train_epoch_steps = 250000
        self.eval_epoch_steps = 125000
        self.replay_capacity = 1000000
        self.normalize_weights = True
        self.screen_capture_freq = 250
        self.save_model_freq = 10000
        self.observation_steps = 50000
        self.learning_rate = 0.00025
        self.target_model_update_freq = 10000
        self.model = './best_model.ckpt'
class Agent(object):
    def __init__(self, sess, min_action_set):
        self.sess = sess
        self.min_action_set = min_action_set
        self.build_dqn()
        self.state = State()
        self.epsilon = .05

    def build_dqn(self):
        """
        # TODO
            You need to build your DQN here.
            And load the pre-trained model named as './best_model.ckpt'.
            For example, 
                saver.restore(self.sess, './best_model.ckpt')
        """
        args = Argument()
        self.dqn = dqn.DeepQNetwork(len(self.min_action_set), './', args)

    def getSetting(self):
        """
        # TODO
            You can only modify these three parameters.
            Adding any other parameters are not allowed.
            1. action_repeat: number of time for repeating the same action 
            2. screen_type: return 0 for RGB; return 1 for GrayScale
        """
        action_repeat = 4
        screen_type = 0
        return action_repeat, screen_type

    def play(self, screen):
        """
        # TODO
            The "action" is your DQN argmax ouput.
            The "min_action_set" is used to transform DQN argmax ouput into real action number.
            For example,
                 DQN output = [0.1, 0.2, 0.1, 0.6]
                 argmax = 3
                 min_action_set = [0, 1, 3, 4]
                 real action number = 4
        """
        action = 0 # you can remove this line

        if self.state is None or random.random() > (1 - self.epsilon):
            action = random.randrange(len(self.min_action_set))
        else:
            self.state = self.state.stateByAddingScreen(screen, 0)
            screens = np.reshape(self.state.getScreens(), (1, 84, 84, 4))
            action = self.dqn.inference(screens)

        return self.min_action_set[action]
