import copy
import functools

import gymnasium.spaces as spaces
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from moving_out.benchmarks.moving_out import MovingOutEnv

from pettingzoo import ParallelEnv


class parallel_env(ParallelEnv):
    metadata = {
        "name": "moving_out_v0",
    }

    def __init__(
        self,
        render_mode,
        max_cycles=300,
        state_size=None,
        map_name="HandOff",
        reward_setting="dense",
        dense_rewards_setting=None,
        repeat_actions=1,
        add_noise_to_item=False,
    ):
        self.map_name = map_name
        self.env = MovingOutEnv(
            reward_setting=reward_setting,
            dense_rewards_setting=dense_rewards_setting,
            map_name=map_name,
            add_noise_to_item=add_noise_to_item,
        )

        self.possible_agents = ["robot_1", "robot_2"]
        self.agents = ["robot_1", "robot_2"]
        # self.possible_agents = 2
        self.steps = 0
        self.max_cycles = max_cycles
        self.batch_size = 10
        if(state_size is None):
            from moving_out.utils.states_encoding import StatesEncoder
            states_encoder = StatesEncoder()
            state = self.env.get_all_states()
            state_size = len(states_encoder.get_state_by_current_obs_states(state)[0])
        self.state_size = state_size
        

    def reset(self, seed=0, options=None):
        self.agents = ["robot_1", "robot_2"]
        obs = self.env.reset(map_name=self.map_name)
        self.env.seed(seed)
        self.steps = 0
        observations = {
            self.agents[0]: obs[0],
            self.agents[1]: obs[1],
        }

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        self.steps += 1
        action_0 = actions[self.agents[0]]
        if action_0[2] > 0:
            action_0[2] = True
        else:
            action_0[2] = False
        action_0 = list(action_0)
        # action_0 = list(action_0[0]) + [action_0[1]]

        action_1 = actions[self.agents[1]]
        # action_1 = list(action_1[0]) + [action_1[1]]

        if action_1[2] > 0:
            action_1[2] = True
        else:
            action_1[2] = False
        action_1 = list(action_1)

        actions = [action_0, action_1]
        for i, action in enumerate(actions):
            action[1] = action[1] * np.pi
            actions[i] = action

        obs, rew, done, _, info = self.env.step(actions)

        infos = {a: {} for a in self.agents}
        rewards = {self.agents[0]: rew[0], self.agents[1]: rew[1]}
        if self.steps >= self.max_cycles:
            reach_max_step = True
        else:
            reach_max_step = False
        truncated = {self.agents[0]: reach_max_step, self.agents[1]: reach_max_step}

        terminated = {self.agents[0]: done, self.agents[1]: done}

        observations = {
            self.agents[0]: obs[0],
            self.agents[1]: obs[1],
        }
        if done or reach_max_step:
            self.agents = []

        return observations, rewards, terminated, truncated, infos

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        observation_low = np.ones(self.state_size) * -1.2

        observation_high = np.ones(self.state_size) * 1.2

        observation_space = spaces.Box(
            low=observation_low, high=observation_high, dtype=np.float32
        )

        observation_spaces = {agent: observation_space for agent in self.agents}

        return observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # observation_low = np.array([-1, -1])
        # observation_high = np.array([1, 1])
        # discrete_action_space = spaces.Discrete(2)
        # observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        # action_space = spaces.Tuple((observation_space, discrete_action_space))
        # action_spaces = {agent: action_space for agent in self.agents}

        action_space_low = np.array([-1, -1, -1])
        action_space_high = np.array([1, 1, 1])

        action_space = spaces.Box(
            low=action_space_low, high=action_space_high, dtype=np.float32
        )
        # action_space = spaces.Tuple((observation_space, discrete_action_space))
        action_spaces = {agent: action_space for agent in self.agents}

        return action_spaces[agent]
