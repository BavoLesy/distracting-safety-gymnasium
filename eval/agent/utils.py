# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
import cv2


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs



class ActionRepeat:
    """Action repeat wrapper for safety gymnasium environments."""
    
    def __init__(self, env, action_repeat):
        self.env = env
        self.action_repeat = action_repeat
        self._last_action = None
        self._step_count = 0
        self._episode_steps = 0
        
        # Validate action_repeat
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")
        
        print(f"ActionRepeat wrapper initialized with action_repeat={action_repeat}")
        
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)
    
    def reset(self, **kwargs):
        """Reset the environment and action repeat state."""
        self._last_action = None
        self._step_count = 0
        self._episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step the environment with action repeat."""
        self._last_action = action
        self._step_count = 0
        
        total_reward = 0
        total_cost = 0
        done = False
        info = {}
        obs = None
        
        # Repeat the action for action_repeat steps
        for i in range(self.action_repeat):
            if done:
                break
                
            obs, reward, cost, done, info = self.env.step(action)
            total_reward += reward
            total_cost += cost
            self._step_count += 1
            self._episode_steps += 1
            
        # Add action repeat info to the info dict
        if info is None:
            info = {}
        info['action_repeat_steps'] = self._step_count
        info['episode_steps'] = self._episode_steps
            
        return obs, total_reward, total_cost, done, info
    
    @property
    def action_space(self):
        """Return the action space of the wrapped environment."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Return the observation space of the wrapped environment."""
        return self.env.observation_space
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def seed(self, seed=None):
        """Set the random seed."""
        return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
    
    def get_action_repeat(self):
        """Get the action repeat value."""
        return self.action_repeat
    
    def get_step_count(self):
        """Get the current step count for the last action."""
        return self._step_count
    
    def get_episode_steps(self):
        """Get the total episode steps."""
        return self._episode_steps



class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False, num_samples=None):

        if num_samples is None:
            num_samples = self.batch_size
        
        #print("number of samples: ", num_samples)

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=num_samples
        )

        #print("next index: ", self.idx)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end

class ReplayBufferSafety(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.curr_cost = np.empty((capacity, 1), dtype=np.float32)
        self.cost = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, curr_cost, cost, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.curr_cost[self.idx], curr_cost)
        np.copyto(self.cost[self.idx], cost)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False, num_samples=None):

        if num_samples is None:
            num_samples = self.batch_size
        

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=num_samples
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        curr_cost = torch.as_tensor(self.curr_cost[idxs], device=self.device)
        cost = torch.as_tensor(self.cost[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, curr_reward, rewards, next_obses, curr_cost, cost, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, curr_cost, cost, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.cost[self.last_save:self.idx],
            self.curr_cost[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.cost[start:end] = payload[6]
            self.cost[start:end] = payload[7]
            self.idx = end

class FrameStack(gym.Wrapper):
    """Frame stacking and resizing in one function."""
    def __init__(self, env, k, resize_shape=(64, 64), vision_key='vision'):
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.resize_shape = resize_shape
        self.vision_key = vision_key
        
        if hasattr(env, 'observation_space') and isinstance(env.observation_space, gym.spaces.Dict):
            if self.vision_key == 'vision_front_back':
                channels = 6
                original_shape = (resize_shape[0], resize_shape[1], channels)
            else:
                key = self.vision_key if self.vision_key in env.observation_space.spaces else 'vision'
                original_shape = env.observation_space[key].shape
                channels = original_shape[-1]
        else:
            original_shape = env.observation_space.shape
            channels = 6 if self.vision_key == 'vision_front_back' else (3 if len(original_shape) == 1 else original_shape[-1])
        print("original shape: ", original_shape)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(resize_shape[0], resize_shape[1], channels * k),
            dtype=np.uint8
        )

    def reset(self):
        reset_obs = self.env.reset()
        obs = self._get_vision_obs(reset_obs)
        
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._frames.append(self._get_vision_obs(obs))
        return self._get_obs(), reward, cost, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        obs = np.concatenate(list(self._frames), axis=-1)
        return obs

    def _get_vision_obs(self, obs):
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, _ = obs

        def ensure_hwc_rgb(img):
            if img is None:
                return None
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            return img

        if self.vision_key == 'vision_front_back':
            front = None
            back = None

            if isinstance(obs, dict):
                if 'image' in obs and 'image2' in obs:
                    front = obs['image']
                    back = obs['image2']
                else:
                    if 'vision' in obs:
                        front = obs['vision']
                    if 'vision_back' in obs:
                        back = obs['vision_back']

            if (front is None or back is None) and hasattr(self.env, 'task') and hasattr(self.env.task, 'render'):
                try:
                    if front is None:
                        front = self.env.task.render(width=self.resize_shape[0], height=self.resize_shape[1], mode='rgb_array', camera_name='vision', cost={})
                    if back is None:
                        back = self.env.task.render(width=self.resize_shape[0], height=self.resize_shape[1], mode='rgb_array', camera_name='vision_back', cost={})
                except Exception:
                    pass

            if front is None and hasattr(self.env, 'render'):
                try:
                    front = self.env.render()
                except Exception:
                    front = None
            if back is None:
                back = front

            front = ensure_hwc_rgb(front)
            back = ensure_hwc_rgb(back)
            if front is None:
                front = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)
            if back is None:
                back = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)

            front = cv2.resize(front, self.resize_shape, interpolation=cv2.INTER_AREA)
            back = cv2.resize(back, self.resize_shape, interpolation=cv2.INTER_AREA)
            vision6 = np.concatenate([front, back], axis=-1)
            return vision6

        
        vision = None

        
        if isinstance(obs, dict):
            key = self.vision_key if self.vision_key in obs else ('vision' if 'vision' in obs else None)
            if key is not None:
                vision = obs[key]

        # Fallback: use rendered frame from environment (RGB)
        if vision is None and hasattr(self.env, 'render'):
            try:
                frame = self.env.render()
                if frame is not None:
                    vision = frame
            except Exception:
                vision = None


        vision = ensure_hwc_rgb(vision)
        resized_vision = cv2.resize(vision, self.resize_shape, interpolation=cv2.INTER_AREA)
        return resized_vision