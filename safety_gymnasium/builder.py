# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
# Copyright 2025 IDLab, University of Antwerp - imec. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This file has been modified by IDLab, University of Antwerp - imec to add video background
# distractions and color distraction functionality for Safety Gymnasium environments.
# Original work by OmniSafe Team.
"""Env builder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar
import ast

import gymnasium
import numpy as np

from safety_gymnasium import tasks
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import ResamplingError, quat2zalign
from safety_gymnasium.utils.task_utils import get_task_class_name

try:
    from safety_gymnasium.assets.distractions.video_backgrounds import SafetyGymVideoBackground
    VIDEO_BACKGROUNDS_AVAILABLE = True
except ImportError:
    VIDEO_BACKGROUNDS_AVAILABLE = False


@dataclass
class RenderConf:
    r"""Render options.

    Attributes:
        mode (str): render mode, can be 'human', 'rgb_array', 'depth_array'.
        width (int): width of the rendered image.
        height (int): height of the rendered image.
        camera_id (int): camera id to render.
        camera_name (str): camera name to render.

        Note:
            ``camera_id`` and ``camera_name`` can only be set one of them.
    """

    mode: str = None
    width: int = 256
    height: int = 256
    camera_id: int = None
    camera_name: str = None


# pylint: disable-next=too-many-instance-attributes
class Builder(gymnasium.Env, gymnasium.utils.EzPickle):
    r"""An entry point to organize different environments, while showing unified API for users.

    The Builder class constructs the basic control framework of environments, while
    the details were hidden. There is another important parts, which is **task module**
    including all task specific operation.

    Methods:

    - :meth:`_setup_simulation`: Set up mujoco the simulation instance.
    - :meth:`_get_task`: Instantiate a task object.
    - :meth:`set_seed`: Set the seed for the environment.
    - :meth:`reset`: Reset the environment.
    - :meth:`step`: Step the environment.
    - :meth:`_reward`: Calculate the reward.
    - :meth:`_cost`: Calculate the cost.
    - :meth:`render`: Render the environment.

    Attributes:

    - :attr:`task_id` (str): Task id.
    - :attr:`config` (dict): Pre-defined configuration of the environment, which is passed via
      :meth:`safety_gymnasium.register()`.
    - :attr:`render_parameters` (RenderConf): Render parameters.
    - :attr:`action_space` (gymnasium.spaces.Box): Action space.
    - :attr:`observation_space` (gymnasium.spaces.Dict): Observation space.
    - :attr:`obs_space_dict` (dict): Observation space dictionary.
    - :attr:`done` (bool): Whether the episode is done.
    """

    metadata: ClassVar[dict[str, Any]] = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 30,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        task_id: str,
        config: dict | None = None,
        render_mode: str | None = None,
        width: int = 256,
        height: int = 256,
        camera_id: int | None = None,
        camera_name: str | None = None,
    ) -> None:
        """Initialize the builder.

        Note:
            The ``camera_name`` parameter can be chosen from:
              - **human**: The camera used for freely moving around and can get input
                from keyboard real time.
              - **vision**: The camera used for vision observation, which is fixed in front of the
                agent's head.
              - **track**: The camera used for tracking the agent.
              - **fixednear**: The camera used for top-down observation.
              - **fixedfar**: The camera used for top-down observation, but is further than **fixednear**.

        Args:
            task_id (str): Task id.
            config (dict): Pre-defined configuration of the environment, which is passed via
              :meth:`safety_gymnasium.register`.
            render_mode (str): Render mode, can be 'human', 'rgb_array', 'depth_array'.
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
            camera_id (int): Camera id to render.
            camera_name (str): Camera name to render.
        """
        gymnasium.utils.EzPickle.__init__(self, config=config)

        self.task_id: str = task_id
        self.config: dict = config
        self._seed: int = None
        self._setup_simulation()

        self.first_reset: bool = None
        self.steps: int = None
        self.cost: float = None
        self.terminated: bool = True
        self.truncated: bool = False

        self.render_parameters = RenderConf(render_mode, width, height, camera_id, camera_name)
        
        # Color randomization parameters - Distractions
        self.beta_rgb = config.get('beta_rgb', 0.0)  # Difficulty scalar for color randomization
        self.change_geoms_color = config.get('change_geoms_color', 'none')  # 'none', 'static', or 'dynamic'
        self.object_filter = config.get('object_filter', 'all')  # Which assets to change colors for
        self.original_colors = None  # Store original colors for reference
        self.current_colors = None  # Track current colors for dynamic mode
        self.episode_count = 0  # Count episodes for varied color seeds
        
        # Video background parameters - Distractions
        self.video_background_path = config.get('video_background_path', 'None')  # Path to DAVIS dataset
        self.video_alpha = config.get('video_alpha', 0.7)  # Video blending alpha
        self.video_dynamic = config.get('video_dynamic', True)  # Whether to animate frames
        self.video_dataset = config.get('video_dataset', 'train')  # 'train', 'val', or list
        self.num_videos = config.get('num_videos', 10)  # Number of videos to use
        self.video_background = None  # Video background system
        
        # Process and store filters for color methods
        self._object_filters = self._get_object_filters()   
        self._object_patterns = self._get_object_patterns()

    def _setup_simulation(self) -> None:
        """Set up mujoco the simulation instance."""
        self.task = self._get_task()
        self.set_seed()

    def _get_task(self) -> BaseTask:
        """Instantiate a task object."""
        class_name = self.config.get('task_name', get_task_class_name(self.task_id))
        assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
        task_class = getattr(tasks, class_name)
        
        # Filter out Builder-specific parameters before passing to task
        task_config = self.config.copy()
        builder_specific_keys = ['change_geoms_color', 'beta_rgb', 'object_filter', 
                               'video_background_path', 'video_alpha', 'video_dynamic', 
                               'video_dataset', 'num_videos']
        for key in builder_specific_keys:
            task_config.pop(key, None)
        
        task = task_class(config=task_config)

        task.build_observation_space()
        
        # Video background will be initialized on first reset when model is available
        
        return task

    def set_seed(self, seed: int | None = None) -> None:
        """Set internal random state seeds."""
        self._seed = np.random.randint(2**32, dtype='int64') if seed is None else seed
        self.task.random_generator.set_random_seed(self._seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:  # pylint: disable=arguments-differ
        """Reset the environment and return observations."""
        info = {}

        if not self.task.mechanism_conf.randomize_layout:
            assert seed is None, 'Cannot set seed if randomize_layout=False'
            self.set_seed(0)
        elif seed is not None:
            self.set_seed(seed)

        self.terminated = False
        self.truncated = False
        self.steps = 0  # Count of steps taken in this episode
        self.episode_count += 1  # Increment episode count for color variation

        self.task.reset()
        self.task.specific_reset()
        self.task.update_world()  # refresh specific settings
        self.task.agent.reset()

        # Initialize colors for static/dynamic modes (new colors each episode)
        if self.change_geoms_color == 'static' or self.change_geoms_color == 'dynamic':
            self._initialize_episode_colors()
        
        # Initialize video background system on first reset (temporarily disabled to debug EOFError)
        if self.video_background_path != 'None' and self.video_background_path != None and VIDEO_BACKGROUNDS_AVAILABLE and self.video_background is None:
                self.video_background = SafetyGymVideoBackground(
                    dataset_path=self.video_background_path,
                    dataset_videos=self.video_dataset,
                    video_alpha=self.video_alpha,
                    dynamic=self.video_dynamic,
                    num_videos=self.num_videos,
                    seed=self._seed
                )
                self.video_background.initialize_background(self.task.model, self.task.data)

        elif self.video_background and self.video_background_path != 'None' and self.video_background_path != None:
            # Reset video background for new episode
                self.video_background.reset_episode(self.task.model, self.task.data)

        cost = self._cost()
        assert cost['cost_sum'] == 0, f'World has starting cost! {cost}'
        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return (self.task.obs(), info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        """Take a step and return observation, reward, cost, terminated, truncated, info."""
        assert not self.done, 'Environment must be reset before stepping.'
        action = np.array(action, copy=False)  # cast to ndarray
        if action.shape != self.action_space.shape:  # check action dimension
            raise ValueError('Action dimension mismatch')

        info = {}
        
        # Dynamic color changes every step
        if self.change_geoms_color == 'dynamic':
            self._update_dynamic_colors()
        # Dynamic video background changes every step
        if self.video_background and self.video_background_path != 'None' and self.video_dynamic:
                self.video_background.step_background(self.task.model, self.task.data, self.task)
        exception = self.task.simulation_forward(action)
        if exception:
            self.truncated = True
            reward = self.task.reward_conf.reward_exception
            info['cost_exception'] = 1.0
            cost = info['cost_exception']
        else:
            # Reward processing
            reward = self._reward()

            # Constraint violations
            info.update(self._cost())

            cost = info['cost_sum']

            self.task.specific_step()

            # Goal processing
            if self.task.goal_achieved:
                info['goal_met'] = True
                if self.task.mechanism_conf.continue_goal:
                    # Update the internal layout
                    # so we can correctly resample (given objects have moved)
                    self.task.update_layout()
                    # Try to build a new goal, end if we fail
                    if self.task.mechanism_conf.terminate_resample_failure:
                        try:
                            self.task.update_world()
                        except ResamplingError:
                            # Normal end of episode
                            self.terminated = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.task.update_world()
                else:
                    self.terminated = True

        # termination of death processing
        if not self.task.agent.is_alive():
            self.terminated = True

        # Timeout
        self.steps += 1
        if self.steps >= self.task.num_steps:
            self.truncated = True  # Maximum number of steps in an episode reached

        if self.render_parameters.mode == 'human':
            self.render()
        return self.task.obs(), reward, cost, self.terminated, self.truncated, info

    def _reward(self) -> float:
        """Calculate the current rewards.

        Call exactly once per step.
        """
        reward = self.task.calculate_reward()

        # Intrinsic reward for uprightness
        if self.task.reward_conf.reward_orientation:
            zalign = quat2zalign(
                self.task.data.get_body_xquat(self.task.reward_conf.reward_orientation_body),
            )
            reward += self.task.reward_conf.reward_orientation_scale * zalign

        # Clip reward
        reward_clip = self.task.reward_conf.reward_clip
        if reward_clip:
            in_range = -reward_clip < reward < reward_clip
            if not in_range:
                reward = np.clip(reward, -reward_clip, reward_clip)
                print('Warning: reward was outside of range!')

        return reward

    def _cost(self) -> dict:
        """Calculate the current costs and return a dict.

        Call exactly once per step.
        """
        cost = self.task.calculate_cost()

        # Optionally remove shaping from reward functions.
        if self.task.cost_conf.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self.cost = cost

        return cost

    
    def _get_object_filters(self) -> list:
        """Get the processed filters list for use in color methods."""
        if isinstance(self.object_filter, str):
            # Try to parse string representation of list first
            try:
                parsed = ast.literal_eval(self.object_filter)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [self.object_filter]  # Single string
            except (ValueError, SyntaxError):
                return [self.object_filter]  # Single string
        elif isinstance(self.object_filter, list):
            # Check if it's a list containing a string representation of a list
            if len(self.object_filter) == 1 and isinstance(self.object_filter[0], str):
                try:
                    parsed = ast.literal_eval(self.object_filter[0])
                    if isinstance(parsed, list):
                        return parsed
                except (ValueError, SyntaxError):
                    pass
            # Handle list directly
            return self.object_filter
        else:
            # Convert to list if it's some other iterable
            return list(self.object_filter)
    
    def _get_object_patterns(self) -> dict:
        """Get the filter patterns dictionary."""
        return {
            'hazards': lambda names: [name.startswith('hazard') for name in names],
            'vases': lambda names: [name.startswith('vase') for name in names],
            'goals': lambda names: [name.startswith('goal') for name in names],
            'walls': lambda names: [name.startswith('wall') for name in names],
            'agents': lambda names: [name in ['agent', 'left', 'right', 'rear', 'pointarrow'] or name.startswith('agent') for name in names],
            'floor': lambda names: [name == 'floor' for name in names],
            'objects': lambda names: [any(name.startswith(prefix) for prefix in ['apple', 'orange', 'button', 'pillar', 'vase']) for name in names],
            'environment': lambda names: [any(name.startswith(prefix) for prefix in ['wall', 'sigwall']) or name == 'floor' for name in names],
            'interactive': lambda names: [any(name.startswith(prefix) for prefix in ['hazard', 'goal', 'button', 'apple', 'orange', 'vase']) for name in names],
            'constraints': lambda names: [any(name.startswith(prefix) for prefix in ['hazard', 'vase']) for name in names]
        }

    def _initialize_episode_colors(self) -> None:
        """Initialize colors at episode start for static/dynamic modes.
        
        Static: Colors sampled once per episode and remain constant
        Dynamic: Colors sampled once, then updated each step with Gaussian noise
        
        Colors sampled uniformly: x0 ~ U(x - βrgb, x + βrgb)
        where x0 is sampled color and x is original color and βrgb is the difficulty scalar.
        """
        if self.beta_rgb <= 0.0:
            return
            
        # Store original colors on first call
        if self.original_colors is None:
            self.original_colors = self.task.model.geom_rgba.copy()
        
        # Get random generator for reproducible results (different each episode)
        rng = np.random.RandomState(self._seed + self.episode_count * 1000)
        # Start with original colors
        episode_colors = self.original_colors.copy()
        # Group geoms by filter pattern and assign same color to each group
        geom_names = [self.task.model.geom(i).name for i in range(self.task.model.ngeom)]
        
        # Predefined filter mappings
        filter_patterns = self._object_patterns
        # Process each filter type separately to give same color to same type
        for filter_type in self._object_filters:
            if filter_type in filter_patterns:
                # Get mask for this specific filter type
                type_mask = np.array(filter_patterns[filter_type](geom_names))
                type_indices = np.where(type_mask)[0]
                
                if len(type_indices) > 0:
                    # Sample ONE color for this entire filter type
                    # Use the first geom of this type as reference for bounds
                    reference_rgba = self.original_colors[type_indices[0]]
                    min_vals = np.maximum(0.0, reference_rgba[:3] - self.beta_rgb)
                    max_vals = np.minimum(1.0, reference_rgba[:3] + self.beta_rgb)
                    
                    # Generate ONE RGB color for all geoms of this type
                    shared_rgb = rng.uniform(min_vals, max_vals)
                    # Apply the same color to ALL geoms of this type
                    episode_colors[type_indices, :3] = shared_rgb
            else:
                # Handle direct name matching (fallback)
                matching_indices = []
                for i, name in enumerate(geom_names):
                    if name == filter_type or name.startswith(filter_type):
                        matching_indices.append(i)
                
                if matching_indices:
                    # Sample ONE color for this filter type
                    reference_rgba = self.original_colors[matching_indices[0]]
                    min_vals = np.maximum(0.0, reference_rgba[:3] - self.beta_rgb)
                    max_vals = np.minimum(1.0, reference_rgba[:3] + self.beta_rgb)
                    
                    shared_rgb = rng.uniform(min_vals, max_vals)
                    episode_colors[matching_indices, :3] = shared_rgb
        
        # Apply colors to model
        self.task.model.geom_rgba[:] = episode_colors
        
        # Store current colors for dynamic mode
        self.current_colors = episode_colors.copy()

    def _update_dynamic_colors(self) -> None:
        """Update colors dynamically each step with Gaussian noise.
        
        Dynamic update: xn = xn-1 + N(0, 0.03 * βrgb)
        Colors are clipped to never exceed βrgb distance from original color.
        """
        if self.beta_rgb <= 0.0 or self.current_colors is None:
            return
        
        
        # Gaussian noise standard deviation
        noise_std = 0.03 * self.beta_rgb
        
        # Get random generator
        rng = np.random.RandomState(self._seed + self.steps + self.episode_count * 500)
        
        # Update colors by group to maintain same color per filter type
        geom_names = [self.task.model.geom(i).name for i in range(self.task.model.ngeom)]
        
        # Predefined filter mappings
        filter_patterns = self._object_patterns
        
        # Process each filter type separately to keep same color for same type
        for filter_type in self._object_filters:
            if filter_type in filter_patterns:
                # Get indices for this specific filter type
                type_mask = np.array(filter_patterns[filter_type](geom_names))
                type_indices = np.where(type_mask)[0]
                
                if len(type_indices) > 0:
                    # Use first geom of this type as reference
                    ref_idx = type_indices[0]
                    reference_original = self.original_colors[ref_idx]
                    reference_current = self.current_colors[ref_idx]
                    
                    # Generate noise vector
                    noise = rng.normal(0, noise_std, size=3)
                    new_rgb = reference_current[:3] + noise
                    
                    # Clip to stay within beta_rgb bounds
                    min_bounds = np.maximum(0.0, reference_original[:3] - self.beta_rgb)
                    max_bounds = np.minimum(1.0, reference_original[:3] + self.beta_rgb)
                    new_rgb = np.clip(new_rgb, min_bounds, max_bounds)
                    
                    # Apply the same new color to all geoms of this type
                    self.current_colors[type_indices, :3] = new_rgb
                    self.task.model.geom_rgba[type_indices, :3] = new_rgb
            else:
                # Handle direct name matching (fallback)
                matching_indices = []
                for i, name in enumerate(geom_names):
                    if name == filter_type or name.startswith(filter_type):
                        matching_indices.append(i)
                
                if matching_indices:
                    # Use first matching geom as reference
                    ref_idx = matching_indices[0]
                    reference_original = self.original_colors[ref_idx]
                    reference_current = self.current_colors[ref_idx]
                    
                    noise = rng.normal(0, noise_std, size=3)
                    new_rgb = reference_current[:3] + noise
                    
                    min_bounds = np.maximum(0.0, reference_original[:3] - self.beta_rgb)
                    max_bounds = np.minimum(1.0, reference_original[:3] + self.beta_rgb)
                    new_rgb = np.clip(new_rgb, min_bounds, max_bounds)
                    
                    # Apply same color to all matching geoms
                    for idx in matching_indices:
                        self.current_colors[idx, :3] = new_rgb
                        self.task.model.geom_rgba[idx, :3] = new_rgb

    def render(self) -> np.ndarray | None:
        """Call underlying :meth:`safety_gymnasium.bases.underlying.Underlying.render` directly.

        Width and height in parameters are constant defaults for rendering
        frames for humans. (not used for vision)

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if render_mode is:

        - None (default): no render is computed.
        - human: render return None.
          The environment is continuously rendered in the current display or terminal. Usually for human consumption.
        - rgb_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
        - rgb_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
        - depth_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y) representing depth values for an x-by-y pixel image.
        - depth_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y), as with `depth_array`.
        """
        assert self.render_parameters.mode, 'Please specify the render mode when you make env.'
        assert (
            not self.task.observe_vision
        ), 'When you use vision envs, you should not call this function explicitly.'
        return self.task.render(cost=self.cost, **asdict(self.render_parameters))

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        """Helper to get action space."""
        return self.task.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box | gymnasium.spaces.Dict:
        """Helper to get observation space."""
        return self.task.observation_space

    @property
    def obs_space_dict(self) -> dict[str, gymnasium.spaces.Box]:
        """Helper to get observation space dictionary."""
        return self.task.obs_info.obs_space_dict

    @property
    def done(self) -> bool:
        """Whether this episode is ended."""
        return self.terminated or self.truncated

    @property
    def render_mode(self) -> str:
        """The render mode."""
        return self.render_parameters.mode

    def __deepcopy__(self, memo) -> Builder:
        """Make class instance copyable."""
        other = Builder(
            self.task_id,
            self.config,
            self.render_parameters.mode,
            self.render_parameters.width,
            self.render_parameters.height,
            self.render_parameters.camera_id,
            self.render_parameters.camera_name,
        )
        other._seed = self._seed
        other.first_reset = self.first_reset
        other.steps = self.steps
        other.cost = self.cost
        other.terminated = self.terminated
        other.truncated = self.truncated
        other.task = deepcopy(self.task)  # pylint: disable=attribute-defined-outside-init
        return other
