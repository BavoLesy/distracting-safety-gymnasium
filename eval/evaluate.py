#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium
from collections import deque

from agent.sac_ae import Actor
import agent.utils as utils


class SAIRAgent:
    def __init__(self, obs_shape, action_shape, device, **kwargs):
        self.device = device
        self.actor = Actor(
            obs_shape, action_shape, kwargs.get('hidden_dim', 256),
            kwargs.get('encoder_type', 'pixel'), kwargs.get('encoder_feature_dim', 100),
            kwargs.get('actor_log_std_min', -5), kwargs.get('actor_log_std_max', 2),
            kwargs.get('num_layers', 4), kwargs.get('num_filters', 32),
            kwargs.get('encoder_stride', 1)
        ).to(device)
    
    def select_action(self, obs):
        """Select action for evaluation."""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            if obs.dim() != 4:
                obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()
    
    def load(self, model_dir, distractions):
        model_path = os.path.join(model_dir, f'model_{distractions}.pt')
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded actor from {model_path}")
        else:
            raise FileNotFoundError(f"Actor weights not found at {model_path}")


def create_environment(domain_name, trained_model='none',color_distractions='none', frame_stack=3, action_repeat=4, video_dynamic=None, difficulty=1, video_background_path=None, seed=42):
    if video_dynamic is not None:
        video_dynamic_config = video_dynamic
    else:
        video_background_path = None
        video_dynamic_config = False
    config = {
        'change_geoms_color': color_distractions,
        'observe_vision': True,
        'vision_env_conf.vision_size': (64, 64),
        'beta_rgb': difficulty,
        'object_filter': ['hazards', 'walls', 'vases', 'agent', 'floor'],
        'video_background_path': video_background_path,
        'video_dynamic': video_dynamic_config,
        'video_dataset': 'train',
        'num_videos': 10,
        'video_alpha': difficulty
    }
    env = safety_gymnasium.make(
        domain_name,
        render_mode='rgb_array',
        camera_name='vision_front_back',
        config=config
    )
    env.set_seed(seed)
    
    # Apply wrappers
    env = utils.FrameStack(env, k=frame_stack, resize_shape=(64, 64), vision_key='vision_front_back')
    env = utils.ActionRepeat(env, action_repeat)
    return env


def evaluate_episode(env, agent, seed, episode_idx):
    """Evaluate a single episode."""
    obs = env.reset() #seed=seed
    done = False
    episode_reward = 0
    episode_cost = 0
    episode_length = 0
    episode_cost_rate = 0
    
    while not done and episode_length < 1000:
        action = agent.select_action(obs)
        obs, reward, cost, done, info = env.step(action)
        if cost > 0:
            episode_cost_rate += 1
        episode_reward += reward
        episode_cost += cost
        episode_length += 1
    return {
        'episode_idx': episode_idx,
        'total_reward': episode_reward,
        'total_cost': episode_cost,
        'episode_length': episode_length,
        'cost_rate': episode_cost_rate / episode_length
    }


def main():
    parser = argparse.ArgumentParser(description='SAIR Evaluation')
    
    # Required arguments
    parser.add_argument('--model_dir', required=True, help='Directory containing actor weights')
    
    # Environment
    parser.add_argument('--domain_name', default='SafetyPointGoal1-v0', help='Environment name')
    parser.add_argument('--trained_model', default='none', help='Model trained on which distractions', choices=['none', 'static', 'dynamic'])
    parser.add_argument('--action_repeat', default=4, type=int, help='Action repeat factor')
    parser.add_argument('--frame_stack', default=3, type=int, help='Frame stack')
    parser.add_argument('--difficulty', default=1, type=float, help='Difficulty level for color distractions')
    parser.add_argument('--video_distractions', default='none', help='Video dynamic or static', choices=['none', 'static', 'dynamic'])
    parser.add_argument('--video_background_path', default='/app/DAVIS/JPEGImages/480p', help='Path to video backgrounds')
    parser.add_argument('--color_distractions', default='none', help='Distraction type', choices=['none', 'static', 'dynamic'])
    parser.add_argument('--encoder_feature_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--actor_log_std_min', default=-5, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--encoder_stride', default=1, type=int)
    
    # Evaluation
    parser.add_argument('--num_episodes', default=50, type=int, help='Number of episodes')
    parser.add_argument('--output_dir', default='./eval_results', help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    args = parser.parse_args()
    if args.video_distractions == 'static':
        video_dynamic = False
    elif args.video_distractions == 'dynamic':
        video_dynamic = True
    else:
        video_dynamic = None

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(
        args.domain_name, args.trained_model, args.color_distractions, args.frame_stack, args.action_repeat, video_dynamic, args.difficulty, args.video_background_path, args.seed
    )
    
    # Get observation and action shapes
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    # Create minimal agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = SAIRAgent(
        obs_shape, action_shape, device,
        encoder_type='pixel',
        encoder_feature_dim=args.encoder_feature_dim,
        hidden_dim=args.hidden_dim,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        encoder_stride=args.encoder_stride
    )
    
    # Load actor weights
    agent.load(args.model_dir, args.trained_model)
    
    print(f"Evaluating agent for {args.num_episodes} episodes...")
    
    # Run evaluation
    results = []
    for episode_idx in range(args.num_episodes):
        episode_result = evaluate_episode(env, agent, args.seed, episode_idx)
        results.append(episode_result)
        
        print(f"Episode {episode_idx + 1}: Reward={episode_result['total_reward']:.2f}, "
              f"Cost={episode_result['total_cost']:.2f}, "
              f"Length={episode_result['episode_length']}, "
              f"Cost Rate={episode_result['cost_rate']:.2f}")
    
    env.close()
    
    # Compute statistics
    rewards = [r['total_reward'] for r in results]
    costs = [r['total_cost'] for r in results]
    lengths = [r['episode_length'] for r in results]
    
    stats = {
        'num_episodes': len(results),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'avg_cost_rate': np.mean([r['cost_rate'] for r in results]),
        'std_cost_rate': np.std([r['cost_rate'] for r in results])
    }
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'statistics': stats}, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Average Cost Rate: {stats['avg_cost_rate']:.3f} ± {stats['std_cost_rate']:.3f}")
    print(f"Average Reward: {stats['avg_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"Average Cost: {stats['avg_cost']:.3f} ± {stats['std_cost']:.3f}")
    print(f"Average Length: {stats['avg_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
