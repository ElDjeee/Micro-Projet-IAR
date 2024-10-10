import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from omegaconf import OmegaConf
from datetime import datetime
import os

def run_td3(config):
    seed = config.algorithm.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_name = config.gym_env.env_name
    env = gym.make(env_name)
    env.reset(seed=seed)

    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=config.algorithm.action_noise * np.ones(n_actions)
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=config.algorithm.architecture.actor_hidden_size,
            qf=config.algorithm.architecture.critic_hidden_size,
        ),
        activation_fn=torch.nn.ReLU,
    )

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = f"{env_name}/td3-S{seed}_{current_time}"

    os.makedirs(base_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=base_dir if config.save_best else None,
        log_path=base_dir,
        eval_freq=config.algorithm.eval_interval,
        deterministic=True,
        render=False,
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=config.actor_optimizer.lr,
        buffer_size=config.algorithm.buffer_size,
        batch_size=config.algorithm.batch_size,
        gamma=config.algorithm.discount_factor,
        tau=config.algorithm.tau_target,
        train_freq=1,
        gradient_steps=1,
        learning_starts=config.algorithm.learning_starts,
        seed=seed,
        verbose=1,
        tensorboard_log=f"{base_dir}/sb3/tblogs/",
    )

    total_timesteps = config.algorithm.max_epochs * config.algorithm.n_steps

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(f"{base_dir}/sb3/td3_lunarlander")

    if config.collect_stats:
        if config.save_best:
            best_model = TD3.load(f"{base_dir}/sb3/best_model")
        else:
            best_model = model

        mean_reward, std_reward = evaluate_policy(
            best_model, eval_env, n_eval_episodes=config.algorithm.nb_evals, deterministic=True
        )
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        if config.plot_agents:
            obs, _ = eval_env.reset()
            for _ in range(1000):
                action, _states = best_model.predict(obs, deterministic=True)
                obs, reward, done, truncation, info = eval_env.step(action)
                eval_env.render()
                if done or truncation:
                    obs, _ = eval_env.reset()
            eval_env.close()



nb_seed = 15
for i in range(nb_seed):
    config = OmegaConf.load('../params.yaml')
    config = OmegaConf.merge({"algorithm" : {"seed": i}}, config)
    run_td3(config)
