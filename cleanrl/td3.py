import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from bbrl_utils.nn import setup_optimizer

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(), 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias



nb_seed = 15
for i in range(nb_seed):
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
    poetry run pip install "stable_baselines3==2.0.0a1"
    """
        )

    # Load configuration from params.yaml using OmegaConf
    args = OmegaConf.load("../params.yaml")

    args.algorithm.seed = i
    args.exp_name = os.path.basename(__file__)[: -len(".py")]
    args.track = args.get("track", False)
    args.torch_deterministic = args.get("torch_deterministic", True)
    args.cuda = args.get("cuda", True)
    args.save_model = args.get("save_model", False)
    args.upload_model = args.get("upload_model", False)
    args.hf_entity = args.get("hf_entity", "")
    args.capture_video = args.get("capture_video", False)
    args.policy_noise = args.get("policy_noise", 0.2)
    args.policy_frequency = args.get("policy_frequency", 2)
    args.noise_clip = args.get("noise_clip", 0.5)

    # Calculate total timesteps
    args.total_timesteps = args.algorithm.max_epochs * args.algorithm.n_steps

    run_name = f"{args.gym_env.env_name}__{args.exp_name}__{args.algorithm.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.get("wandb_project_name", "cleanRL"),
            entity=args.get("wandb_entity", None),
            sync_tensorboard=True,
            config=OmegaConf.to_container(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"outputs/sb3/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in OmegaConf.to_container(args).items()
                ]
            )
        ),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.algorithm.seed)
    np.random.seed(args.algorithm.seed)
    torch.manual_seed(args.algorithm.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.gym_env.env_name,
                args.algorithm.seed,
                0,
                args.capture_video,
                run_name,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Set up optimizers using parameters from params.yaml
    actor_optimizer = setup_optimizer(args.actor_optimizer, actor)
    q1_optimizer = setup_optimizer(args.actor_optimizer, qf1)
    q2_optimizer = setup_optimizer(args.actor_optimizer, qf2)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.algorithm.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.algorithm.seed)
    for global_step in range(int(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.algorithm.learning_starts:
            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(
                    0,
                    actor.action_scale * args.algorithm.action_noise,
                )
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return",
                    info["episode"]["r"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    info["episode"]["l"],
                    global_step,
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.algorithm.learning_starts:
            data = rb.sample(args.algorithm.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device)
                    * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0],
                    envs.single_action_space.high[0],
                )
                qf1_next_target = qf1_target(
                    data.next_observations, next_state_actions
                )
                qf2_next_target = qf2_target(
                    data.next_observations, next_state_actions
                )
                min_qf_next_target = torch.min(
                    qf1_next_target, qf2_next_target
                )
                next_q_value = (
                    data.rewards.flatten()
                    + (1 - data.dones.flatten())
                    * args.algorithm.discount_factor
                    * (min_qf_next_target).view(-1)
                )

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q1_optimizer.zero_grad()
            q2_optimizer.zero_grad()
            qf_loss.backward()
            q1_optimizer.step()
            q2_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(
                    data.observations, actor(data.observations)
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.algorithm.tau_target * param.data
                        + (1 - args.algorithm.tau_target) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.algorithm.tau_target * param.data
                        + (1 - args.algorithm.tau_target) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.algorithm.tau_target * param.data
                        + (1 - args.algorithm.tau_target) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf1_loss", qf1_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_loss", qf2_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), global_step
                )
                print(
                    "SPS:", int(global_step / (time.time() - start_time))
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    if args.save_model:
        model_path = f"outputs/sb3/runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            (actor.state_dict(), qf1.state_dict(), qf2.state_dict()),
            model_path,
        )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.gym_env.env_name,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.algorithm.action_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.gym_env.env_name}-{args.exp_name}-seed{args.algorithm.seed}"
            repo_id = (
                f"{args.hf_entity}/{repo_name}"
                if args.hf_entity
                else repo_name
            )
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "TD3",
                f"outputs/sb3/runs/{run_name}",
                f"outputs/sb3/videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
