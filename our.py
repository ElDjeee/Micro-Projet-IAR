# Prepare the environment
try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run

    assert (
        run(["pip", "install", "easypip"]).returncode == 0
    ), "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils").setup(maze_mdp=True)
 
import math
import copy
import torch
import torch.nn as nn
from bbrl import get_arguments, get_class
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from bbrl.visu.plot_policies import plot_policy
from omegaconf import OmegaConf
from torch.distributions import Normal

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Get the current state $s_t$ and the chosen action $a_t$
        obs = self.get(("env/env_obs", t))  # shape B x D_{obs}
        action = self.get(("action", t))  # shape B x D_{action}

        # Compute the Q-value(s_t, a_t)
        obs_act = torch.cat((obs, action), dim=1)  # shape B x (D_{obs} + D_{action})
        # Get the q-value (and remove the last dimension since it is a scalar)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        q_value = self.model(obs_act)
        return q_value

class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.model(obs)

class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)

class AddOUNoise(Agent):
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)



mse = nn.MSELoss()

setup_tensorboard("./outputs/tblogs")

# Defines the (Torch) mse loss function
# `mse(x, y)` computes $\|x-y\|^2$
mse = nn.MSELoss()

def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference
    target = (
        reward[1]
        + cfg.algorithm.discount_factor * target_q_values[1] * must_bootstrap[1].int()
    )
    # Compute critic loss
    critic_loss = mse(q_values[0], target)
    return critic_loss

def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    return -q_values[0].mean()

class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")

        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic_2/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        self.target_actor = copy.deepcopy(self.actor).with_prefix("target_actor/")

        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)
        self.t_target_actor = TemporalAgent(self.target_actor)

        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)

def run_td3(td3: TD3):
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)
        terminated, reward = rb_workspace["env/terminated", "env/reward"]

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://github.com/osigaud/bbrl/blob/master/docs/time_limits.md
        must_bootstrap = ~terminated

        # Critic update
        # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)
        td3.t_critic_2(rb_workspace, t=0, n_steps=1)

        with torch.no_grad():
            # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute
            # Q(s_{t+1}, \pi(s_{t+1}) below
            td3.t_target_actor(rb_workspace, t=1, n_steps=1)
            # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
            td3.t_target_critic_1(rb_workspace, t=1, n_steps=1)
            td3.t_target_critic_2(rb_workspace, t=1, n_steps=1)

        # finally q_values contains the above collection at t=0 and t=1
        q_values_1, post_q_values_1, q_values_2, post_q_values_2 = rb_workspace[
            "critic_1/q_value", "target-critic_1/q_value", "critic_2/q_value", "target-critic_2/q_value"
        ]

        post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)

        # Compute critic loss
        critic_loss_1 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_1, post_q_values
        )
        critic_loss_2 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_2, post_q_values
        )

        td3.logger.add_log("critic_loss_1", critic_loss_1, td3.nb_steps)
        td3.logger.add_log("critic_loss_2", critic_loss_2, td3.nb_steps)
        td3.critic_optimizer_1.zero_grad()
        td3.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_1.step()
        td3.critic_optimizer_2.step()

        # Actor update

        # Now we determine the actions the current policy would take in the states from the RB
        td3.t_actor(rb_workspace, t=0, n_steps=1)

        # We determine the Q values resulting from actions of the current policy
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)
        td3.t_critic_2(rb_workspace, t=0, n_steps=1)

        # and we back-propagate the corresponding loss to maximize the Q values
        q_values_1 = rb_workspace["critic_1/q_value"]
        actor_loss = compute_actor_loss(q_values_1)

        td3.logger.add_log("actor_loss", actor_loss, td3.nb_steps)

        # if -25 < actor_loss < 0 and nb_steps > 2e5:
        td3.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.actor_optimizer.step()

        soft_update_params(
            td3.actor, td3.target_actor, td3.cfg.algorithm.tau_target
        )
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )
        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )

        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )
params = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": True,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": True,
    "algorithm": {
        "seed": 1,
        "max_grad_norm": 0.5,
        "epsilon": 0.02,
        "n_envs": 1,
        "n_steps": 100,
        "nb_evals": 10,
        "discount_factor": 0.98,
        "buffer_size": 2e5,
        "batch_size": 64,
        "tau_target": 0.05,
        "eval_interval": 2_000,
        "max_epochs": 11_000,
        # Minimum number of transitions before learning starts
        "learning_starts": 10000,
        "action_noise": 0.1,
        "architecture": {
            "actor_hidden_size": [400, 300],
            "critic_hidden_size": [400, 300],
        },
    },
    "gym_env": {
        "env_name": "LunarLanderContinuous-v2",
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-3,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-3,
    },
}

td3 = TD3(OmegaConf.create(params))
run_td3(td3)
td3.visualize_best()


# def set_seed(seed):
# 	random.seed(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
