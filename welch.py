import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_ind
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# See bbrl.stats
def compute_central_tendency_and_error(id_central, id_error, sample):
    try:
        id_error = int(id_error)
    except Exception:
        pass

    if id_central == "mean":
        central = np.nanmean(sample, axis=1)
    elif id_central == "median":
        central = np.nanmedian(sample, axis=1)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=1)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=1)
    elif id_error == "std":
        low = central - np.nanstd(sample, axis=1)
        high = central + np.nanstd(sample, axis=1)
    elif id_error == "sem":
        low = central - np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
        high = central + np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
    else:
        raise NotImplementedError

    return central, low, high

class Test(ABC):
    def plot(
        self,
        data1,  # array of performance of dimension (n_steps, n_seeds) for alg 1
        data2,  # array of performance of dimension (n_steps, n_seeds) for alg 2
        point_every=1,  # evaluation frequency, one datapoint every X steps/episodes
        confidence_level=0.01,  # confidence level alpha of the test
        id_central="median",  # id of the central tendency ('mean' or 'median')
        id_error=80,  # id of the error areas ('std', 'sem', or percentiles in ]0, 100]
        legends="alg 1/alg 2",  # labels of the two input vectors
        xlabel="training steps",  # label of the x axis
        save=True,  # save in ./plot.png if True
        downsampling_fact=5,  # factor of downsampling on the x-axis for visualization purpose (increase for smoother plots)
    ):
        assert (
            data1.ndim == 2
        ), "data should be an array of dimension 2 (n_steps, n_seeds)"
        assert (
            data2.ndim == 2
        ), "data should be an array of dimension 2 (n_steps, n_seeds)"

        nb_steps = max(data1.shape[0], data2.shape[0])
        steps = [0]
        while len(steps) < nb_steps:
            steps.append(steps[-1] + point_every)
        steps = np.array(steps)
        if steps is not None:
            assert (
                steps.size == nb_steps
            ), "x should be of the size of the longest data array"

        sample_size1 = data1.shape[1]
        sample_size2 = data2.shape[1]

        # downsample for visualization purpose
        sub_steps = np.arange(0, nb_steps, downsampling_fact)
        steps = steps[sub_steps]
        nb_steps = steps.size

        # handle arrays of different lengths by padding with nans
        sample1 = np.zeros([nb_steps, sample_size1])
        sample1.fill(np.nan)
        sample2 = np.zeros([nb_steps, sample_size2])
        sample2.fill(np.nan)
        sub_steps1 = sub_steps[: data1.shape[0] // downsampling_fact]
        sub_steps2 = sub_steps[: data2.shape[0] // downsampling_fact]
        sample1[: data1[sub_steps1, :].shape[0], :] = data1[sub_steps1, :]
        sample2[: data2[sub_steps2, :].shape[0], :] = data2[sub_steps2, :]

        # test
        sign_diff = np.zeros([len(steps)])
        for i in range(len(steps)):
            sign_diff[i] = self.run_test(
                sample1[i, :].squeeze(), sample2[i, :].squeeze(), alpha=confidence_level
            )

        central1, low1, high1 = compute_central_tendency_and_error(
            id_central, id_error, sample1
        )
        central2, low2, high2 = compute_central_tendency_and_error(
            id_central, id_error, sample2
        )

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        lab1 = plt.xlabel(xlabel)
        lab2 = plt.ylabel("performance")

        plt.plot(steps, central1, linewidth=10)
        plt.plot(steps, central2, linewidth=10)
        plt.fill_between(steps, low1, high1, alpha=0.3)
        plt.fill_between(steps, low2, high2, alpha=0.3)
        splitted = legends.split("/")
        leg = ax.legend((splitted[0], splitted[1]), frameon=False)

        # plot significative difference as dots
        idx = np.argwhere(sign_diff == 1)
        y = max(np.nanmax(high1), np.nanmax(high2))
        plt.scatter(
            steps[idx], y * 1.05 * np.ones([idx.size]), s=100, c="k", marker="o"
        )

        # style
        for line in leg.get_lines():
            line.set_linewidth(10.0)
        ax.spines["top"].set_linewidth(5)
        ax.spines["right"].set_linewidth(5)
        ax.spines["bottom"].set_linewidth(5)
        ax.spines["left"].set_linewidth(5)

        if save:
            plt.savefig(
                "./plot.png",
                bbox_extra_artists=(leg, lab1, lab2),
                bbox_inches="tight",
                dpi=100,
            )

        plt.show()

    @abstractmethod
    def run_test(self, data1, data2, alpha=0.05):
        """
        Compute tests comparing data1 and data2 with confidence level alpha
        :param data1: (np.ndarray) sample 1
        :param data2: (np.ndarray) sample 2
        :param alpha: (float) confidence level of the test
        :return: (bool) if True, the null hypothesis is rejected
        """

class WelchTTest(Test):
    """Welch t-test (recommended)"""

    def run_test(self, data1, data2, alpha=0.05):
        _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha


def BBRL_DATA():
    directories = [
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S1_20241010-193749/',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S1_20241010-193749',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S2_20241010-194326',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S3_20241010-194858',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S4_20241010-195508',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S5_20241010-200113',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S6_20241010-200716',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S7_20241010-201256',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S8_20241010-201605',
        'our/outputs/tblogs/LunarLanderContinuous-v2/td3-S9_20241010-201542'
    ]
    
    log_data = {}
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_acc = EventAccumulator(os.path.join(root, file))
                    event_acc.Reload()
                    
                    tags = event_acc.Tags()['scalars']
                    
                    for tag in tags:
                        events = event_acc.Scalars(tag)
                        steps = [event.step for event in events]
                        values = [event.value for event in events]
                        
                        if tag not in log_data:
                            log_data[tag] = []
                        log_data[tag].append(values)

    return log_data



def CLEANRL_DATA():
    log_files = [
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__0__1728424315/events.out.tfevents.1728424319.MacBook-Air-de-Yanis-3.local.41135.0',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__1__1728430126/events.out.tfevents.1728430126.MacBook-Air-de-Yanis-3.local.41135.1',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__2__1728436417/events.out.tfevents.1728436417.MacBook-Air-de-Yanis-3.local.41135.2',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__3__1728442291/events.out.tfevents.1728442291.MacBook-Air-de-Yanis-3.local.41135.3',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__4__1728447219/events.out.tfevents.1728447219.MacBook-Air-de-Yanis-3.local.41135.4',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__5__1728452214/events.out.tfevents.1728452214.MacBook-Air-de-Yanis-3.local.41135.5',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__6__1728456823/events.out.tfevents.1728456823.MacBook-Air-de-Yanis-3.local.41135.6',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__7__1728461952/events.out.tfevents.1728461952.MacBook-Air-de-Yanis-3.local.41135.7',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__8__1728467113/events.out.tfevents.1728467113.MacBook-Air-de-Yanis-3.local.41135.8',
        'cleanrl/outputs/sb3/runs/LunarLanderContinuous-v2__cleanrl__9__1728472128/events.out.tfevents.1728472128.MacBook-Air-de-Yanis-3.local.41135.9',
    ]
    
    all_log_data = {}
    
    for log_file in log_files:
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()
        
        # Obtenir tous les tags scalaires
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            if tag not in all_log_data:
                all_log_data[tag] = []
            all_log_data[tag].append(values)

    return all_log_data



def SB3_DATA():
    log_files = [
        'sb3/LunarLanderContinuous-v2/td3-S0_2024-10-08_23-57-54/sb3/tblogs/TD3_1/events.out.tfevents.1728424677.MacBook-Air-de-Yanis-3.local.41568.0',
        'sb3/LunarLanderContinuous-v2/td3-S2_2024-10-09_04-45-18/sb3/tblogs/TD3_1/events.out.tfevents.1728441918.MacBook-Air-de-Yanis-3.local.41568.2',
        'sb3/LunarLanderContinuous-v2/td3-S3_2024-10-09_06-42-14/sb3/tblogs/TD3_1/events.out.tfevents.1728448934.MacBook-Air-de-Yanis-3.local.41568.3',
        'sb3/LunarLanderContinuous-v2/td3-S4_2024-10-09_08-33-03/sb3/tblogs/TD3_1/events.out.tfevents.1728455583.MacBook-Air-de-Yanis-3.local.41568.4',
        'sb3/LunarLanderContinuous-v2/td3-S5_2024-10-09_10-34-06/sb3/tblogs/TD3_1/events.out.tfevents.1728462847.MacBook-Air-de-Yanis-3.local.41568.5',
        'sb3/LunarLanderContinuous-v2/td3-S6_2024-10-09_12-27-22/sb3/tblogs/TD3_1/events.out.tfevents.1728469642.MacBook-Air-de-Yanis-3.local.41568.6',
        'sb3/LunarLanderContinuous-v2/td3-S7_2024-10-09_14-22-28/sb3/tblogs/TD3_1/events.out.tfevents.1728476548.MacBook-Air-de-Yanis-3.local.41568.7',
        'sb3/LunarLanderContinuous-v2/td3-S8_2024-10-09_16-18-07/sb3/tblogs/TD3_1/events.out.tfevents.1728483487.MacBook-Air-de-Yanis-3.local.41568.8',
        'sb3/LunarLanderContinuous-v2/td3-S9_2024-10-09_18-08-31/sb3/tblogs/TD3_1/events.out.tfevents.1728490111.MacBook-Air-de-Yanis-3.local.41568.9',
        'sb3/LunarLanderContinuous-v2/td3-S10_2024-10-09_20-02-13/sb3/tblogs/TD3_1/events.out.tfevents.1728496933.MacBook-Air-de-Yanis-3.local.41568.10'
    ]

    all_log_data = {}
    
    for log_file in log_files:
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()
        
        # Obtenir tous les tags scalaires
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            if tag not in all_log_data:
                all_log_data[tag] = []
            all_log_data[tag].append(values)

    return all_log_data


bbrl = BBRL_DATA()
cleanrl = CLEANRL_DATA()
sb3 = SB3_DATA()


min_len_actor_loss = min(
    min([len(x) for x in bbrl['actor_loss']]),
    min([len(x) for x in cleanrl['losses/actor_loss']]),
    min([len(x) for x in sb3['train/actor_loss']])
)
min_len_reward = min(
    min([len(x) for x in bbrl['reward/mean']]),
    min([len(x) for x in cleanrl['charts/episodic_return']]),
    min([len(x) for x in sb3['eval/mean_reward']])
)
min_len_critic_loss = min(
    min([len(x) for x in bbrl['critic_loss']]),
    min([len(x) for x in cleanrl['losses/qf_loss']]),
    min([len(x) for x in sb3['train/critic_loss']])
)

bbrl_actor_loss = torch.Tensor([x[:min_len_actor_loss] for x in bbrl['actor_loss']]).t()
bbrl_reward = torch.Tensor([x[:min_len_reward] for x in bbrl['reward/mean']]).t()
bbrl_critic_loss = torch.Tensor([x[:min_len_critic_loss] for x in bbrl['critic_loss']]).t()
cleanrl_actor_loss = torch.Tensor([x[:min_len_actor_loss] for x in cleanrl['losses/actor_loss']]).t()
cleanrl_reward = torch.Tensor([x[:min_len_reward] for x in cleanrl['charts/episodic_return']]).t()
cleanrl_critic_loss = torch.Tensor([x[:min_len_critic_loss] for x in cleanrl['losses/qf_loss']]).t()
sb3_actor_loss = torch.Tensor([x[:min_len_actor_loss] for x in sb3['train/actor_loss']]).t()
sb3_reward = torch.Tensor([x[:min_len_reward] for x in sb3['eval/mean_reward']]).t()
sb3_critic_loss = torch.Tensor([x[:min_len_critic_loss] for x in sb3['train/critic_loss']]).t()

WelchTTest().plot(
    bbrl_reward,
    sb3_reward,
    legends="Reward: bbrl/sb3 ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    bbrl_reward,
    cleanrl_reward,
    legends="Reward: bbrl/cleanrl ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    sb3_reward,
    cleanrl_reward,
    legends="Reward: sb3/cleanrl ; En fonction des steps",
    save=False,
)

WelchTTest().plot(
    bbrl_actor_loss,
    sb3_actor_loss,
    legends="Actor: bbrl/sb3 ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    bbrl_actor_loss,
    cleanrl_actor_loss,
    legends="Actor: bbrl/cleanrl ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    sb3_actor_loss,
    cleanrl_actor_loss,
    legends="Actor: sb3/cleanrl ; En fonction des steps",
    save=False,
)

WelchTTest().plot(
    bbrl_critic_loss,
    sb3_critic_loss,
    legends="Critic: bbrl/sb3 ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    bbrl_critic_loss,
    cleanrl_critic_loss,
    legends="Critic: bbrl/cleanrl ; En fonction des steps",
    save=False,
)
WelchTTest().plot(
    sb3_critic_loss,
    cleanrl_critic_loss,
    legends="Critic: sb3/cleanrl ; En fonction des steps",
    save=False,
)
