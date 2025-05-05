# mab.py
import numpy as np

class MAB:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.true_arms = [np.random.randn(2) for _ in range(n_arms)]
        self.alpha = np.ones(n_arms)
        self.beta_vals = np.ones(n_arms)

        self.chosen_arms = []
        self.rewards = []
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.cumulative_rewards = []

        self.t = 0

    def euclidean_reward(self, x):
        distance = np.linalg.norm(x)
        return 1 / (1 + distance)

    def step(self):
        sampled_probs = [np.random.beta(self.alpha[i], self.beta_vals[i]) for i in range(self.n_arms)]
        chosen_arm = np.argmax(sampled_probs)
        arm_vector = self.true_arms[chosen_arm]
        reward = self.euclidean_reward(arm_vector)
        binary_reward = np.random.rand() < reward

        if binary_reward:
            self.alpha[chosen_arm] += 1
        else:
            self.beta_vals[chosen_arm] += 1

        self.chosen_arms.append(chosen_arm)
        self.rewards.append(reward)
        self.rewards_per_arm[chosen_arm].append(reward)
        self.cumulative_rewards.append(np.mean(self.rewards))

        self.t += 1
        return {
            "t": self.t,
            "chosen_arm": chosen_arm,
            "arm_vector": arm_vector,
            "reward": reward,
            "binary_reward": binary_reward
        }

    def get_data(self):
        return {
            "true_arms": self.true_arms,
            "alpha": self.alpha,
            "beta_vals": self.beta_vals,
            "chosen_arms": self.chosen_arms,
            "rewards": self.rewards,
            "rewards_per_arm": self.rewards_per_arm,
            "cumulative_rewards": self.cumulative_rewards
        }