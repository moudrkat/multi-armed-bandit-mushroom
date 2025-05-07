import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

# Apply the black background style globally
plt.rcParams.update({
    'axes.facecolor': 'black',  # Background of the axes
    'axes.edgecolor': 'white',  # Border color of the axes
    'axes.labelcolor': 'white',  # Color of labels (x and y)
    'xtick.color': 'white',  # Color of x ticks
    'ytick.color': 'white',  # Color of y ticks
    'figure.facecolor': 'black',  # Background of the figure
    'figure.edgecolor': 'black',  # Border color of the figure
    'grid.color': 'white',  # Grid color
    'text.color': 'white',  # Text color
    'legend.facecolor': 'black',  # Legend background color
    'legend.edgecolor': 'white',  # Legend border color
    'legend.fontsize': 10,  # Legend font size
})

def plot_learning_curve(cumulative_rewards, max_steps=500):
    fig, ax = plt.subplots(figsize=(10, 2.5))  # Wide and short
    ax.plot(cumulative_rewards, label='Cumulative Avg Reward', color='white', linewidth=2)
    ax.set_xlim(0, max_steps)
    ax.set_xlabel("Round")
    ax.set_ylabel("Average Reward")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_latent_selection(true_arms, chosen_arms):
    selection_counts = np.bincount(chosen_arms, minlength=len(true_arms))
    xs = [vec[0] for vec in true_arms]
    ys = [vec[1] for vec in true_arms]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xs, ys, c=selection_counts, cmap='plasma', s=500, edgecolors='black')
    fig.colorbar(sc, ax=ax, label='Selection Count')

    # Mark the ideal latent vector at (0, 0)
    ax.scatter(0, 0, color='white', edgecolor='white', s=500, marker='x', linewidths=4, label='Ideal Mushroom')

    # Highlight the most recently chosen arm
    last_chosen_idx = chosen_arms[-1]
    last_x, last_y = true_arms[last_chosen_idx]
    ax.scatter(last_x, last_y, s=500, color='none', edgecolors='white', linewidths=4, marker='o', label='Selected Mushroom', zorder=5)

    # Add labels for each arm (mushroom)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, f'{i+1}', color='black', fontsize=20, ha='center', va='center', fontweight='bold')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("z1", fontsize=12)
    ax.set_ylabel("z2", fontsize=12)
    ax.grid(True)
    ax.legend(labelspacing=1.0, markerscale=0.5) 
    return fig

def plot_reward_distributions(rewards_per_arm):
    fig, ax = plt.subplots()
    for i, rewards in enumerate(rewards_per_arm):
        ax.hist(rewards, bins=20, alpha=0.5, label=f'Arm {i}')
    ax.set_title("Reward Distributions per Arm")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    return fig

def plot_posteriors(alpha, beta_vals, chosen_arms):
    n_arms = len(alpha)
    cols = n_arms
    rows = 1

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 5))

    # Ensure axs is iterable
    if n_arms == 1:
        axs = [axs]

    x = np.linspace(0, 1, 500)
    last_chosen_idx = chosen_arms[-1]  # Index of last chosen arm

    for i in range(n_arms):
        ax = axs[i]
        a, b = alpha[i], beta_vals[i]
        y = beta_dist.pdf(x, a, b)

        ax.plot(x, y, label=f'Beta({int(a)}, {int(b)})', color='white', linewidth=3)
        ax.set_title(f"Mushroom {i+1}")
        # ax.set_xlabel("p (success probability)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Highlight border if this is the last chosen arm
        if i == last_chosen_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor('white')  # or 'red', etc.
                spine.set_linewidth(6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig