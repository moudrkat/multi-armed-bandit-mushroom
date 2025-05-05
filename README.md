🎰 Multi-Armed Bandit Simulation with Thompson Sampling
This is a simple interactive app that simulates a multi-armed bandit (MAB) scenario using Thompson Sampling, built with Streamlit.

The goal: to explore how reinforcement learning algorithms can optimize decision-making in real time — particularly in a marketing context, like choosing the most effective ad image to show to users.

🧠 Background
This project was inspired by a talk at a machine learning conference in Prague about using AI in e-commerce. I wanted to understand how multi-armed bandit algorithms work in practice, so I built this small experiment.

The simulation is framed as a toy marketing problem:

Imagine you want to sell mushroom-based products during a successful foraging season. You generate a variety of mushroom images to use in ads — but you don’t know which one will perform best.

You want users to click on the ad (maximize CTR), but traditional A/B testing wastes time and shows bad variants to too many users — and by the time you get results, the mushroom season might be over 🍄.

🚀 What the App Does
Randomly generates a small set of mushroom images using a custom VAE-based image generator.

Each image corresponds to a point in a 2D latent space, which is interpretable and interpolatable.

The optimal image is located at coordinate [0, 0] in this space — the closer an image is to that point, the higher the probability of a "click" (reward = 1).

The algorithm doesn’t know the reward probabilities in advance and must learn by interacting with the environment.

Thompson Sampling is used to balance exploration and exploitation as it chooses which image to show next.

📐 How the Simulation Works (Theory)
Each image is represented by a 2D coordinate in latent space.

The true reward (click) probability is inversely proportional to the Euclidean distance from [0, 0].

The simulation generates binary outcomes (0 or 1) stochastically based on this probability.

The Thompson Sampling agent gradually learns to select images closer to the ideal latent vector by updating beta distributions for each arm (image).


🔗 Try It Out
👉 multi-shroomed-bandit.streamlit.app

🤖 Business Relevance
This kind of bandit-based optimization is used in:

Ad image or copy optimization (CTR maximization)

Personalized content selection

Online recommendation systems

Adaptive A/B/n testing in real-time

⚠️ Disclaimer
This project was created purely out of curiosity and for educational purposes. I am not planning to launch a black market for mushrooms 🍄 — and if I were, I’d definitely use a slightly modified algorithm. 😉

