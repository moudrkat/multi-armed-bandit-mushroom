# app.py

import streamlit as st
import time
from mab import MAB
import plotting
from vae_decoder import load_vae_model, generate_image_from_latent

# --- UI ---
st.title("The Multi-Shroomed Bandit")
st.markdown("**Thompson Sampling in Latent Space (VAE) Simulation**")

st.markdown("The algorithm aims to identify the mushroom most similar to the ideal one among a set of randomly sampled candidates.")

with st.expander("How the Algorithm Works"):
    st.markdown("""
    - This app simulates a **multi-armed bandit problem** using **Thompson Sampling**.
    - Each "arm" is a **mushroom image**, represented by a random point in a 2D **latent space**.
    - The **closer an image is to the center (0.0)**, the higher its chance of getting a "click".
    - Thompson Sampling keeps a **Beta distribution for each image** and:
      1. Samples from each distribution.
      2. Selects the image with the highest sampled value.
      3. Updates distributions based on success (click) or failure (no click).
    - Over time, the algorithm **learns which image performs best**, while still exploring alternatives.
    """)

# --- Controls ---
c1, c2 = st.columns(2)
with c1:
    number_of_mushrooms = st.slider("Number of arms (mushrooms)", 4, 10, 6)
with c2:
    auto_run = st.checkbox("RUN SIMULATION", value=False)

MAX_STEPS = 500
INTERVAL = 0.2

# --- Initialize session state ---
if 'mab' not in st.session_state or st.session_state.number_of_arms != number_of_mushrooms:
    st.session_state.mab = MAB(n_arms=number_of_mushrooms)
    st.session_state.number_of_arms = number_of_mushrooms
    st.session_state.steps = 0

if 'generator' not in st.session_state:
    st.session_state.generator = load_vae_model()

mab = st.session_state.mab
generator = st.session_state.generator

# --- Reset Button ---
if st.button("Reset Simulation"):
    st.session_state.steps = 0
    st.session_state.mab = MAB(n_arms=number_of_mushrooms)
    st.rerun()  # Force immediate re-run with cleared state

# --- Auto-run full simulation loop up to MAX_STEPS ---
placeholder = st.empty()

if auto_run and st.session_state.steps < MAX_STEPS:
    for _ in range(st.session_state.steps, MAX_STEPS):
        step_result = mab.step()
        st.session_state.steps += 1
        data = mab.get_data()

        with placeholder.container():
            st.subheader(f"Round {step_result['t']}")
            st.markdown("**Updated Distributions of Mushroom Success Probabilities**")
            st.pyplot(plotting.plot_posteriors(data["alpha"], data["beta_vals"], data["chosen_arms"]))

            col1, col2, col3 = st.columns([0.3, 0.1, 0.6])
            with col1:
                st.markdown("**Ideal Mushroom**")
                ideal_mushroom = generate_image_from_latent([0, 0], generator)
                st.pyplot(ideal_mushroom, use_container_width=False)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("**Selected Mushroom**")
                selected_mushroom = generate_image_from_latent(step_result['arm_vector'], generator)
                st.pyplot(selected_mushroom, use_container_width=False)

            with col3:
                st.markdown("**Selection Heatmap**")
                st.pyplot(plotting.plot_latent_selection(data["true_arms"], data["chosen_arms"]))

            st.markdown("**Learning Curve**")
            st.pyplot(plotting.plot_learning_curve(data["cumulative_rewards"]))

        time.sleep(INTERVAL)

# --- After MAX_STEP steps ---
if st.session_state.steps >= MAX_STEPS:
    st.success(f"Simulation finished after {MAX_STEPS} steps!")