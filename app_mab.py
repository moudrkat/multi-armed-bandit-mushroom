# app.py
import streamlit as st
import time
from mab import MAB
import plotting
from vae_decoder import load_vae_model, generate_image_from_latent

st.set_page_config(layout="wide")
st.title("The Multi-Shroomed Bandit")
st.markdown("**Thompson Sampling in Latent Space (VAE) Simulation**")
st.markdown("The Multi-Armed Bandit (MAB) is attempting to identify the mushroom that is most simmilar to the ideal mushroom (closest in the latent space), out of a few randomly sampled mushrooms.")

c1, c2 = st.columns(2)
with c1:
    number_of_mushrooms = st.slider("Number of arms (mushrooms)", 4, 8, 4)
# with c2:
#     # Control simulation
#     run_sim = st.button("Run Next Step")
with c2:
    auto_run = st.checkbox("Run Simulation", value=False)

# Initialize MAB instance only once
# Re-initialize MAB if number of arms changed
if 'mab' not in st.session_state or st.session_state.number_of_arms != number_of_mushrooms:
    st.session_state.mab = MAB(n_arms=number_of_mushrooms)
    st.session_state.number_of_arms = number_of_mushrooms
    st.session_state.steps = 0

mab = st.session_state.mab

# Load VAE model only once
if 'generator'not in st.session_state:
    st.session_state.generator = load_vae_model()

generator = st.session_state.generator
 
# interval = st.slider("Time Interval (sec)", 0.1, 2.0, 0.5)
interval = 0.005

if auto_run:
    placeholder = st.empty()

    for _ in range(500):
        step_result = mab.step()
        data = mab.get_data()

        with placeholder.container():
            st.subheader(f"Round {step_result['t']}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                
                st.markdown("**Ideal Mushroom**")
                ideal_mushroom = generate_image_from_latent([0,0], generator)
                st.pyplot(ideal_mushroom, use_container_width=False)

            with col2:
                st.markdown(f"**Closest To Ideal Mushroom As Of Round {step_result['t']}**")
                ideal_mushroom = generate_image_from_latent([0,0], generator)
                st.pyplot(ideal_mushroom, use_container_width=False)

            with col3: 
                st.markdown(f"**Latent Space Mushroom Selection Heatmap**")
                st.pyplot(plotting.plot_latent_selection(data["true_arms"], data["chosen_arms"]))

            with col4: 
                st.markdown(f"**Tested Mushroom In Round {step_result['t']}**")
                selected_mushroom = generate_image_from_latent(step_result['arm_vector'], generator)
                st.pyplot(selected_mushroom, use_container_width=False)


            st.markdown("**Posterior Distributions (Beta) of Mushroom Success Probabilities**")
            st.pyplot(plotting.plot_posteriors(data["alpha"], data["beta_vals"]))  

            st.markdown("**Learning Curve**")    
            st.pyplot(plotting.plot_learning_curve(data["cumulative_rewards"]))
                

        if not auto_run:
            break
        time.sleep(interval)