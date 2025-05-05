# vae_decoder.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the VAE model (assuming you have the model saved)
def load_vae_model(model_path="trained_decoder_VAE_mushroom_finalANNEAL.h5"):
    """Load the VAE model from a saved file."""
    vae_model = tf.keras.models.load_model(model_path)
    return vae_model  

# Function to generate an image from a given latent vector
def generate_image_from_latent(latent_vector, decoder):
    """Generate and display image from the latent vector."""
    latent_vector = np.expand_dims(latent_vector, axis=0)  # Add batch dimension
    generated_image = decoder(latent_vector)  # Generate image using the decoder
    generated_image = generated_image.numpy().squeeze()  # Remove extra dimensions

    # Plot the generated image
    fig, ax = plt.subplots(figsize=(1.1, 1.1))  # Create a figure and axes
    ax.imshow(generated_image, cmap='gray')  # Use appropriate color map for your images
    ax.axis('off')  # Turn off axes
    return fig  # Return the figure object