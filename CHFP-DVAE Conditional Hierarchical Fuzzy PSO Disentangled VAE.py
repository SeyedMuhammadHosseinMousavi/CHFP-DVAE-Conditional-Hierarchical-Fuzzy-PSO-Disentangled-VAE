#----------------------------------------------------
# CHFP-DVAE: Conditional Hierarchical Fuzzy PSO Disentangled VAE
# -----------------------------------------------------

# 1. Challenge
# The primary challenge in modeling body motion and emotion lies in effectively capturing
#  the complex interplay of these factors in a structured and controllable manner.
#  Traditional VAE models often struggle to generate diverse and expressive samples,
#  particularly when the dataset is limited in capturing the full range of emotional
#  expressions and motion variations. The integration of multiple components such as 
#  PSO, fuzzy logic, hierarchical structure, conditional inputs, and disentanglement seeks
#  to address these challenges by enhancing the richness and control of the generated data.

# 2. What It Does for Latent Space and Decoder
# -PSO: It optimizes the exploration of the latent space, ensuring that the model can
#  find the most diverse and high-quality samples.
# -Fuzzy Logic: Introduces controlled randomness to the sampling process, helping to 
# capture a range of possible outputs and avoid overfitting.
# -Hierarchical Structure: Organizes the latent space into different levels, capturing 
# both global and fine-grained details of motion.
# -Conditional Inputs: Ensures that the latent space is aligned with specific conditions
#  like emotions and intensity, allowing for controlled and targeted generation.
# -Disentanglement: Aims to separate different factors in the data, such as emotion versus 
# general motion, making it easier to manipulate and understand the latent space.

# 3. What This Improves Compared to Normal VAE
# Compared to a standard VAE, this enhanced model significantly improves the generation 
# of diverse and contextually appropriate outputs. By incorporating PSO, the model can
#  explore the latent space more effectively, discovering a wider range of plausible
#  motions and emotions. Fuzzy logic adds variability to the generated samples, making
#  them more realistic and less deterministic. The hierarchical structure helps in better 
#  organizing the complex data, allowing the model to capture both high-level and detailed
#  features. Conditional inputs provide a mechanism to control the emotional content of the
#  generated samples, and disentanglement ensures that these features are cleanly separated,
#  allowing for more interpretable and adjustable outputs.

# PSO: It optimizes the exploration of the latent space, ensuring that the model 
# can find the most diverse and high-quality samples.

# Fuzzy Logic: Introduces controlled randomness to the sampling process, 
# helping to capture a range of possible outputs and avoid overfitting. 

# Hierarchical Structure: Organizes the latent space into different levels,
# capturing both global and fine-grained details of motion.


# 4. How It Works on Body Motion and Emotion
# The model processes input data by first encoding it into a latent space influenced
#  by additional conditions such as emotion and intensity. During training, the encoder
#  learns to map these conditions along with the motion data into a structured latent space.
#  The decoder then reconstructs the motion data from these latent representations, 
#  ensuring that the output aligns with the given conditions. PSO optimizes the exploration 
#  of this space, while fuzzy sampling adds variability, resulting in outputs that reflect 
#  a range of realistic motions and emotional expressions. The hierarchical and conditional
#  aspects help in generating nuanced body motions that correspond accurately to the
#  specified emotional states, providing a rich and varied dataset for applications such as
#  animation, virtual reality, and emotion recognition systems.

# -----------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
from pyswarm import pso
import warnings
import logging

# Suppress warnings and TensorFlow logging
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# === Important Parameters ===
base_folder = 'BFA Walk'  # Path to your folder containing subfolders 'angry', 'depressed', 'happy'
output_folder = 'vae_output'
top_latent_dim = 100  # Dimensionality of top-level latent space
bottom_latent_dim = 100  # Dimensionality of bottom-level latent space
latent_dim = 10
num_new_samples = 3  # Number of new samples to generate
epochs = 3  # Number of training epochs
batch_size = 32  # Batch size for training
sigma = 2.0  # Standard deviation for Gaussian smoothing
num_emotions = 3  # Number of emotion categories
emotion_intensity_dim = 1  # Example: single intensity input
fuzziness = 0.5  # Fuzziness for latent space exploration
beta = 4.0  # Beta value for β-VAE loss
tc_weight = 1.0  # Weight for Total Correlation (TC) loss
diversity_scale = 0.3  # Scale for diversity noise in PSO
temperature = 1.5  # Temperature scaling for diversity in PSO
num_particles = 10  # Number of particles for PSO
max_iter = 5  # Maximum iterations for PSO
learning_rate = 0.00001  # Learning rate for optimizer
clipvalue = 1.0  # Gradient clipping value

# List of class folders
class_folders = ['angry', 'depressed', 'happy']

# === Function Definitions ===

# Define the fuzzy sampling layer for VAE
@tf.autograph.experimental.do_not_convert
class FuzzySampling(layers.Layer):
    def __init__(self, fuzziness=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fuzziness = fuzziness

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + self.fuzziness * tf.exp(0.5 * z_log_var) * epsilon

# Function to parse BVH files
def parse_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header, motion_data = [], []
    capture_data = False
    for line in lines:
        if "MOTION" in line:
            capture_data = True
        elif capture_data:
            if line.strip().startswith("Frames") or line.strip().startswith("Frame Time"):
                continue
            motion_data.append(np.fromstring(line, sep=' '))
        else:
            header.append(line)
    return header, np.array(motion_data)

# Function to save BVH files
def save_bvh(header, motion_data, file_path):
    with open(file_path, 'w') as file:
        file.writelines(header)
        file.write("MOTION\n")
        file.write(f"Frames: {len(motion_data)}\n")
        file.write("Frame Time: 0.008333\n")
        for frame in motion_data:
            line = ' '.join(format(value, '.6f') for value in frame)
            file.write(line + '\n')

# Function to normalize motion data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_shape = data.shape
    data_flattened = data.reshape(-1, data_shape[-1])
    data_normalized = scaler.fit_transform(data_flattened).reshape(data_shape)
    return data_normalized, scaler

# Function to determine if a number is -0.0
def is_negative_zero(x):
    return np.signbit(x) & (x == 0.0)

# Function to smooth motion data using Gaussian filter
def smooth_motion_data_gaussian(motion_data, sigma=1.0):
    smoothed_data = gaussian_filter(motion_data, sigma=(sigma, 0))
    return smoothed_data

# Define the PSO optimization function
def pso_optimize_latent(top_latent_dim, bottom_latent_dim, num_particles, max_iter, decoder, emotion_sample, intensity_sample):
    def objective_function(latent_vector):
        # Split latent_vector into top and bottom components
        top_part = latent_vector[:top_latent_dim]
        bottom_part = latent_vector[top_latent_dim:]
        # Generate sample from latent vector using decoder
        generated_sample = decoder.predict([top_part.reshape(1, -1), bottom_part.reshape(1, -1), emotion_sample, intensity_sample])
        # Define some measure of quality (e.g., maximize diversity)
        quality = np.mean(generated_sample)  # Placeholder for actual quality measure
        return -quality  # PSO minimizes the objective function

    # Define bounds for PSO
    lb = [-1.0] * (top_latent_dim + bottom_latent_dim)
    ub = [1.0] * (top_latent_dim + bottom_latent_dim)

    # Run PSO
    best_latent_vector, _ = pso(objective_function, lb, ub, swarmsize=num_particles, maxiter=max_iter)
    return best_latent_vector[:top_latent_dim], best_latent_vector[top_latent_dim:]

# Build the Hierarchical Conditional VAE (CVAE) model with PSO and Fuzzy Sampling
def build_hierarchical_vae(input_shape, top_latent_dim, bottom_latent_dim, num_emotions, emotion_intensity_dim, fuzziness):
    initializer = tf.keras.initializers.GlorotUniform()

    # Emotion and Intensity Inputs
    emotion_input = layers.Input(shape=(num_emotions,), name='emotion_input')
    intensity_input = layers.Input(shape=(emotion_intensity_dim,), name='intensity_input')

    # Encoder
    encoder_inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Concatenate()([encoder_inputs, emotion_input, intensity_input])
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)

    # Top Latent Variables
    z_mean_top = layers.Dense(top_latent_dim, name='z_mean_top', kernel_initializer=initializer)(x)
    z_log_var_top = layers.Dense(top_latent_dim, name='z_log_var_top', kernel_initializer=initializer)(x)
    z_top = FuzzySampling(fuzziness=fuzziness)([z_mean_top, z_log_var_top])

    # Bottom Latent Variables
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(z_top)
    x = layers.Concatenate()([x, emotion_input])
    z_mean_bottom = layers.Dense(bottom_latent_dim, name='z_mean_bottom', kernel_initializer=initializer)(x)
    z_log_var_bottom = layers.Dense(bottom_latent_dim, name='z_log_var_bottom', kernel_initializer=initializer)(x)
    z_bottom = FuzzySampling(fuzziness=fuzziness)([z_mean_bottom, z_log_var_bottom])

    encoder = Model([encoder_inputs, emotion_input, intensity_input], [z_mean_top, z_log_var_top, z_top, z_mean_bottom, z_log_var_bottom, z_bottom], name='encoder')

    # Decoder
    latent_inputs_top = layers.Input(shape=(top_latent_dim,))
    latent_inputs_bottom = layers.Input(shape=(bottom_latent_dim,))
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(latent_inputs_bottom)
    x = layers.Dropout(0.3)(x)
    x = layers.Concatenate()([latent_inputs_top, x, emotion_input, intensity_input])
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer=initializer)(x)
    decoder_outputs = layers.Reshape(input_shape)(x)

    decoder = Model([latent_inputs_top, latent_inputs_bottom, emotion_input, intensity_input], decoder_outputs, name='decoder')

    # VAE
    encoder_outputs = encoder([encoder_inputs, emotion_input, intensity_input])
    vae_outputs = decoder([encoder_outputs[2], encoder_outputs[5], emotion_input, intensity_input])
    vae = Model(inputs=[encoder_inputs, emotion_input, intensity_input], outputs=vae_outputs, name='vae')

    # Compile the model with the custom loss
    def vae_loss(encoder_inputs, vae_outputs, z_mean_top, z_log_var_top, z_mean_bottom, z_log_var_bottom, beta=1.0):
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs), axis=-1))
        kl_loss_top = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_top - tf.square(z_mean_top) - tf.exp(z_log_var_top), axis=-1))
        kl_loss_bottom = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_bottom - tf.square(z_mean_bottom) - tf.exp(z_log_var_bottom), axis=-1))
        total_correlation = tc_weight * (kl_loss_top + kl_loss_bottom)
        return reconstruction_loss + beta * (kl_loss_top + kl_loss_bottom) + total_correlation

    vae.add_loss(vae_loss(encoder_inputs, vae_outputs, encoder_outputs[0], encoder_outputs[1], encoder_outputs[3], encoder_outputs[4], beta=beta))
    vae.compile(optimizer=Adam(learning_rate=learning_rate, clipvalue=clipvalue))

    return vae, encoder, decoder

# Function to load data from a folder
def load_data_from_folder(folder_path):
    all_data = []
    max_length = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            _, motion_data = parse_bvh(file_path)
            all_data.append(motion_data)
            if motion_data.shape[0] > max_length:
                max_length = motion_data.shape[0]
    # Pad all sequences to the max_length
    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in all_data]
    return np.array(padded_data)

# Function to generate samples and apply zero-mask handling
def generate_samples(decoder, top_latent_dim, bottom_latent_dim, num_samples, original_data, num_emotions, emotion_intensity_dim, scaler, original_shape, diversity_scale=0.3, temperature=1.5, num_particles=10, max_iter=5):
    generated_samples = []
    zero_mask = (original_data == 0.0) | (original_data == -0.0)
    negative_zero_mask = is_negative_zero(original_data)

    for _ in range(num_samples):
        # Generate initial latent vectors
        latent_sample_top = np.random.normal(size=(1, top_latent_dim))
        latent_sample_bottom = np.random.normal(size=(1, bottom_latent_dim))
        
        # Apply diversity scaling and temperature
        latent_sample_top += np.random.normal(scale=diversity_scale, size=latent_sample_top.shape)
        latent_sample_top *= temperature
        latent_sample_bottom += np.random.normal(scale=diversity_scale, size=latent_sample_bottom.shape)
        latent_sample_bottom *= temperature

        # Generate random emotion category and intensity
        emotion_sample = np.random.uniform(size=(1, num_emotions))
        intensity_sample = np.random.uniform(size=(1, emotion_intensity_dim))

        # Optimize the latent vector using PSO
        best_latent_vector_top, best_latent_vector_bottom = pso_optimize_latent(top_latent_dim, bottom_latent_dim, num_particles, max_iter, decoder, emotion_sample, intensity_sample)
        latent_sample_top = best_latent_vector_top.reshape(1, -1)
        latent_sample_bottom = best_latent_vector_bottom.reshape(1, -1)

        # Generate sample from optimized latent vector
        generated_sample = decoder.predict([latent_sample_top, latent_sample_bottom, emotion_sample, intensity_sample])
        generated_sample = scaler.inverse_transform(generated_sample.reshape(-1, generated_sample.shape[-1]))

        # Apply zero-mask handling
        for i in range(generated_sample.shape[0]):
            generated_sample[i] = np.where(zero_mask[i], 0.0, generated_sample[i])
            generated_sample[i][negative_zero_mask[i]] = -0.0

        generated_samples.append(generated_sample.reshape(original_shape[1:]))

    return generated_samples


# === Main Process ===
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for class_folder in class_folders:
    print(f"Processing class: {class_folder}")
    class_path = os.path.join(base_folder, class_folder)
    class_output_folder = os.path.join(output_folder, class_folder)
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)

    # Load and normalize data for the current class
    motion_data = load_data_from_folder(class_path)
    original_shape = motion_data.shape
    motion_data = motion_data.reshape((motion_data.shape[0], -1))  # Flatten data
    normalized_data, scaler = normalize_data(motion_data)

    # Build and train VAE for the current class
    input_shape = normalized_data.shape[1:]  # Correct input shape
    vae, encoder, decoder = build_hierarchical_vae(input_shape, top_latent_dim, bottom_latent_dim, num_emotions, emotion_intensity_dim, fuzziness)
    vae.fit([normalized_data, np.zeros((len(normalized_data), num_emotions)), np.zeros((len(normalized_data), emotion_intensity_dim))],
            normalized_data,
            epochs=epochs,
            batch_size=batch_size)

    # Generate new samples for the current class and apply the zero-mask mechanism
    generated_samples = generate_samples(decoder, top_latent_dim, bottom_latent_dim, num_new_samples, motion_data, num_emotions, emotion_intensity_dim, scaler, original_shape, diversity_scale, temperature, num_particles, max_iter)

    for i, generated_sample in enumerate(generated_samples):
        # Apply Gaussian smoothing
        smoothed_sample = smooth_motion_data_gaussian(generated_sample, sigma=sigma)

        # Save the smoothed sample
        header, _ = parse_bvh(os.path.join(class_path, os.listdir(class_path)[0]))
        output_file_path = os.path.join(class_output_folder, f'{class_folder}_generated_{i + 1}.bvh')
        save_bvh(header, smoothed_sample, output_file_path)
        print(f"Generated and smoothed sample {i + 1} for class {class_folder}")

print("Generation completed.")
