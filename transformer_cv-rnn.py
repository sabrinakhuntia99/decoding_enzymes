import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import gc

# Clear memory
def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

clear_memory()

# One-hot encoding for amino acids
aa_to_int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def one_hot_encode(aa):
    encoding = np.zeros(20)
    if aa in aa_to_int:
        encoding[aa_to_int[aa]] = 1
    return encoding

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['Full Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    return df

def data_generator(df, seq_length=31, batch_size=256):
    while True:
        data, labels = [], []
        for _, row in df.iterrows():
            full_seq = row['Full Sequence'].lstrip('M')
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)
            for i in range(peptide_start - 1, peptide_end + 1):
                if 0 <= i < len(full_seq):
                    context_31mer = full_seq[max(0, i - 15): min(len(full_seq), i + 16)].ljust(seq_length, 'X')
                    context_31mer_one_hot = np.array([one_hot_encode(aa) for aa in context_31mer], dtype=np.float32)
                    label = 1 if i == peptide_start - 1 or i == peptide_end - 1 else 0
                    data.append(context_31mer_one_hot)
                    labels.append(label)
                    if len(data) == batch_size:
                        yield np.array(data), np.array(labels, dtype=np.int8)
                        data, labels = [], []
        if data:
            yield np.array(data), np.array(labels, dtype=np.int8)

# Transformer Model
def create_transformer_model(seq_length=31):
    seq_input = tf.keras.layers.Input(shape=(seq_length, 20))
    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(seq_input, seq_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Complex-Valued Activation for Wave Propagation
class ComplexWaveActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))
        phase = tf.math.atan2(imag_part, real_part)
        new_real = magnitude * tf.cos(tf.sin(phase))
        new_imag = magnitude * tf.sin(tf.sin(phase))
        return tf.concat([new_real, new_imag], axis=-1)

# Complex-Valued RNN Cell with Traveling Waves
class ComplexWaveRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, wave_speed=0.5, **kwargs):
        super(ComplexWaveRNNCell, self).__init__(**kwargs)
        self.units = units
        self.wave_speed = wave_speed
        self.state_size = [units, units]

    def build(self, input_shape):
        self.W_real = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W_imag = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.U_real = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.U_imag = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.b_real = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        self.b_imag = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs, states):
        prev_state_real, prev_state_imag = states
        wave_term_real = self.wave_speed * (prev_state_real - tf.roll(prev_state_real, shift=1, axis=0))
        wave_term_imag = self.wave_speed * (prev_state_imag - tf.roll(prev_state_imag, shift=1, axis=0))
        real_part = tf.matmul(inputs, self.W_real) + tf.matmul(prev_state_real, self.U_real) + wave_term_real + self.b_real
        imag_part = tf.matmul(inputs, self.W_imag) + tf.matmul(prev_state_imag, self.U_imag) + wave_term_imag + self.b_imag
        next_state_real = tf.tanh(real_part)
        next_state_imag = tf.tanh(imag_part)
        return next_state_real, [next_state_real, next_state_imag]

class ComplexWaveRNN(tf.keras.layers.RNN):
    def __init__(self, units, wave_speed=0.5, return_sequences=False, return_state=False, **kwargs):
        cell = ComplexWaveRNNCell(units, wave_speed=wave_speed)
        super(ComplexWaveRNN, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,
                                             **kwargs)

def build_wave_cv_rnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = ComplexWaveRNN(64, wave_speed=0.5, return_sequences=True)(inputs)
    x = ComplexWaveActivation()(x)
    x = ComplexWaveRNN(32, wave_speed=0.3, return_sequences=False)(x)
    x = ComplexWaveActivation()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 2D Wave Equation-Based Model
def solve_wave_equation_2d(u, dt, dx, dy, c, steps):
    """
    Solve the 2D wave equation using finite difference method.
    u: initial state
    dt: time step
    dx, dy: spatial steps
    c: wave speed
    steps: number of time steps
    """
    nx, ny = u.shape
    u_next = np.zeros_like(u)
    u_prev = np.zeros_like(u)

    for _ in range(steps):
        u_next[1:-1, 1:-1] = (2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                              (c * dt / dx) ** 2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) +
                              (c * dt / dy) ** 2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]))
        u_prev, u = u, u_next

    return u


def wave_equation_based_model(X, steps=10):
    """
    Apply the 2D wave equation-based model to the input data.
    X: input data (batch_size, seq_length, 20)
    steps: number of time steps for the wave equation
    """
    batch_size, seq_length, _ = X.shape
    output = np.zeros_like(X)

    for i in range(batch_size):
        u = X[i].T  # Transpose to get (20, seq_length)
        dt = 0.1
        dx = 1.0
        dy = 1.0
        c = 1.0  # Wave speed
        u_next = solve_wave_equation_2d(u, dt, dx, dy, c, steps)
        output[i] = u_next.T  # Transpose back to (seq_length, 20)

    return output


# Load data
df = load_data("specified_trypsin_peptides_with_sequences_6.tsv")
data, labels = next(data_generator(df, batch_size=len(df)))
ros = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data.reshape(data.shape[0], -1), labels)
data_resampled = data_resampled.reshape(data_resampled.shape[0], 31, 20)
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)

# Train Transformer
transformer_model = create_transformer_model()
transformer_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=32)

# Identify low-confidence predictions
y_pred_prob = transformer_model.predict(X_test)
low_confidence_mask = (y_pred_prob > 0.4) & (y_pred_prob < 0.6)
X_low_confidence = X_test[low_confidence_mask.flatten()]
y_low_confidence = y_test[low_confidence_mask.flatten()]
'''
if len(X_low_confidence) > 0:
    cv_rnn_model = build_wave_cv_rnn_model((31, 20))
    cv_rnn_model.fit(X_low_confidence, y_low_confidence, validation_split=0.2, epochs=10, batch_size=8)


# Print class distribution of low-confidence samples
if len(y_low_confidence) > 0:
    unique, counts = np.unique(y_low_confidence, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class Distribution in Low-Confidence Samples: {class_distribution}")

    # Evaluate the cv-RNN model on low-confidence samples
    test_loss, test_acc = cv_rnn_model.evaluate(X_low_confidence, y_low_confidence)
    print(f"Test Accuracy on Low-Confidence Samples: {test_acc:.4f}")
else:
    print("No low-confidence samples to evaluate.")
'''
# Apply Wave Equation-Based Model to low-confidence samples
if len(X_low_confidence) > 0:
    X_low_confidence_wave = wave_equation_based_model(X_low_confidence, steps=5)

    # Train a simple classifier on the wave-processed data
    wave_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(31, 20)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    wave_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    wave_model.fit(X_low_confidence_wave, y_low_confidence, validation_split=0.2, epochs=10, batch_size=8)

    # Evaluate the wave-based model on low-confidence samples
    test_loss, test_acc = wave_model.evaluate(X_low_confidence_wave, y_low_confidence)
    print(f"Wave Equation-Based Model Test Accuracy on Low-Confidence Samples: {test_acc:.4f}")
else:
    print("No low-confidence samples to evaluate.")

import matplotlib.pyplot as plt
import numpy as np

# One-hot encoding to amino acid letter mapping
aa_list = 'ACDEFGHIKLMNPQRSTVWY'

import matplotlib.pyplot as plt

# Charge classification for amino acids
aa_charge_map = {
    'A': 'neutral', 'C': 'neutral', 'F': 'neutral', 'G': 'neutral', 'I': 'neutral', 'L': 'neutral', 'M': 'neutral',
    'N': 'neutral', 'P': 'neutral', 'Q': 'neutral', 'S': 'neutral', 'T': 'neutral', 'V': 'neutral', 'W': 'neutral',
    'Y': 'neutral',
    'D': 'negative', 'E': 'negative',
    'H': 'positive', 'K': 'positive', 'R': 'positive'
}


def charge_influence_by_distance(X_before, X_after, y_before, max_distance=4):
    """
    Analyze the influence of charge on cleavage at different distances (1 to max_distance) from the cleavage site.
    X_before: input data before wave propagation (seq_length, 20)
    X_after: input data after wave propagation (seq_length, 20)
    y_before: labels before wave propagation (cleavage: 1, non-cleavage: 0)
    max_distance: maximum distance from the cleavage site to consider (1 to max_distance residues)
    """
    # Calculate the average change per feature (as before)
    feature_diff = np.abs(X_after - X_before).mean(axis=0)

    # Separate the cleavage and non-cleavage samples
    cleavage_samples = X_before[y_before == 1]
    non_cleavage_samples = X_before[y_before == 0]

    # Initialize a dictionary to store feature differences at each distance
    charge_influence = {i: {'positive': [], 'negative': [], 'neutral': []} for i in range(1, max_distance + 1)}
    non_cleavage_influence = {i: {'positive': [], 'negative': [], 'neutral': []} for i in range(1, max_distance + 1)}

    # Iterate over each cleavage sample
    for idx, sample in enumerate(cleavage_samples):
        cleavage_index = np.argmax(y_before == 1)  # Find the cleavage index for the current sample

        # Get charge influence at each distance
        for dist in range(1, max_distance + 1):
            # Get the indices of the amino acids within `dist` residues of the cleavage site
            for i in range(max(0, cleavage_index - dist), min(len(sample), cleavage_index + dist + 1)):
                aa = aa_list[i]  # Amino acid at position i
                charge_group = aa_charge_map.get(aa, 'neutral')
                charge_influence[dist][charge_group].append(feature_diff[i])

    # Iterate over each non-cleavage sample
    for idx, sample in enumerate(non_cleavage_samples):
        cleavage_index = np.argmax(y_before == 0)  # Find the non-cleavage index for the current sample

        # Get charge influence at each distance
        for dist in range(1, max_distance + 1):
            # Get the indices of the amino acids within `dist` residues of the non-cleavage site
            for i in range(max(0, cleavage_index - dist), min(len(sample), cleavage_index + dist + 1)):
                aa = aa_list[i]  # Amino acid at position i
                charge_group = aa_charge_map.get(aa, 'neutral')
                non_cleavage_influence[dist][charge_group].append(feature_diff[i])

    # Calculate the average charge influence for each distance and charge group for cleavage and non-cleavage samples
    avg_positive_influence = [np.mean(charge_influence[dist]['positive']) for dist in range(1, max_distance + 1)]
    avg_negative_influence = [np.mean(charge_influence[dist]['negative']) for dist in range(1, max_distance + 1)]
    avg_neutral_influence = [np.mean(charge_influence[dist]['neutral']) for dist in range(1, max_distance + 1)]

    avg_positive_non_cleavage = [np.mean(non_cleavage_influence[dist]['positive']) for dist in
                                 range(1, max_distance + 1)]
    avg_negative_non_cleavage = [np.mean(non_cleavage_influence[dist]['negative']) for dist in
                                 range(1, max_distance + 1)]
    avg_neutral_non_cleavage = [np.mean(non_cleavage_influence[dist]['neutral']) for dist in range(1, max_distance + 1)]

    # Plotting the charge influence over different distances (1 to max_distance) for both cleavage and non-cleavage samples
    distances = np.arange(1, max_distance + 1)

    plt.figure(figsize=(8, 6))

    # Plot for cleavage samples
    plt.plot(distances, avg_positive_influence, label="Positive Charge (Cleavage)", marker='o', linestyle='-',
             color='b')
    plt.plot(distances, avg_negative_influence, label="Negative Charge (Cleavage)", marker='o', linestyle='-',
             color='r')
    plt.plot(distances, avg_neutral_influence, label="Neutral Charge (Cleavage)", marker='o', linestyle='-', color='g')

    # Plot for non-cleavage samples
    plt.plot(distances, avg_positive_non_cleavage, label="Positive Charge (Non-Cleavage)", marker='x', linestyle='--',
             color='b')
    plt.plot(distances, avg_negative_non_cleavage, label="Negative Charge (Non-Cleavage)", marker='x', linestyle='--',
             color='r')
    plt.plot(distances, avg_neutral_non_cleavage, label="Neutral Charge (Non-Cleavage)", marker='x', linestyle='--',
             color='g')

    plt.xlabel("Distance from Cleavage Site (Residues)")
    plt.ylabel("Average Charge Influence on Cleavage")
    plt.title("Charge Influence on Cleavage and Non-Cleavage at Different Distances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Apply to low-confidence samples (cleavage vs non-cleavage)
if len(X_low_confidence) > 0:
    # Extract low-confidence cleavage vs non-cleavage samples
    y_low_confidence = y_low_confidence.flatten()

    # Perform wave propagation on the low-confidence samples
    X_low_confidence_wave = wave_equation_based_model(X_low_confidence, steps=5)

    # Analyze charge influence on cleavage and non-cleavage at different distances from the cleavage site
    charge_influence_by_distance(X_low_confidence, X_low_confidence_wave, y_low_confidence, max_distance=4)
else:
    print("No low-confidence samples to evaluate.")

