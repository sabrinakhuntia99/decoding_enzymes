import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import gc
import matplotlib.pyplot as plt
from collections import defaultdict

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

def data_generator(df, seq_length=9, batch_size=256):
    while True:
        data, labels = [], []
        for _, row in df.iterrows():
            full_seq = row['Full Sequence'].lstrip('M')
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)
            for i in range(peptide_start - 1, peptide_end + 1):
                if 0 <= i < len(full_seq):
                    context_9mer = full_seq[max(0, i - 4): min(len(full_seq), i + 5)].ljust(seq_length, 'X')
                    context_9mer_one_hot = np.array([one_hot_encode(aa) for aa in context_9mer], dtype=np.float32)
                    label = 1 if i == peptide_start - 1 or i == peptide_end - 1 else 0
                    data.append(context_9mer_one_hot)
                    labels.append(label)
                    if len(data) == batch_size:
                        yield np.array(data), np.array(labels, dtype=np.int8)
                        data, labels = [], []
        if data:
            yield np.array(data), np.array(labels, dtype=np.int8)

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

from collections import Counter

# Load data
df = load_data("chymotrypsin_peptides_with_sequences.tsv")
data, labels = next(data_generator(df, batch_size=len(df)))

# Print class distribution before resampling
print("Class distribution before resampling:", Counter(labels))
'''
# Perform Random OverSampling
ros = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data.reshape(data.shape[0], -1), labels)
'''
from imblearn.under_sampling import RandomUnderSampler

# Perform Random UnderSampling
rus = RandomUnderSampler(random_state=42)
data_resampled, labels_resampled = rus.fit_resample(data.reshape(data.shape[0], -1), labels)


# Print class distribution after resampling
print("Class distribution after resampling:", Counter(labels_resampled))

# Reshape back to original dimensions
data_resampled = data_resampled.reshape(data_resampled.shape[0], 9, 20)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.5, random_state=42)

# Apply Wave Equation-Based Model to the full dataset
X_train_wave = wave_equation_based_model(X_train, steps=5)
X_test_wave = wave_equation_based_model(X_test, steps=5)

# Train a simple classifier on the wave-processed data
wave_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(9, 20)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
wave_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
wave_model.fit(X_train_wave, y_train, validation_data=(X_test_wave, y_test), epochs=2, batch_size=32)

# Evaluate the wave-based model on the test dataset
test_loss, test_acc = wave_model.evaluate(X_test_wave, y_test)
print(f"Test Accuracy on Wave Model: {test_acc:.4f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Predict on test set
y_pred_prob = wave_model.predict(X_test_wave)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Cleavage (0)", "Cleavage (1)"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Compute additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Amino acids and charge groups
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
aa_charge_map = {
    'A': 'neutral', 'C': 'neutral', 'F': 'neutral', 'G': 'neutral', 'I': 'neutral',
    'L': 'neutral', 'M': 'neutral', 'N': 'neutral', 'P': 'neutral', 'Q': 'neutral',
    'S': 'neutral', 'T': 'neutral', 'V': 'neutral', 'W': 'neutral', 'Y': 'neutral',
    'D': 'negative', 'E': 'negative', 'H': 'positive', 'K': 'positive', 'R': 'positive'
}


# Define hydrophobicity scale for amino acids (example values)
hydrophobicity_map = {
    'A': 1.8,   # Alanine
    'R': -4.5,  # Arginine
    'N': -3.5,  # Asparagine
    'D': -3.5,  # Aspartic Acid
    'C': 2.5,   # Cysteine
    'Q': -3.5,  # Glutamine
    'E': -3.5,  # Glutamic Acid
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,   # Isoleucine
    'L': 3.8,   # Leucine
    'K': -3.9,  # Lysine
    'M': 1.9,   # Methionine
    'F': 2.8,   # Phenylalanine
    'P': -1.6,  # Proline
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
    'V': 4.2    # Valine
}

def extract_cleavage_rules(X_before, X_after, y_labels, seq_length=9, window_size=4):
    """
    Analyze how the wave equation affects cleavage by focusing on position 4 and surrounding residues.
    """
    cleavage_pos = 4  # Middle position where cleavage occurs

    # Compute absolute feature change due to wave propagation
    feature_change = np.abs(X_after - X_before).mean(axis=0)

    # Get indices of cleavage sites
    cleavage_indices = np.where(y_labels == 1)[0]

    # Track residue occurrences at cleavage and surrounding positions
    cleavage_residues = []
    charge_effect = {'positive': [], 'negative': [], 'neutral': []}
    neighbor_effect = {'positive': [], 'negative': [], 'neutral': []}
    window_data = {size: {'positive': [], 'negative': [], 'neutral': []} for size in range(1, window_size + 1)}  # Store charge data for each window size

    for idx in cleavage_indices:
        # Identify the amino acid at the cleavage site
        cleavage_aa = aa_list[np.argmax(X_before[idx][cleavage_pos])]
        cleavage_residues.append(cleavage_aa)

        # Analyze surrounding residues within the specified window
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue  # Skip the cleavage position itself
            neighbor_pos = cleavage_pos + offset
            neighbor_aa = aa_list[np.argmax(X_before[idx][neighbor_pos])]
            charge_group = aa_charge_map[neighbor_aa]
            neighbor_effect[charge_group].append(feature_change[neighbor_pos].mean())

            # Store charge group data within the window size
            for size in range(1, window_size + 1):
                if abs(offset) <= size:  # Only include residues within the current window size
                    window_data[size][charge_group].append(neighbor_aa)

    # Compute charge probabilities at the cleavage site
    charge_probabilities = {
        k: sum(v) / len(cleavage_residues) if len(v) > 0 else 0
        for k, v in neighbor_effect.items()
    }

    # Count the occurrences of each amino acid at cleavage sites
    cleavage_counts = pd.Series(cleavage_residues).value_counts()

    # Calculate the total number of cleavage events
    total_cleavages = len(cleavage_residues)

    # Calculate the probabilities (relative frequency) of cleavage for each amino acid
    cleavage_probabilities = cleavage_counts / total_cleavages

    # Define a dynamic threshold based on the maximum cleavage probability
    max_probability = cleavage_probabilities.max()

    # Find residues with a probability significantly greater than the others
    threshold = max_probability * 0.5

    # Filter residues that meet this threshold
    most_likely_residues = cleavage_probabilities[cleavage_probabilities >= threshold].index.tolist()

    # Print rule-based insights
    print("Cleavage Analysis:")
    print(f"- Cleavage most likely occurs after residues: {most_likely_residues}")

# Apply the function
extract_cleavage_rules(X_test, X_test_wave, y_test, seq_length=9, window_size=4)


# Calculate Wave Energy at each position in the sequence for each amino acid
def calculate_wave_energy_by_position(X_wave, seq_length=9):
    """
    Calculate the wave energy (mean squared value) for each position in the sequence for each amino acid.
    """
    wave_energy_per_position = np.zeros((seq_length, 20))  # (seq_length, 20 amino acids)

    batch_size, seq_length, _ = X_wave.shape

    # Calculate the squared wave values
    for i in range(batch_size):
        wave_squared = np.square(X_wave[i])  # Squared values of wave output

        # Sum the energy for each position and each amino acid
        for pos in range(seq_length):
            wave_energy_per_position[pos] += wave_squared[pos]

    # Normalize by batch size
    wave_energy_per_position /= batch_size
    return wave_energy_per_position


# Calculate the wave energy for training and test data
wave_energy_train = calculate_wave_energy_by_position(X_train_wave)
wave_energy_test = calculate_wave_energy_by_position(X_test_wave)

# Set up a 5x4 grid for subplots (20 subplots total)
fig, axes = plt.subplots(5, 4, figsize=(15, 15))

# Flatten axes array for easier indexing
axes = axes.flatten()

# Convert dict_keys to list to make it indexable
aa_list = list(aa_to_int.keys())

# Plot energy for each amino acid at each sequence position
for aa_idx in range(20):
    ax = axes[aa_idx]
    ax.plot(range(9), wave_energy_train[:, aa_idx], label=f'{aa_list[aa_idx]}')
    ax.set_title(f'{aa_list[aa_idx]}')
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Wave Energy')
    ax.grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Display the plot
plt.show()

# Function to calculate average wave energy by charge group
def calculate_wave_energy_by_charge_group(X_wave, seq_length=9):
    """
    Calculate the average wave energy for each charge group (neutral, positive, negative) at each position in the sequence.
    """
    wave_energy_by_charge = {
        'neutral': np.zeros(seq_length),
        'positive': np.zeros(seq_length),
        'negative': np.zeros(seq_length)
    }

    batch_size, seq_length, _ = X_wave.shape

    # Calculate the squared wave values
    for i in range(batch_size):
        wave_squared = np.square(X_wave[i])  # Squared values of wave output

        # Sum the energy for each position and each charge group
        for pos in range(seq_length):
            for aa_idx in range(20):
                aa = aa_list[aa_idx]
                charge_group = aa_charge_map[aa]
                wave_energy_by_charge[charge_group][pos] += wave_squared[pos][aa_idx]

    # Normalize by the number of amino acids in each charge group
    charge_counts = {
        'neutral': sum(1 for aa in aa_charge_map.values() if aa == 'neutral'),
        'positive': sum(1 for aa in aa_charge_map.values() if aa == 'positive'),
        'negative': sum(1 for aa in aa_charge_map.values() if aa == 'negative')
    }

    for charge_group in wave_energy_by_charge:
        wave_energy_by_charge[charge_group] /= (batch_size * charge_counts[charge_group])

    return wave_energy_by_charge


# Calculate the wave energy by charge group for training and test data
wave_energy_by_charge_train = calculate_wave_energy_by_charge_group(X_train_wave)
wave_energy_by_charge_test = calculate_wave_energy_by_charge_group(X_test_wave)

# Plot all three charge groups on one graph
plt.figure(figsize=(10, 6))

# Plot neutral charge group
plt.plot(range(9), wave_energy_by_charge_train['neutral'], label='Neutral', color='green', linestyle='-', linewidth=2)

# Plot positive charge group
plt.plot(range(9), wave_energy_by_charge_train['positive'], label='Positive', color='red', linestyle='--', linewidth=2)

# Plot negative charge group
plt.plot(range(9), wave_energy_by_charge_train['negative'], label='Negative', color='blue', linestyle=':', linewidth=2)

# Add labels, title, and legend
plt.title('Average Wave Energy by Charge Group Across Sequence Positions', fontsize=14)
plt.xlabel('Position in Sequence', fontsize=12)
plt.ylabel('Average Wave Energy', fontsize=12)
plt.legend(title='Charge Group', fontsize=10, title_fontsize=12)

# Set consistent domain and range for axes
plt.xlim(0, 8)  # Sequence positions from 0 to 30
plt.ylim(0, np.max([  # Set y-axis range to include all data
    np.max(wave_energy_by_charge_train['neutral']),
    np.max(wave_energy_by_charge_train['positive']),
    np.max(wave_energy_by_charge_train['negative'])
]) * 1.1)  # Add 10% padding to the top

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the derivatives of the wave energy (numerical derivative)
def calculate_derivative(data):
    return np.gradient(data)

# Calculate derivatives for each charge group
derivative_neutral = calculate_derivative(wave_energy_by_charge_train['neutral'])
derivative_positive = calculate_derivative(wave_energy_by_charge_train['positive'])
derivative_negative = calculate_derivative(wave_energy_by_charge_train['negative'])

# Plot the derivatives of the wave energy by charge group
plt.figure(figsize=(10, 6))
plt.plot(range(9), derivative_neutral, label='Neutral Derivative', color='green', linestyle='-', linewidth=2)
plt.plot(range(9), derivative_positive, label='Positive Derivative', color='red', linestyle='--', linewidth=2)
plt.plot(range(9), derivative_negative, label='Negative Derivative', color='blue', linestyle=':', linewidth=2)
plt.title('Derivative of Average Wave Energy by Charge Group', fontsize=14)
plt.xlabel('Position in Sequence', fontsize=12)
plt.ylabel('Derivative of Wave Energy', fontsize=12)
plt.legend(title='Charge Group', fontsize=10, title_fontsize=12)
plt.xlim(0, 8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

