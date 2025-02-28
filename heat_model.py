import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

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


# -----------------------------------------------------------------------------
# 2D Heat Diffusion Model
# -----------------------------------------------------------------------------
def solve_heat_diffusion_2d(u, dt, dx, dy, alpha, steps):
    """
    Solve the 2D heat diffusion equation using a finite difference method.

    u: initial state (2D numpy array)
    dt: time step
    dx, dy: spatial steps
    alpha: diffusion coefficient
    steps: number of time steps
    """
    for _ in range(steps):
        u_new = np.copy(u)
        # Compute discrete Laplacian (second-order spatial derivatives)
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * dt * (
                (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx ** 2) +
                (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy ** 2)
        )
        u = u_new
    return u


def heat_diffusion_model(X, steps=10):
    """
    Apply the 2D heat diffusion model to the input data.

    X: input data of shape (batch_size, seq_length, 20)
    steps: number of time steps for the diffusion process
    """
    batch_size, seq_length, _ = X.shape
    output = np.zeros_like(X)

    # Parameters for the diffusion process
    dt = 0.1
    dx = 1.0
    dy = 1.0
    alpha = 0.1  # Diffusion coefficient (adjust to control smoothing)

    for i in range(batch_size):
        u = X[i].T  # Transpose to shape (20, seq_length)
        u_next = solve_heat_diffusion_2d(u, dt, dx, dy, alpha, steps)
        output[i] = u_next.T  # Transpose back to (seq_length, 20)
    return output


# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------

# Load data
df = load_data("trypsin_peptides_with_sequences.tsv")
data, labels = next(data_generator(df, batch_size=len(df)))

# Print class distribution before resampling
print("Class distribution before resampling:", Counter(labels))


# Perform Random UnderSampling
rus = RandomUnderSampler(random_state=42)
data_resampled, labels_resampled = rus.fit_resample(data.reshape(data.shape[0], -1), labels)

# Print class distribution after resampling
print("Class distribution after resampling:", Counter(labels_resampled))

# Reshape back to original dimensions
data_resampled = data_resampled.reshape(data_resampled.shape[0], 9, 20)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.5, random_state=42)

# -----------------------------------------------------------------------------
# Apply Heat Diffusion Model to the Dataset
# -----------------------------------------------------------------------------
X_train_heat = heat_diffusion_model(X_train, steps=5)
X_test_heat = heat_diffusion_model(X_test, steps=5)

# -----------------------------------------------------------------------------
# Train a Simple Classifier on the Heat-Diffused Data
# -----------------------------------------------------------------------------
heat_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(9, 20)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
heat_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
heat_model.fit(X_train_heat, y_train, validation_data=(X_test_heat, y_test), epochs=2, batch_size=32)

# Evaluate the heat-based model on the test dataset
test_loss, test_acc = heat_model.evaluate(X_test_heat, y_test)
print(f"Test Accuracy on Heat Diffusion Model: {test_acc:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Predict on test set
y_pred_prob = heat_model.predict(X_test_heat)
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



# -----------------------------------------------------------------------------
# Post-Processing: Graphs for Heat Diffusion Energy
# -----------------------------------------------------------------------------

# (1) Heat Energy by Position for Each Amino Acid
def calculate_heat_energy_by_position(X_heat, seq_length=9):
    """
    Calculate the "heat energy" (mean squared values) at each sequence position for each amino acid.
    """
    heat_energy_per_position = np.zeros((seq_length, 20))  # (seq_length, 20 amino acids)
    batch_size, seq_length, _ = X_heat.shape

    for i in range(batch_size):
        heat_squared = np.square(X_heat[i])
        for pos in range(seq_length):
            heat_energy_per_position[pos] += heat_squared[pos]
    heat_energy_per_position /= batch_size
    return heat_energy_per_position


# Compute heat energy for the training set
heat_energy_train = calculate_heat_energy_by_position(X_train_heat)
# Calculate global min and max for y-axis across all amino acids
global_min = np.min(heat_energy_train)
global_max = np.max(heat_energy_train)

# Plot a 5x4 grid with consistent y-axis limits
fig, axes = plt.subplots(5, 4, figsize=(15, 15))
axes = axes.flatten()
aa_list = list(aa_to_int.keys())

for aa_idx in range(20):
    ax = axes[aa_idx]
    ax.plot(range(9), heat_energy_train[:, aa_idx], label=f'{aa_list[aa_idx]}')
    ax.set_title(f'{aa_list[aa_idx]}')
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Heat Energy')
    ax.set_ylim(global_min, global_max)  # Set consistent y-axis limits
    ax.grid(True)

# Set a title for the entire figure
fig.suptitle('Trypsin: Amino Acid Heat Energy Across Sequence Positions',
            fontsize=16, fontweight='bold')

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# (2) Heat Energy by Charge Group
# Define charge groups
aa_charge_map = {
    'A': 'neutral', 'C': 'neutral', 'F': 'neutral', 'G': 'neutral', 'I': 'neutral',
    'L': 'neutral', 'M': 'neutral', 'N': 'neutral', 'P': 'neutral', 'Q': 'neutral',
    'S': 'neutral', 'T': 'neutral', 'V': 'neutral', 'W': 'neutral', 'Y': 'neutral',
    'D': 'negative', 'E': 'negative', 'H': 'positive', 'K': 'positive', 'R': 'positive'
}


def calculate_heat_energy_by_charge_group(X_heat, seq_length=9):
    """
    Calculate the average heat energy for each charge group at each sequence position.
    """
    heat_energy_by_charge = {
        'neutral': np.zeros(seq_length),
        'positive': np.zeros(seq_length),
        'negative': np.zeros(seq_length)
    }
    batch_size, seq_length, _ = X_heat.shape

    for i in range(batch_size):
        heat_squared = np.square(X_heat[i])
        for pos in range(seq_length):
            for aa_idx in range(20):
                aa = aa_list[aa_idx]
                charge_group = aa_charge_map[aa]
                heat_energy_by_charge[charge_group][pos] += heat_squared[pos][aa_idx]

    # Normalize by the number of amino acids in each charge group and batch size
    charge_counts = {
        'neutral': sum(1 for aa in aa_charge_map.values() if aa == 'neutral'),
        'positive': sum(1 for aa in aa_charge_map.values() if aa == 'positive'),
        'negative': sum(1 for aa in aa_charge_map.values() if aa == 'negative')
    }
    for group in heat_energy_by_charge:
        heat_energy_by_charge[group] /= (batch_size * charge_counts[group])

    return heat_energy_by_charge


# Compute heat energy by charge group for training data
heat_energy_by_charge_train = calculate_heat_energy_by_charge_group(X_train_heat)

# Plot heat energy by charge group
plt.figure(figsize=(10, 6))
plt.plot(range(9), heat_energy_by_charge_train['neutral'], label='Neutral', color='green', linestyle='-', linewidth=2)
plt.plot(range(9), heat_energy_by_charge_train['positive'], label='Positive', color='red', linestyle='--', linewidth=2)
plt.plot(range(9), heat_energy_by_charge_train['negative'], label='Negative', color='blue', linestyle=':', linewidth=2)
plt.title('Trypsin: Average Heat Energy by Charge Group Across Sequence Positions', fontsize=14)
plt.xlabel('Position in Sequence', fontsize=12)
plt.ylabel('Average Heat Energy', fontsize=12)
plt.legend(title='Charge Group', fontsize=10, title_fontsize=12)
plt.xlim(0, 8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# -----------------------------------------------------------------------------
# Post-Processing: 2-mer Distributions (Site + Following Residue)
# -----------------------------------------------------------------------------


def plot_2mer_distributions(data, labels):
    # Extract 2-mers at positions 4-5 for both classes
    cleavage_pairs = []
    non_cleavage_pairs = []

    for seq, label in zip(data, labels):
        # Convert one-hot encoded sequence to amino acids
        seq_aa = [list(aa_to_int.keys())[np.argmax(pos)] for pos in seq]
        pair = seq_aa[4] + seq_aa[5]

        if label == 1:
            cleavage_pairs.append(pair)
        else:
            non_cleavage_pairs.append(pair)

    # Get counts for all pairs
    def get_counts(pairs_list):
        return Counter(pairs_list)

    cleavage_counts = get_counts(cleavage_pairs)
    non_cleavage_counts = get_counts(non_cleavage_pairs)

    # Function to get top 40 pairs and their counts
    def get_top_40(counts):
        top_40 = counts.most_common(40)
        pairs = [pair for pair, _ in top_40]
        values = [count for _, count in top_40]
        return pairs, values

    # Get top 40 for cleavage and non-cleavage
    cleavage_pairs_top, cleavage_values_top = get_top_40(cleavage_counts)
    non_cleavage_pairs_top, non_cleavage_values_top = get_top_40(non_cleavage_counts)

    # Find the maximum y-axis value to use for both plots
    max_y = max(max(cleavage_values_top), max(non_cleavage_values_top))

    # Plot cleavage site distribution
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(cleavage_pairs_top, cleavage_values_top, color='red', alpha=0.7)
    ax1.set_title('Cleavage Sites (Label = 1) - Top 40 2-mers')
    ax1.set_xlabel('2-mer at Positions 4-5')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.set_ylim(0, max_y)  # Set the same y-axis limit for both plots
    plt.suptitle('Cleavage Site Distribution at Positions 4-5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot non-cleavage site distribution
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(non_cleavage_pairs_top, non_cleavage_values_top, color='blue', alpha=0.7)
    ax2.set_title('Non-Cleavage Sites (Label = 0) - Top 40 2-mers')
    ax2.set_xlabel('2-mer at Positions 4-5')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.set_ylim(0, max_y)  # Set the same y-axis limit for both plots
    plt.suptitle('Non-Cleavage Site Distribution at Positions 4-5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_2mer_distributions(data, labels)
