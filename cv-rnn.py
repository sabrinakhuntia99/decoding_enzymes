import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.under_sampling import RandomUnderSampler

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
    df = df[df['Full Sequence'] != 'Sequence not found']
    df = df.drop_duplicates()
    return df

def get_31mer(full_seq, residue_index, seq_length=31):
    start = max(0, residue_index - 15)
    end = min(len(full_seq), residue_index + 16)
    context_31mer = full_seq[start:end].ljust(seq_length, 'X')
    return context_31mer

def assign_labels(peptide_start, peptide_end, i):
    return 1 if i == peptide_start - 1 or i == peptide_end - 1 else 0

def prepare_training_data(df, seq_length=31):
    data, labels = [], []
    with tqdm(total=len(df), desc="Processing Data") as pbar:
        for _, row in df.iterrows():
            full_seq = row['Full Sequence'].lstrip('M')
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)
            for i in range(peptide_start - 15, peptide_end + 16):
                if 0 <= i < len(full_seq):
                    context_31mer = get_31mer(full_seq, i, seq_length)
                    context_31mer_one_hot = np.array([one_hot_encode(aa) for aa in context_31mer])
                    label = assign_labels(peptide_start, peptide_end, i)
                    data.append(context_31mer_one_hot)
                    labels.append(label)
            pbar.update(1)
    return np.array(data), np.array(labels, dtype=np.int8)

# Load dataset
file_path = "specified_trypsin_peptides_with_sequences_6.tsv"
df = load_data(file_path)
X, y = prepare_training_data(df)

# Shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply undersampling
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train.reshape(len(X_train), -1), y_train)
X_train_resampled = X_train_resampled.reshape(-1, 31, 20)

# Verify class distribution after undersampling
unique_classes, class_counts = np.unique(y_train_resampled, return_counts=True)
print("Class distribution after undersampling:")
for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls}: {count} samples")

# Complex-Valued Activation for Wave Propagation
class ComplexWaveActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        """Applies a wave-like activation function to real-valued inputs."""
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)

        magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))  # Compute magnitude
        phase = tf.math.atan2(imag_part, real_part)  # Compute phase angle

        new_real = magnitude * tf.cos(tf.sin(phase))  # Apply wave-like transformation
        new_imag = magnitude * tf.sin(tf.sin(phase))

        return tf.concat([new_real, new_imag], axis=-1)  # Concatenate to keep it real-valued


# Complex-Valued RNN Cell with Traveling Waves
class ComplexWaveRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, wave_speed=0.2, **kwargs):
        super(ComplexWaveRNNCell, self).__init__(**kwargs)
        self.units = units
        self.wave_speed = wave_speed
        self.state_size = [units, units]  # Define state size as two separate components (real & imaginary)

    def build(self, input_shape):
        """Initialize complex-valued weight matrices."""
        self.W_real = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W_imag = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.U_real = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.U_imag = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.b_real = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        self.b_imag = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs, states):
        """Compute the next state with wave propagation."""
        prev_state_real, prev_state_imag = states  # Extract real and imaginary states

        # Wave propagation term: ∂²u/∂t² = c²∇²u (approximated)
        wave_term_real = self.wave_speed * (prev_state_real - tf.roll(prev_state_real, shift=1, axis=0))
        wave_term_imag = self.wave_speed * (prev_state_imag - tf.roll(prev_state_imag, shift=1, axis=0))

        # Standard RNN recurrence
        real_part = tf.matmul(inputs, self.W_real) + tf.matmul(prev_state_real, self.U_real) + wave_term_real + self.b_real
        imag_part = tf.matmul(inputs, self.W_imag) + tf.matmul(prev_state_imag, self.U_imag) + wave_term_imag + self.b_imag

        next_state_real = tf.tanh(real_part)
        next_state_imag = tf.tanh(imag_part)

        return next_state_real, [next_state_real, next_state_imag]


# Complex-Valued RNN Layer with Traveling Waves
class ComplexWaveRNN(tf.keras.layers.RNN):
    def __init__(self, units, wave_speed=0.2, return_sequences=False, return_state=False, **kwargs):
        cell = ComplexWaveRNNCell(units, wave_speed=wave_speed)
        super(ComplexWaveRNN, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,
                                             **kwargs)


# Build the cv-RNN Model for Image Segmentation
def build_wave_cv_rnn_model(input_shape):
    """Builds a complex-valued RNN model with traveling waves."""
    inputs = tf.keras.Input(shape=input_shape)

    x = ComplexWaveRNN(64, wave_speed=0.2, return_sequences=True)(inputs)  # First cv-RNN with traveling waves
    x = ComplexWaveActivation()(x)

    x = ComplexWaveRNN(32, wave_speed=0.3, return_sequences=False)(x)  # Second cv-RNN
    x = ComplexWaveActivation()(x)

    class RealPartLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.math.real(inputs)  # Extract real part in a Keras-friendly way

    x = RealPartLayer()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # ✅ Now valid in Keras

    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train the model
input_shape = (31, 20)
model = build_wave_cv_rnn_model(input_shape)
model.fit(X_train_resampled, y_train_resampled, validation_data=(X_test, y_test), batch_size=64, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")