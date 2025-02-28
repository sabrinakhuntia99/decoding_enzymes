import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import gc
import matplotlib.pyplot as plt


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


def create_cnn_bilstm_model(seq_length=31):
    seq_input = tf.keras.layers.Input(shape=(seq_length, 20))
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(seq_input)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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


df = load_data("specified_trypsin_peptides_with_sequences_6.tsv")

data, labels = next(data_generator(df, batch_size=len(df)))
ros = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data.reshape(data.shape[0], -1), labels)
data_resampled = data_resampled.reshape(data_resampled.shape[0], 31, 20)

X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)

print(f"Total number of 31-mers for Transformer training: {len(X_train)}")

# Train the Transformer model on the full dataset
transformer_model = create_transformer_model()
transformer_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=32)

y_pred_prob = transformer_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Find false positives
false_positives = (y_pred.flatten() == 1) & (y_test.flatten() == 0)
false_positive_count = np.sum(false_positives)

# Get the false positive data and labels
X_false_positives = X_test[false_positives]
y_false_positives = y_test[false_positives]

# Find low-confidence predictions (i.e., predicted probability is between 0.45 and 0.55)
low_confidence = (y_pred_prob.flatten() > 0.45) & (y_pred_prob.flatten() < 0.55)
X_low_confidence = X_test[low_confidence]
y_low_confidence = y_test[low_confidence]

# Combine false positives and low-confidence data
X_expanded = np.concatenate([X_false_positives, X_low_confidence], axis=0)
y_expanded = np.concatenate([y_false_positives, y_low_confidence], axis=0)

# Print some statistics to make sure the dataset was expanded
print(f"Total number of false positives: {len(X_false_positives)}")
print(f"Total number of low-confidence predictions: {len(X_low_confidence)}")
print(f"Total number of samples for CNN-BiLSTM training: {len(X_expanded)}")

# Train the CNN-BiLSTM model on the false positives and low-confidence data
if len(X_expanded) > 0:
    cnn_bilstm_model = create_cnn_bilstm_model()
    cnn_bilstm_model.fit(X_expanded, y_expanded, epochs=2, batch_size=8)
else:
    print("Not enough data to train CNN-BiLSTM.")


# Function to calculate and plot saliency maps for CNN-BiLstm
def plot_saliency_maps(cnn_bilstm_model, X_sample, class_index):
    X_sample_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_sample_tensor)
        predictions = cnn_bilstm_model(X_sample_tensor)
        loss = predictions[0, class_index]
    grads = tape.gradient(loss, X_sample_tensor)
    saliency_map = grads.numpy().squeeze()

    plt.figure(figsize=(10, 6))
    plt.imshow(saliency_map, cmap='hot', aspect='auto', interpolation='nearest')
    plt.title(f"Saliency Map for Class: {class_index}")
    plt.colorbar()
    plt.show()

# Plot saliency maps
sample_idx = 5  # Example index
sample = X_test[sample_idx]
for i in range(2):  # Classes 0 and 1
    plot_saliency_maps(cnn_bilstm_model, sample[np.newaxis, :], i)

