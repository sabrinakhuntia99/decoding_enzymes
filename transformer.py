import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LayerNormalization, MultiHeadAttention, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler  # Import the oversampling tool
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import tensorflow as tf
import shap  # Import SHAP

# Clear TensorFlow session to free up memory
def clear_memory():
    print("Clearing session and memory...")
    tf.keras.backend.clear_session()
    gc.collect()

# Clear memory before starting the main logic
clear_memory()

# Define amino acid to integer mapping globally
aa_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
             'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

# Create one-hot encoding for each amino acid
def one_hot_encode(aa):
    encoding = np.zeros(20)
    encoding[aa_to_int.get(aa, -1)] = 1  # Ensure that the amino acid exists in the dictionary
    return encoding

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['Full Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df = df[df['Full Sequence'] != 'Sequence not found']
    return df

# Get 31-mer context around a residue
def get_31mer(full_seq, residue_index, seq_length=31):
    start = max(0, residue_index - 15)
    end = min(len(full_seq), residue_index + 16)
    context_31mer = full_seq[start:end]
    return context_31mer.ljust(seq_length, 'X')  # Pad with 'X'

# Assign labels for the residues based on their location
def assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i):
    if peptide_start - 1 < i < peptide_end - 1 and (i == cleave_residue_N or i == cleave_residue_C):  # Missed cleavage
        return 2  # Missed cleavage
    elif i == cleave_residue_N or i == cleave_residue_C:  # Cleavage site
        return 1  # Cleavage site
    else:  # Non-cleavage site
        return 0  # Non-cleavage

# Prepare training data with one-hot encoding
def prepare_training_data(df, seq_length=31):
    data = []
    labels = []
    total_residues = 0

    with tqdm(total=len(df), desc="Preparing training data") as pbar:
        for _, row in df.iterrows():
            full_seq = row['Full Sequence']
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)

            # Cleavage residue indices (before and after the peptide)
            cleave_residue_N = peptide_start - 1 if peptide_start > 0 else -1
            cleave_residue_C = peptide_end if peptide_end < len(full_seq) else len(full_seq)

            # Iterate over the sequence and assign labels
            for i in range(peptide_start - 1, peptide_end + 1):  # Cleavage sites (before and after peptide)
                if i >= 0 and i < len(full_seq):  # Ensure within bounds
                    context_31mer = get_31mer(full_seq, i, seq_length)
                    context_31mer_one_hot = np.array([one_hot_encode(aa) for aa in context_31mer])

                    label = assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i)
                    data.append(context_31mer_one_hot)
                    labels.append(label)
                    total_residues += 1

            # Non-cleavage sites (30 residues around the cleavage site)
            for i in range(peptide_start - 15, peptide_end + 16):  # Context window before/after peptide
                if i >= 0 and i < len(full_seq):  # Ensure within bounds
                    context_31mer = get_31mer(full_seq, i, seq_length)
                    context_31mer_one_hot = np.array([one_hot_encode(aa) for aa in context_31mer])

                    if not (peptide_start - 1 <= i < peptide_end):
                        label = 0  # Non-cleavage site (within 31-mer)

                    data.append(context_31mer_one_hot)
                    labels.append(label)
                    total_residues += 1

            pbar.update(1)

    print(f"Total 31-mers generated: {total_residues}")
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

# Create Transformer model
def create_transformer_model(seq_length=31):
    seq_input = Input(shape=(seq_length, 20), name="Sequence_Input")

    # Apply a position encoding to the input sequence
    x = Embedding(input_dim=20, output_dim=64)(seq_input)

    # Add Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    attention_output = LayerNormalization()(attention_output)

    # Add a Feed-Forward Neural Network
    ff_output = Dense(128, activation='relu')(attention_output)
    ff_output = Dropout(0.5)(ff_output)

    # Flatten the output
    x = Flatten()(ff_output)

    # Dense layer for classification
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
df = load_data("chymotrypsin_peptides_with_sequences.tsv")

# Prepare training data
data, labels = prepare_training_data(df)

# Reshape the data
data_reshaped = data.reshape(data.shape[0], -1)

# Apply Random Oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data_reshaped, labels)

# Reshape the data back to 3D
data_resampled = data_resampled.reshape(data_resampled.shape[0], 31, 20)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)

# Create transformer model
model = create_transformer_model()

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, callbacks=[early_stopping], class_weight=class_weights)

# Evaluate the model on the test data
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_results[1] * 100:.2f}%")

# Predict the labels for the test set
y_pred = model.predict(X_test).round()

# Compute weighted F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Weighted F1 Score: {f1:.2f}")

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# SHAP explainer setup
explainer = shap.Explainer(model, X_train)

# Generate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# SHAP force plot for the first test instance
shap.initjs()  # Initialize JavaScript for rendering the plot
shap.force_plot(shap_values[0])  # Visualize the explanation for the first instance
