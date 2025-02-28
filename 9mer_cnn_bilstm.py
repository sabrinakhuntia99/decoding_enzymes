import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, GlobalAveragePooling1D, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from tensorflow.keras.utils import to_categorical


# Clear TensorFlow session to free up memory
def clear_memory():
    print("Clearing session and memory...")
    tf.keras.backend.clear_session()
    gc.collect()


# Define amino acid to integer mapping globally
aa_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
             'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}


# Create one-hot encoding for each amino acid
def one_hot_encode(aa):
    encoding = np.zeros(20)
    encoding[aa_to_int.get(aa, -1)] = 1
    return encoding


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['Full Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df = df[df['Full Sequence'] != 'Sequence not found']
    df = df.drop_duplicates()  # Remove duplicate rows
    return df


# Get 9-mer context around a residue
def get_9mer(full_seq, residue_index, seq_length=9):
    start = max(0, residue_index - 4)
    end = min(len(full_seq), residue_index + 5)
    context_9mer = full_seq[start:end]
    return context_9mer.ljust(seq_length, 'X')


# Assign labels for multi-class classification
def assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i):
    if peptide_start - 1 < i < peptide_end - 1 and i == cleave_residue_N or i == cleave_residue_C:  # Missed cleavage
        return 2
    elif i == cleave_residue_N or i == cleave_residue_C:  # Cleavage
        return 1
    else:  # Non-cleavage
        return 0


# Prepare training data
def prepare_training_data(df, seq_length=9):
    data = []
    labels = []
    with tqdm(total=len(df), desc="Preparing training data") as pbar:
        for _, row in df.iterrows():
            full_seq = row['Full Sequence']
            full_seq = full_seq.lstrip('M')  # Remove all 'M' from the start of the sequence
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)
            cleave_residue_N = peptide_start - 1
            cleave_residue_C = peptide_end - 1

            for i in range(peptide_start - 4, peptide_end + 5):
                if 0 <= i < len(full_seq):
                    context_9mer = get_9mer(full_seq, i, seq_length)
                    context_9mer_one_hot = np.array([one_hot_encode(aa) for aa in context_9mer])
                    label = assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i)
                    data.append(context_9mer_one_hot)
                    labels.append(label)
            pbar.update(1)
    return np.array(data), np.array(labels)


# Create CNN-BiLSTM model
def create_model(seq_length=9):
    seq_input = Input(shape=(seq_length, 20), name="Sequence_Input")
    x = Conv1D(64, 3, activation='relu', padding='same')(seq_input)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Generate ICELOGO-like Bubble Chart (with percent difference as the y-axis)
def generate_icelogo_bubble_chart(X_test, y_test_labels, class_of_interest):
    """
    Generate a plot with amino acids as letters sized according to percentage differences.
    """
    residues = []
    for sample, label in zip(X_test, y_test_labels):
        if label == class_of_interest:  # Cleavage or missed cleavage
            for pos, aa_encoding in enumerate(sample):
                aa_index = np.argmax(aa_encoding)
                if aa_index < 20:  # Ignore padding
                    residues.append((pos, aa_index))

    # Compute position-specific frequencies for cleavage and non-cleavage classes
    cleavage_frequencies = np.zeros((9, 20))  # 9-mer by 20 amino acids
    non_cleavage_frequencies = np.zeros((9, 20))  # 9-mer by 20 amino acids

    for pos, aa_index in residues:
        if y_test_labels[pos] == 1:  # Cleavage class
            cleavage_frequencies[pos, aa_index] += 1
        else:  # Non-cleavage class
            non_cleavage_frequencies[pos, aa_index] += 1

    # Normalize to compute probabilities with a small epsilon to avoid division by zero
    epsilon = 1e-10  # Small constant to avoid division by zero
    cleavage_frequencies /= (cleavage_frequencies.sum(axis=1, keepdims=True) + epsilon)
    non_cleavage_frequencies /= (non_cleavage_frequencies.sum(axis=1, keepdims=True) + epsilon)

    # Calculate percentage difference for each position
    percent_diff = np.zeros((9, 20))
    for pos in range(9):
        for aa_index in range(20):
            # Calculate percentage difference between cleavage and non-cleavage classes
            cleavage_value = cleavage_frequencies[pos, aa_index]
            non_cleavage_value = non_cleavage_frequencies[pos, aa_index]
            percent_diff[pos, aa_index] = (cleavage_value - non_cleavage_value) * 100

    # Positions labels (P4, P3, P2, P1, P1', P2', P3', P4')
    position_labels = ['P4', 'P3', 'P2', 'P1', 'P1\'', 'P2\'', 'P3\'', 'P4\'', 'Site']

    # Plot bubble chart with letters
    plt.figure(figsize=(12, 6))  # Increase figure size to allow for more space

    # Adjustments for vertical positioning of bubbles (letters) to avoid crowding
    vertical_offset_range = 30  # Increase the range for vertical offset to avoid overlap

    # Loop through each position (P4 to P4') and plot the top 5 amino acids
    for pos in range(9):
        # Get the indices of the top 5 amino acids with the highest absolute percentage difference
        top_5_indices = np.argsort(np.abs(percent_diff[pos]))[-5:]

        for aa_index in top_5_indices:
            # Create bubble size based on percentage difference
            bubble_size = abs(percent_diff[pos, aa_index]) * 2  # Adjust bubble size
            if bubble_size > 0:
                aa_letter = list(aa_to_int.keys())[aa_index]

                # Apply vertical offset to each bubble (random offset for separation)
                vertical_offset = np.random.uniform(-vertical_offset_range, vertical_offset_range)

                # Plot the amino acid letter with an offset and scaled bubble size
                plt.text(pos, percent_diff[pos, aa_index] + vertical_offset, aa_letter, ha='center', va='center',
                         fontsize=bubble_size + 15,  # Increase font size for better visibility
                         color='black', fontweight='bold')

    # Add axis labels and grid
    plt.xticks(np.arange(9), position_labels)
    plt.yticks(np.arange(-100, 101, 20))  # Percent difference range from -100 to 100
    plt.xlabel("Positions in 9-mer")
    plt.ylabel("Percent Difference")
    plt.title(f"ICELOGO Bubble Chart for Class {class_of_interest}")
    plt.grid(True)
    plt.show()


# Main execution
clear_memory()

df = load_data("trypsin_peptides_with_sequences_6.tsv")
data, labels = prepare_training_data(df)

# Reshape and oversample
data_reshaped = data.reshape(data.shape[0], -1)
ros = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data_reshaped, labels)
data_resampled = data_resampled.reshape(data_resampled.shape[0], 9, 20)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels_resampled), y=labels_resampled)
class_weight_dict = dict(enumerate(class_weights))

# Create and train the model
model = create_model(seq_length=9)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, class_weight=class_weight_dict,
          callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Cleavage", "Cleavage", "Missed Cleavage"],
            yticklabels=["Non-Cleavage", "Cleavage", "Missed Cleavage"])
plt.title("Confusion Matrix")
plt.show()

# Generate the ICELOGO-like bubble chart for the "Cleavage" class
generate_icelogo_bubble_chart(X_test, y_test_classes, class_of_interest=1)
'''
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Step 1: Extract False Positive Sequences
def extract_false_positives(y_true, y_pred, sequences):
    # Ensure that the lengths match
    assert len(y_true) == len(y_pred) == len(sequences), "Mismatch between labels and sequences."

    false_positives = []

    # Loop through all predictions and check for false positives (predicted cleavage, actual non-cleavage)
    for idx in range(len(y_true)):
        if y_pred[idx] == 1 and y_true[idx] == 0:  # Predicted cleavage, but actual non-cleavage
            # Extract the sequence where the false positive occurs
            sequence = sequences[idx]

            # Find the index of the false positive site (cleavage site)
            cleavage_site = idx  # This is just an example; adjust based on your actual logic
            before_cleavage = sequence[max(0, cleavage_site - 4):cleavage_site]  # 4 residues before
            after_cleavage = sequence[cleavage_site + 1:cleavage_site + 5]  # 4 residues after

            # Store the false positive sequence context (9-mer)
            false_positives.append(before_cleavage + sequence[cleavage_site] + after_cleavage)

    return false_positives

# Step 2: Create a Dataset for CNN-BiLSTM Model on 9-mers
class CNNBiLSTM9MerDataset(Dataset):
    def __init__(self, false_positives, seq_length=9):
        self.data = []
        self.seq_length = seq_length
        self.num_aa = 20  # Number of amino acids in one-hot encoding

        # One-hot encode the 9-mers
        for sequence in false_positives:
            one_hot_sequence = np.array([one_hot_encode(aa) for aa in sequence])
            self.data.append(one_hot_sequence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# Step 3: Define CNN-BiLSTM Model for 9-mers (Unsupervised)
class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=2):
        super(CNNBiLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        # Apply Conv1D
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, seq_length)
        x = F.relu(self.conv1(x))

        # Apply LSTM
        x, (hn, cn) = self.lstm(x.permute(0, 2, 1))  # (batch_size, seq_length, 2 * hidden_dim)

        # Use the last LSTM output for classification (you can also use other strategies like averaging)
        x = x[:, -1, :]

        # Final fully connected layer
        x = self.fc(x)
        return x


# Step 4: K-Means Clustering to Identify Sequence Motifs
def apply_kmeans_clustering(features, num_clusters=3):
    # Standardize features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features_scaled)

    # Evaluate clustering with silhouette score
    silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # Return the cluster labels and cluster centers
    return kmeans.labels_, kmeans.cluster_centers_


# Step 5: Feature Extraction and Clustering on False Positives
def extract_features_and_cluster(false_positives, model, device):
    # Prepare dataset
    dataset = CNNBiLSTM9MerDataset(false_positives)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Extract features using CNN-BiLSTM model
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            # Extract the features (e.g., last hidden state or logits)
            all_features.append(outputs.cpu().numpy())

    # Flatten the features for clustering
    all_features = np.vstack(all_features)

    # Perform k-means clustering
    cluster_labels, cluster_centers = apply_kmeans_clustering(all_features, num_clusters=3)

    return cluster_labels, cluster_centers


# Step 6: Update Main Execution to Include False Positive Extraction and Clustering
# After training and evaluation:
y_true, y_pred = evaluate_model(model, test_loader)

# Extract the false positive sequences (predicted cleavage but actual non-cleavage)
false_positive_sequences = extract_false_positives(y_true, y_pred, df['Full Sequence'].values)

# Define a CNN-BiLSTM model for unsupervised sequence motif extraction
cnn_bilstm_model = CNNBiLSTMModel(input_dim=20, hidden_dim=32, num_layers=2).to(device)

# We can perform unsupervised feature extraction and k-means clustering
cluster_labels, cluster_centers = extract_features_and_cluster(false_positive_sequences, cnn_bilstm_model, device)

# Print clustering results (for now, we are just showing the cluster labels)
print("Cluster Labels for False Positives:", cluster_labels)
print("Cluster Centers:", cluster_centers)

'''
