import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Clear memory
def clear_memory():
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
    df = df.drop_duplicates()
    return df


# Get 31-mer context around a residue
def get_31mer(full_seq, residue_index, seq_length=31):
    start = max(0, residue_index - 15)
    end = min(len(full_seq), residue_index + 16)
    context_31mer = full_seq[start:end]
    return context_31mer.ljust(seq_length, 'X')


# Assign labels for binary classification (Cleavage vs Non-Cleavage)
def assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i, full_seq):
    # If i is peptide_start - 1 or peptide_end - 1, it's a cleavage site
    if i == peptide_start - 1 or i == peptide_end - 1:
        return 1  # Cleavage site

    # Otherwise, it's a non-cleavage site
    else:
        return 0  # Non-cleavage site


def prepare_training_data(df, seq_length=31):
    data = []
    labels = []

    # Initialize counters
    count_cleavage = 0
    count_noncleavage = 0

    # Initialize counters for amino acid cleavage occurrences
    amino_acid_cleavage_counts = np.zeros(20)  # Track cleavage counts for each amino acid
    amino_acid_counts = np.zeros(20)  # Track total occurrences of each amino acid

    with tqdm(total=len(df), desc="Preparing training data") as pbar:
        for _, row in df.iterrows():
            full_seq = row['Full Sequence']
            full_seq = full_seq.lstrip('M')  # Remove leading 'M' (Methionine) if present
            peptide = row['Peptide']
            peptide_start = full_seq.find(peptide)
            peptide_end = peptide_start + len(peptide)

            cleave_residue_N = full_seq[peptide_start - 1]  # Amino acid at peptide_start - 1
            cleave_residue_C = full_seq[peptide_end - 1]  # Amino acid at peptide_end - 1

            for i in range(peptide_start - 15, peptide_end + 16):
                if 0 <= i < len(full_seq):
                    context_31mer = get_31mer(full_seq, i, seq_length)
                    context_31mer_one_hot = np.array([one_hot_encode(aa) for aa in context_31mer])

                    label = assign_labels(peptide_start, peptide_end, cleave_residue_N, cleave_residue_C, i, full_seq)
                    data.append(context_31mer_one_hot)
                    labels.append(label)

                    # Update counters based on label
                    if label == 1:
                        count_cleavage += 1
                        amino_acid_cleavage_counts[aa_to_int.get(full_seq[i], -1)] += 1
                    elif label == 0:
                        count_noncleavage += 1

                    # Track occurrences of each amino acid
                    amino_acid_counts[aa_to_int.get(full_seq[i], -1)] += 1

            pbar.update(1)

    # Print label counts before returning data
    print(f"Number of Cleavage Sites: {count_cleavage}")
    print(f"Number of Non-Cleavage Sites: {count_noncleavage}")

    # Calculate and print proportions of cleavage sites for each amino acid
    print("\nProportion of Cleavage Sites for Each Amino Acid:")
    for i in range(20):
        amino_acid = list(aa_to_int.keys())[i]
        if amino_acid_counts[i] > 0:
            proportion = amino_acid_cleavage_counts[i] / amino_acid_counts[i]
            print(f"{amino_acid}: {proportion:.4f}")
        else:
            print(f"{amino_acid}: No occurrences")

    return np.array(data), np.array(labels)


# PyTorch Dataset
class ProteinDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, seq_length, d_model, num_classes, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(20, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model * seq_length, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(1, 0, 2)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(1, 0, 2).reshape(x.size(1), -1)
        return self.fc(x)


# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss, correct, total = 0, 0, 0

        for batch in dataloader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, "
              f"Accuracy: {correct / total * 100:.2f}%, Time: {time.time() - start_time:.2f}s")


# Evaluation Function with Confusion Matrix
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # Compute and print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Cleavage", "Cleavage"], yticklabels=["Non-Cleavage", "Cleavage"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

    return y_true, y_pred


# Saliency Map Explanation
def compute_gradients(inputs, model, target_class):
    inputs.requires_grad_()
    outputs = model(inputs)
    model.zero_grad()
    outputs[:, target_class].sum().backward()
    return inputs.grad


def explain_with_saliency_map(model, dataloader, num_classes=2):
    model.eval()

    # Get one sample from the dataloader
    x_batch, y_batch = next(iter(dataloader))
    x_batch = x_batch.to(device)

    # Calculate gradients and display saliency maps for all classes
    for target_class in range(num_classes):
        saliency_map = compute_gradients(x_batch, model, target_class)

        saliency_map = saliency_map[0].cpu().numpy().reshape(31, 20)  # Only the first sample
        plt.imshow(saliency_map, cmap='hot', aspect='auto')
        plt.title(f"Saliency Map for Class {target_class}")
        plt.colorbar()
        plt.show()

'''
# Enzyme Specificity Rule Generation with Averaging Across Multiple Samples
def generate_enzyme_specificity_rules(model, dataloader):
    model.eval()

    # Initialize arrays to store cumulative gradients for each amino acid type
    cumulative_gradients_cleavage = np.zeros(20)  # For cleavage (label = 1)
    cumulative_gradients_noncleavage = np.zeros(20)  # For non-cleavage (label = 0)

    count_cleavage = np.zeros(20)
    count_noncleavage = np.zeros(20)

    processed_batches = 0

    # Process batches from the dataloader
    for x_batch, y_batch in dataloader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        for target_class in range(2):  # For each class (cleavage, non-cleavage, missed cleavage)
            gradients = compute_gradients(x_batch, model, target_class)

            for i in range(20):  # For each amino acid type
                mask = (y_batch == target_class)  # Only consider gradients for the current class
                selected_gradients = gradients[mask, :, i].cpu().numpy()
                if selected_gradients.size > 0:
                    avg_grad = np.mean(selected_gradients)
                    if target_class == 0:
                        cumulative_gradients_noncleavage[i] += avg_grad
                        count_noncleavage[i] += 1
                    elif target_class == 1:
                        cumulative_gradients_cleavage[i] += avg_grad
                        count_cleavage[i] += 1

        processed_batches += 1

    # Normalize the cumulative gradients by the number of samples for each class
    gradient_map_cleavage = cumulative_gradients_cleavage / count_cleavage
    gradient_map_noncleavage = cumulative_gradients_noncleavage / count_noncleavage

    # Print the average gradients for each amino acid for each class
    print("\nAverage Gradients for Each Amino Acid (Cleavage Class):")
    for i in range(20):
        amino_acid = list(aa_to_int.keys())[i]
        print(f"{amino_acid}: {gradient_map_cleavage[i]:.4f}")

    print("\nAverage Gradients for Each Amino Acid (Non-Cleavage Class):")
    for i in range(20):
        amino_acid = list(aa_to_int.keys())[i]
        print(f"{amino_acid}: {gradient_map_noncleavage[i]:.4f}")


    # Now check for rules based on the quartiles of the average gradients
    def compute_quartiles(gradients):
        sorted_indices = np.argsort(gradients)
        q1_index = len(sorted_indices) // 4
        q3_index = 3 * len(sorted_indices) // 4

        # Bottom quartile: smallest gradients
        bottom_quartile = sorted_indices[:q1_index]

        # Top quartile: largest gradients
        top_quartile = sorted_indices[q3_index:]

        return top_quartile, bottom_quartile

    # Get quartiles for cleavage
    top_cleavage, bottom_cleavage = compute_quartiles(gradient_map_cleavage)
    print("Top Cleavage Indices and Gradients:", [(i, gradient_map_cleavage[i]) for i in top_cleavage])
    print("Bottom Cleavage Indices and Gradients:", [(i, gradient_map_cleavage[i]) for i in bottom_cleavage])

    # Get quartiles for non-cleavage
    top_noncleavage, bottom_noncleavage = compute_quartiles(gradient_map_noncleavage)
    print("Top Non-Cleavage Indices and Gradients:", [(i, gradient_map_noncleavage[i]) for i in top_noncleavage])
    print("Bottom Non-Cleavage Indices and Gradients:", [(i, gradient_map_noncleavage[i]) for i in bottom_noncleavage])


    # Print rules based on quartile checks
    for i in range(20):  # Iterate through amino acids
        amino_acid = list(aa_to_int.keys())[i]
        if i in top_cleavage:
            print(f"RULE: Enzyme cleaves at C-terminal of amino acid letter {amino_acid}")
        elif i in top_noncleavage:
            print(f"RULE: Enzyme does not cleave at C-terminal of amino acid letter {amino_acid}")
'''


# Main Execution
df = load_data("trypsin_peptides_with_sequences_6.tsv")

data, labels = prepare_training_data(df)
dataset = ProteinDataset(data, labels)

train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(seq_length=31, d_model=64, num_classes=2, nhead=4, num_layers=2, dim_feedforward=256).to(
    device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, num_epochs=1)
evaluate_model(model, test_loader)
explain_with_saliency_map(model, test_loader)
#generate_enzyme_specificity_rules(model, test_loader)
