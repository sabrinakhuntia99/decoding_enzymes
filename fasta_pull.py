import pandas as pd
from tqdm import tqdm

def parse_fasta(fasta_path):
    """Parses a FASTA file into a dictionary of {protein_id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                    current_seq = []
                # Extract protein ID from the header (assumes UniProt-style header)
                parts = line[1:].split('|')
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = None  # Skip entries with unexpected format
            else:
                if current_id is not None:
                    current_seq.append(line)
        # Add the last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    return sequences

# Load dataset
combined_df = pd.read_csv("peptide 6-9.tsv", sep="\t")

# Remove duplicates based on the "Peptide" column
combined_df = combined_df.drop_duplicates(subset=["Peptide"])

# Path to human proteome FASTA file
fasta_path = r"C:\Users\Sabrina\Documents\GitHub\enzyme_decoder\UniProt_Human.fasta"
sequence_dict = parse_fasta(fasta_path)

# Extract sequences for each protein ID
full_sequences = []
for protein_id in tqdm(combined_df["Protein ID"], desc="Extracting sequences"):
    full_sequence = sequence_dict.get(protein_id)
    full_sequences.append({"Protein ID": protein_id, "Full Sequence": full_sequence})

# Merge sequences with the original DataFrame
sequence_df = pd.DataFrame(full_sequences)
combined_df = pd.merge(combined_df, sequence_df, on="Protein ID", how="left")

# Keep required columns and save
combined_df = combined_df[['Peptide', 'Protein ID', 'Full Sequence', 'Spectral Count']]
combined_df.to_csv("specified_trypsin_peptides_with_sequences_6-9.tsv", sep="\t", index=False)

print("Dataset saved as 'specified_trypsin_peptides_with_sequences_6-9.tsv'")