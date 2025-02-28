import pandas as pd
from tqdm import tqdm

# List of TSV files to combine
file_paths = [
    "peptide 6.tsv",
    "peptide 7.tsv",
    "peptide 8.tsv",
    "peptide 9.tsv"
]

# Read and combine files
dfs = []
for file_path in tqdm(file_paths, desc="Reading TSV files"):
    try:
        df = pd.read_csv(file_path, sep='\t')
        dfs.append(df)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

# Combine all DataFrames
if dfs:
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    combined_df = combined_df.drop_duplicates()

    # Save combined file
    combined_df.to_csv("peptide 6-9.tsv", sep='\t', index=False)
    print(f"Successfully combined {len(dfs)} files. Saved to peptide 6-9.tsv")
else:
    print("No valid TSV files were found to combine.")