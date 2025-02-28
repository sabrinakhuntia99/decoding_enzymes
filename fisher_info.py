'''
def compute_frequencies(X_train, y_train, window_size=9, alpha=1):
    """
    Compute frequencies of amino acid occurrences within the 9-residue window.
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    total_aa_count = len(X_train) * window_size

    # Initialize frequency dictionaries
    F_S_Rj = defaultdict(lambda: defaultdict(int))
    F_R = defaultdict(int)
    F_S = defaultdict(int)

    half_window = window_size // 2

    # Iterate through training data
    for seq, label in zip(X_train, y_train):
        cleavage_state = "cleaved" if label == 1 else "non-cleaved"
        F_S[cleavage_state] += 1

        for j in range(window_size):
            aa_index = np.argmax(seq[j + (9 - window_size) // 2])  # Find the one-hot encoded amino acid
            residue = aa_list[aa_index]

            F_S_Rj[cleavage_state][(residue, j)] += 1
            F_R[(residue, j)] += 1

    # Apply Laplace smoothing
    for (residue, j) in F_R.keys():
        F_R[(residue, j)] += alpha  # Add pseudocount
        total_aa_count += alpha * len(aa_list)

    return F_S_Rj, F_R, F_S, total_aa_count

def calculate_information_score(F_S_Rj, F_R, F_S, total_aa_count, window_size=9):
    """
    Compute log-likelihood information score I for each amino acid at each position.
    """
    I_scores = defaultdict(float)

    for state in F_S_Rj.keys():  # Iterate over cleavage/non-cleavage states
        FS = F_S[state]

        for (residue, j), FS_Rj in F_S_Rj[state].items():
            FR = F_R[(residue, j)]

            if FS_Rj > 0 and FR > 0 and FS > 0:
                I_component = (FS_Rj / FS) * np.log2((FS_Rj * total_aa_count) / (FR * FS))
                I_scores[(residue, j, state)] = I_component  # Store per-position score

    # Normalize scores by centering them
    mean_I = np.mean(list(I_scores.values()))
    I_scores = {key: val - mean_I for key, val in I_scores.items()}

    return I_scores

def plot_cleavage_vs_non_cleavage(I_scores, window_size=9):
    """
    Plot cleavage vs non-cleavage information scores for each amino acid across the 9-residue window.
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    position_labels = ["P4'", "P3'", "P2'", "P1'", "P1", "P2", "P3", "P4"]
    half_window = window_size // 2
    position_labels.insert(half_window, "∥")  # Cleavage site marker

    # Prepare data for plotting
    aa_scores_cleaved = {aa: [0] * window_size for aa in aa_list}  # Initialize with zero scores
    aa_scores_non_cleaved = {aa: [0] * window_size for aa in aa_list}  # Initialize with zero scores

    for (residue, pos, state), score in I_scores.items():
        if state == "cleaved":
            aa_scores_cleaved[residue][pos] = score  # Assign cleavage scores
        else:
            aa_scores_non_cleaved[residue][pos] = score  # Assign non-cleavage scores

    # Plot the data for each amino acid
    plt.figure(figsize=(12, 6))
    for aa in aa_list:
        plt.plot(position_labels, aa_scores_cleaved[aa], marker='o', label=f"{aa} (Cleavage)", linestyle='-')
        plt.plot(position_labels, aa_scores_non_cleaved[aa], marker='o', label=f"{aa} (Non-Cleavage)", linestyle='--')

    plt.axvline(x=half_window, color='black', linestyle='--', label="Cleavage Site")
    plt.xlabel("Position in 9-Residue Window")
    plt.ylabel("Information Score (I)")
    plt.title("Log-Likelihood Information Scores by Amino Acid Position (Cleavage vs Non-Cleavage)")
    plt.legend(title="Amino Acid", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_information_scores_separate(I_scores, window_size=9):
    """
    Plot 20 separate subplots for each amino acid's information score across the 9-residue window.
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    position_labels = ["P4'", "P3'", "P2'", "P1'", "P1", "P2", "P3", "P4"]
    half_window = window_size // 2
    position_labels.insert(half_window, "∥")  # Cleavage site marker

    # Prepare data for plotting
    aa_scores_cleaved = {aa: [0] * window_size for aa in aa_list}  # Initialize with zero scores
    aa_scores_non_cleaved = {aa: [0] * window_size for aa in aa_list}  # Initialize with zero scores

    for (residue, pos, state), score in I_scores.items():
        if state == "cleaved":
            aa_scores_cleaved[residue][pos] = score  # Assign cleavage scores
        else:
            aa_scores_non_cleaved[residue][pos] = score  # Assign non-cleavage scores

    # Set up figure with 20 subplots
    fig, axes = plt.subplots(5, 4, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle("Log-Likelihood Information Scores for Each Amino Acid", fontsize=16)

    for i, aa in enumerate(aa_list):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.plot(position_labels, aa_scores_cleaved[aa], marker='o', linestyle='-', color='b', label=f"{aa} (Cleavage)")
        ax.plot(position_labels, aa_scores_non_cleaved[aa], marker='o', linestyle='--', color='r', label=f"{aa} (Non-Cleavage)")
        ax.axvline(x=half_window, color='black', linestyle='--')  # Cleavage site marker
        ax.set_title(f"Amino Acid: {aa}")
        ax.set_xticklabels(position_labels, rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Compute frequencies
F_S_Rj, F_R, F_S, total_aa_count = compute_frequencies(X_train, y_train, window_size=9)

# Compute information scores
I_scores = calculate_information_score(F_S_Rj, F_R, F_S, total_aa_count)



# Generate the 20 separate subplots
plot_information_scores_separate(I_scores, window_size=9)

'''