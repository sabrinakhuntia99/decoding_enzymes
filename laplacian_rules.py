# =============================================================================
# Statistical Analysis
# =============================================================================
'''
# Compute the Laplacian for each training sample if not already computed
laplacian_train = calculate_laplacian(X_diffused_train)  # shape: (n_samples, 9, 20)

# Since the 9-mer already represents the ±4 residues around the center,
# we simply use the entire sequence (indices 0 to 8).
window_start = 0
window_end = 9

# Initialize a feature matrix.
# Features per sample: [avg_abs_laplacian, positive_residue_count, negative_residue_count, neutral_residue_count]
features = []
for i in range(X_diffused_train.shape[0]):
    # Extract Laplacian values over the entire 9-mer and compute average absolute value.
    lap_window = laplacian_train[i, window_start:window_end, :]
    avg_lap = np.mean(np.abs(lap_window))

    # Count charged residues over the entire 9-mer.
    count_pos = 0
    count_neg = 0
    count_neut = 0
    for pos in range(window_start, window_end):
        onehot = X_diffused_train[i, pos, :]  # Now using diffused features
        if np.sum(onehot) == 0:
            continue  # Skip padded positions, if any.
        aa_index = np.argmax(onehot)
        aa = list(aa_to_int.keys())[aa_index]
        if aa in aa_charge_map:
            if aa_charge_map[aa] == 'positive':
                count_pos += 1
            elif aa_charge_map[aa] == 'negative':
                count_neg += 1
            elif aa_charge_map[aa] == 'neutral':
                count_neut += 1
    features.append([avg_lap, count_pos, count_neg, count_neut])

features = np.array(features)

# Perform a t-test comparing the average absolute Laplacian (over the entire 9-mer)
# between cleavage (label = 1) and non-cleavage (label = 0) samples.
cleavage_lap = features[y_train == 1, 0]
non_cleavage_lap = features[y_train == 0, 0]

t_stat, p_val = ttest_ind(cleavage_lap, non_cleavage_lap, equal_var=False)
print("T-test for average absolute Laplacian (entire 9-mer):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# =============================================================================
# Decision Tree for Conditional Rule Extraction
# =============================================================================
# Train a decision tree classifier to predict cleavage based on the extracted features.
clf = DecisionTreeClassifier(max_depth=15, random_state=42)
clf.fit(features, y_train)

# Export and print the decision tree rules.
tree_rules = export_text(clf, feature_names=["avg_abs_laplacian", "positive_residue_count", "negative_residue_count", "neutral_residue_count"])
print("\nExtracted Decision Tree Rules:")
print(tree_rules)

def tree_to_human_sentences(decision_tree, feature_names):
    """
    Convert a decision tree into a list of human-readable rules.

    Parameters:
      decision_tree: a fitted DecisionTreeClassifier.
      feature_names: list of feature names corresponding to the tree's features.

    Returns:
      A list of strings, each representing one rule.
    """
    tree_ = decision_tree.tree_

    def recurse(node, conditions):
        # If the node is not a leaf, get the condition and traverse left and right
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            # Conditions for the left branch: feature <= threshold
            left_conditions = conditions + [f"'{feature}' <= {threshold:.2f}"]
            left_sentences = recurse(tree_.children_left[node], left_conditions)
            # Conditions for the right branch: feature > threshold
            right_conditions = conditions + [f"'{feature}' > {threshold:.2f}"]
            right_sentences = recurse(tree_.children_right[node], right_conditions)
            return left_sentences + right_sentences
        else:
            # At a leaf, determine the predicted class
            value = tree_.value[node][0]
            predicted_class = np.argmax(value)
            if conditions:
                return [f"If {' and '.join(conditions)}, then predict class {predicted_class}."]
            else:
                return [f"Predict class {predicted_class}."]

    return recurse(0, [])


# Use the function with your trained classifier and feature names.
feature_names = ["avg_abs_laplacian", "positive_residue_count", "negative_residue_count", "neutral_residue_count"]
human_sentences = tree_to_human_sentences(clf, feature_names)

print("Interpretable Enzyme Specificity Rules: Amino Acid Charge Group")
for sentence in human_sentences:
    print(sentence)

# -----------------------------------------------------------------------------
# Compute Laplacian for Training Data (using same method as HeatDiffusionLayer)
# -----------------------------------------------------------------------------
def calculate_laplacian(u, dx=1.0, dy=1.0):
    """
    Compute Laplacian using finite differences (aligned with HeatDiffusionLayer).
    """
    laplacian_x = (np.roll(u, shift=-1, axis=1) - 2*u + np.roll(u, shift=1, axis=1)) / (dx**2)
    laplacian_y = (np.roll(u, shift=-1, axis=2) - 2*u + np.roll(u, shift=1, axis=2)) / (dy**2)
    return laplacian_x + laplacian_y

# Compute Laplacian for the training data
laplacian_train = calculate_laplacian(X_diffused_train)  # Shape: (n_samples, 9, 20)

# -----------------------------------------------------------------------------
# Extract Features Based on Laplacian Values of Amino Acids
# -----------------------------------------------------------------------------
def extract_laplacian_features(X, laplacian, aa_to_int):
    """
    For each amino acid in a sequence, sum its Laplacian values across positions.
    """
    features = []
    for i in range(X.shape[0]):
        laplacian_sum = np.zeros(20)  # 20 amino acids
        for pos in range(X.shape[1]):
            onehot = X[i, pos, :]
            if np.sum(onehot) == 0:
                continue  # Skip padded positions
            aa_index = np.argmax(onehot)
            # Sum the Laplacian value for this amino acid at this position
            laplacian_sum[aa_index] += laplacian[i, pos, aa_index]
        features.append(laplacian_sum)
    return np.array(features)

# Extract Laplacian-based features for training data
laplacian_features_train = extract_laplacian_features(X_diffused_train, laplacian_train, aa_to_int)

# -----------------------------------------------------------------------------
# Train Decision Tree on Laplacian Features
# -----------------------------------------------------------------------------
clf_laplacian = DecisionTreeClassifier(max_depth=15, random_state=42)
clf_laplacian.fit(laplacian_features_train, y_train)

# Export and print decision tree rules
tree_rules_laplacian = export_text(clf_laplacian, feature_names=[aa for aa in aa_to_int.keys()])
print("\nExtracted Decision Tree Rules (Laplacian-Based):")
print(tree_rules_laplacian)

# -----------------------------------------------------------------------------
# Generate Human-Readable Rules
# -----------------------------------------------------------------------------
def tree_to_human_sentences_laplacian(decision_tree, feature_names):
    tree_ = decision_tree.tree_
    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            left_conditions = conditions + [f"Sum of Laplacian for '{feature}' ≤ {threshold:.2f}"]
            left_sentences = recurse(tree_.children_left[node], left_conditions)
            right_conditions = conditions + [f"Sum of Laplacian for '{feature}' > {threshold:.2f}"]
            right_sentences = recurse(tree_.children_right[node], right_conditions)
            return left_sentences + right_sentences
        else:
            value = tree_.value[node][0]
            predicted_class = np.argmax(value)
            if conditions:
                return [f"If {' and '.join(conditions)}, predict cleavage ({predicted_class})."]
            else:
                return [f"Baseline prediction: {predicted_class}"]
    return recurse(0, [])

human_sentences_laplacian = tree_to_human_sentences_laplacian(clf_laplacian, list(aa_to_int.keys()))
print("Interpretable Enzyme Specificity Rules: Individual Residues")
for sentence in human_sentences_laplacian:
    print(sentence)
'''

'''
# =============================================================================
# Enhanced Statistical Analysis with Flux and Laplacian
# =============================================================================
def tree_to_biological_rules(decision_tree, feature_names):
    """
    Converts decision tree rules into biological explanations using:
    - Laplacian magnitude → "Laplacian magnitude"
    - Charge group counts → positional residue requirements
    """
    tree_ = decision_tree.tree_
    feature_names = [
        f.replace("avg_abs_laplacian", "Laplacian magnitude")
        .replace("positive_residue_count", "positive residues")
        .replace("negative_residue_count", "negative residues")
        for f in feature_names
    ]

    def recurse(node, conditions, position=None):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Detect if we're in a positional context
            if "position" in feature.lower():
                pos = int(feature.split()[-1])
                new_position = pos if position is None else position
                condition_str = f"at position {new_position}"
            else:
                condition_str = "in the cleavage context"

            left_cond = f"{feature} ≤ {threshold:.2f} {condition_str}"
            right_cond = f"{feature} > {threshold:.2f} {condition_str}"

            left_rules = recurse(tree_.children_left[node], conditions + [left_cond], position)
            right_rules = recurse(tree_.children_right[node], conditions + [right_cond], position)

            return left_rules + right_rules
        else:
            prediction = np.argmax(tree_.value[node][0])
            confidence = np.max(tree_.value[node][0]) / np.sum(tree_.value[node][0])

            if prediction == 1:
                conclusion = "likely cleavage site"
            else:
                conclusion = "unlikely cleavage site"

            if confidence > 0.8:
                conclusion += " (high confidence)"

            if conditions:
                return [f"If {' and '.join(conditions)}, then {conclusion}."]
            return [f"Default case: {conclusion}."]

    return recurse(0, [])


# Compute both Laplacian and Flux for training data
laplacian_train = calculate_laplacian(X_diffused_train)
flux_train = calculate_flux(X_diffused_train, alpha=0.1)  # Use same alpha as in model

# Initialize feature matrix with both Laplacian and Flux metrics
# Features: [avg_abs_laplacian, avg_abs_flux, positive_count, negative_count, neutral_count]
features = []
for i in range(X_diffused_train.shape[0]):
    # Laplacian features
    lap_window = laplacian_train[i, 0:9, :]
    avg_lap = np.mean(np.abs(lap_window))

    # Flux features
    flux_window = flux_train[i, 0:9, :]
    avg_flux = np.mean(np.abs(flux_window))

    # Charge group counts
    count_pos, count_neg, count_neut = 0, 0, 0
    for pos in range(9):
        aa_idx = np.argmax(X_diffused_train[i, pos, :])
        aa = list(aa_to_int.keys())[aa_idx]
        group = aa_charge_map.get(aa, 'neutral')
        count_pos += 1 if group == 'positive' else 0
        count_neg += 1 if group == 'negative' else 0
        count_neut += 1 if group == 'neutral' else 0

    features.append([avg_lap, avg_flux, count_pos, count_neg, count_neut])

features = np.array(features)

# Perform t-tests for both Laplacian and Flux
for metric_idx, metric_name in enumerate(['Laplacian', 'Flux']):
    cleavage_vals = features[y_train == 1, metric_idx]
    non_cleavage_vals = features[y_train == 0, metric_idx]

    t_stat, p_val = ttest_ind(cleavage_vals, non_cleavage_vals, equal_var=False)
    print(f"\nT-test for {metric_name}:")
    print(f"  Cleavage mean: {np.mean(cleavage_vals):.4f}")
    print(f"  Non-cleavage mean: {np.mean(non_cleavage_vals):.4f}")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# =============================================================================
# Enhanced Decision Tree with Flux Features
# =============================================================================
# Train decision tree with both Laplacian and Flux features
clf_enhanced = DecisionTreeClassifier(max_depth=15, random_state=42)
clf_enhanced.fit(features, y_train)

# Export decision tree rules with new features
feature_names = [
    "avg_abs_laplacian",
    "avg_abs_flux",
    "positive_residue_count",
    "negative_residue_count",
    "neutral_residue_count"
]

tree_rules_enhanced = export_text(clf_enhanced, feature_names=feature_names)
print("\nEnhanced Decision Tree Rules (Laplacian + Flux):")
print(tree_rules_enhanced)

# =============================================================================
# Enhanced Statistical Analysis with Flux
# =============================================================================

# Compute Laplacian and Flux for training data
laplacian_train = calculate_laplacian(X_diffused_train)
flux_train = calculate_flux(X_diffused_train, alpha=0.1)

# Initialize feature matrix with both metrics
features = []
for i in range(X_diffused_train.shape[0]):
    # Laplacian features
    lap_window = laplacian_train[i, 0:9, :]
    avg_lap = np.mean(np.abs(lap_window))

    # Flux features
    flux_window = flux_train[i, 0:9, :]
    avg_flux = np.mean(np.abs(flux_window))

    # Charge group counts
    count_pos, count_neg, count_neut = 0, 0, 0
    for pos in range(9):
        aa_idx = np.argmax(X_diffused_train[i, pos, :])
        aa = list(aa_to_int.keys())[aa_idx]
        group = aa_charge_map.get(aa, 'neutral')
        count_pos += 1 if group == 'positive' else 0
        count_neg += 1 if group == 'negative' else 0
        count_neut += 1 if group == 'neutral' else 0

    features.append([avg_lap, avg_flux, count_pos, count_neg, count_neut])

features = np.array(features)

# Perform t-tests for both metrics
for metric_idx, metric_name in enumerate(['Laplacian', 'Flux']):
    cleavage_vals = features[y_train == 1, metric_idx]
    non_cleavage_vals = features[y_train == 0, metric_idx]

    t_stat, p_val = ttest_ind(cleavage_vals, non_cleavage_vals, equal_var=False)
    print(f"\nT-test for {metric_name}:")
    print(f"  Cleavage mean: {np.mean(cleavage_vals):.4f}")
    print(f"  Non-cleavage mean: {np.mean(non_cleavage_vals):.4f}")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# =============================================================================
# Enhanced Decision Tree with Flux
# =============================================================================
clf_enhanced = DecisionTreeClassifier(max_depth=15, random_state=42)
clf_enhanced.fit(features, y_train)

# Update feature names with flux
feature_names = [
    "avg_abs_laplacian",
    "avg_abs_flux",
    "positive_residue_count",
    "negative_residue_count",
    "neutral_residue_count"
]


# =============================================================================
# Updated Biological Rules Function with Flux Support
# =============================================================================
def tree_to_biological_rules(decision_tree, feature_names):
    """
    Enhanced to handle both Laplacian and Flux features with biological context:
    - Laplacian magnitude → Sequence pattern stability
    - Flux magnitude → Directional signal propagation
    - Charge groups → Electrostatic interactions
    """
    tree_ = decision_tree.tree_

    # Create biological interpretations for features
    bio_mapping = {
        "avg_abs_laplacian": "avg. abs. Laplacian",
        "avg_abs_flux": "avg. abs. flux",
        "positive_residue_count": "positive residue count",
        "negative_residue_count": "negative residue count",
        "neutral_residue_count": "neutral residue count"
    }

    formatted_names = [bio_mapping.get(f, f) for f in feature_names]

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_idx = tree_.feature[node]
            feature = formatted_names[feature_idx]
            threshold = tree_.threshold[node]

            # Add special handling for count features
            if "residue" in feature:
                # Convert fractional threshold to integer interpretation
                int_threshold = int(np.ceil(threshold))
                left_cond = f"{feature} ≤ {int_threshold-1}"
                right_cond = f"{feature} ≥ {int_threshold}"
            else:
                left_cond = f"{feature} ≤ {threshold:.2f}"
                right_cond = f"{feature} > {threshold:.2f}"



            left_rules = recurse(tree_.children_left[node], conditions + [left_cond])
            right_rules = recurse(tree_.children_right[node], conditions + [right_cond])

            return left_rules + right_rules
        else:
            prediction = np.argmax(tree_.value[node][0])
            confidence = np.max(tree_.value[node][0]) / np.sum(tree_.value[node][0])

            conclusion = "cleavage favored" if prediction == 1 else "cleavage inhibited"
            if confidence > 0.8:
                conclusion += " (high confidence)"

            if conditions:
                return [f"When {' and '.join(conditions)}, {conclusion}"]
            return [f"Baseline: {conclusion}"]

    return recurse(0, [])


# Generate and print enhanced biological rules with numbering
print("\nTrypsin Specificity Rules Based on Laplacian and Flux Analysis:")
bio_rules = tree_to_biological_rules(clf_enhanced, feature_names)
for i, rule in enumerate(bio_rules, 1):
    print(f"Rule {i}: {rule}")

'''