import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import gc
import matplotlib.pyplot as plt
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.manifold import TSNE

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

def create_cnn_bilstm_feature_extractor(seq_length=31):
    seq_input = tf.keras.layers.Input(shape=(seq_length, 20))
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(seq_input)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    feature_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # Normalize
    model = tf.keras.Model(inputs=seq_input, outputs=feature_output)
    return model

df = load_data("specified_trypsin_peptides_with_sequences_6.tsv")
data, labels = next(data_generator(df, batch_size=len(df)))
ros = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ros.fit_resample(data.reshape(data.shape[0], -1), labels)
data_resampled = data_resampled.reshape(data_resampled.shape[0], 31, 20)
X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)

transformer_model = create_transformer_model()
transformer_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=32)

y_pred_prob = transformer_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
false_positives = (y_pred.flatten() == 1) & (y_test.flatten() == 0)
X_false_positives = X_test[false_positives]
cnn_bilstm_feature_extractor = create_cnn_bilstm_feature_extractor()
X_fp_features = cnn_bilstm_feature_extractor.predict(X_false_positives)

# t-SNE Visualization
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_fp_features)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
plt.title("t-SNE Visualization of CNN-BiLSTM Extracted Features")
plt.show()

# LIME Explainer for Transformer with multiple sequences
explainer = LimeTabularExplainer(
    X_train.reshape(X_train.shape[0], -1),
    mode='classification',
    class_names=['Class 0', 'Class 1'],
    feature_names=[f'AA{i}_{j}' for i in range(1, 32) for j in range(1, 21)]
)

# Function to wrap the transformer prediction for multiple sequences
def wrapped_transformer_predict(flattened_input_batch):
    reshaped_input_batch = flattened_input_batch.reshape((-1, 31, 20))
    prob_class_1 = transformer_model.predict(reshaped_input_batch)
    prob_class_0 = 1 - prob_class_1
    return np.hstack((prob_class_0, prob_class_1))

# Select a batch of sequences for explanation
batch_size_for_lime = 100  # You can adjust this as needed
batch_idx = np.random.choice(len(X_test), batch_size_for_lime, replace=False)
batch_samples = X_test[batch_idx]

# Generate LIME explanations for all samples in the batch
explanations = []
for sample in batch_samples:
    explanation = explainer.explain_instance(sample.flatten(), wrapped_transformer_predict, num_features=10)
    explanations.append(explanation)

# Now average the explanations over the batch
# Get the feature importance scores from each explanation and average them
avg_feature_importance = np.zeros(len(explanations[0].as_list()))
for explanation in explanations:
    feature_importance = np.array([score for _, score in explanation.as_list()])
    avg_feature_importance += feature_importance

avg_feature_importance /= len(explanations)

# Plot the averaged feature importance
features = [f'AA{i}_{j}' for i in range(1, 32) for j in range(1, 21)]
top_features = np.argsort(avg_feature_importance)[-10:]  # Top 10 features

# Plot the top 10 features based on the average importance
plt.barh(np.array(features)[top_features], avg_feature_importance[top_features])
plt.xlabel('Average Feature Importance')
plt.title('Average Feature Importance from LIME for Multiple Sequences')
plt.show()

# Save the averaged explanation to an HTML file
with open('lime_transformer_average_explanation.html', 'w') as f:
    f.write('<html><body><h1>LIME Explanation for Transformer Model</h1>')
    f.write('<p>Average feature importance from multiple sequences:</p>')
    f.write('<ul>')
    for feature, importance in zip(features, avg_feature_importance):
        f.write(f'<li>{feature}: {importance}</li>')
    f.write('</ul>')
    f.write('</body></html>')
