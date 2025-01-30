import matplotlib.pyplot as plt
import numpy as np
import time
import faiss  # FAISS for efficient similarity search
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE  # ✅ Reintegration of SMOTE
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network  # Keep SigNet with Contrastive Loss
from esrgan import ESRGAN  # ESRGAN for super-resolution enhancement

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Initialize FAISS index
faiss_index = faiss.IndexFlatL2(128)  # Assuming 128-dimensional embeddings

# Load ESRGAN only in this script to prevent baseline improvement
esrgan = ESRGAN()

# ✅ Define Triplet Loss inside enhanced.py
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Triplet loss function for Siamese Networks.
    """
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return K.mean(loss)

# Dataset Configuration (Same as Baseline)
datasets = {
    "CEDAR": { "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
               "train_writers": list(range(261, 300)), "test_writers": list(range(300, 315)) },
    "BHSig260_Bengali": { "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
                          "train_writers": list(range(1, 71)), "test_writers": list(range(71, 100)) },
    "BHSig260_Hindi": { "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
                        "train_writers": list(range(101, 213)), "test_writers": list(range(213, 260)) }
}

# ✅ SMOTE Reintegration
def apply_smote(X1, X2, y, sampling_strategy=1.0):
    """
    Applies SMOTE for class balancing.

    Args:
        X1: First set of images (genuine signatures).
        X2: Second set of images (genuine or forged).
        y: Labels (1 for genuine, 0 for forged).
        sampling_strategy: Ratio of minority class samples to be generated.

    Returns:
        Balanced X1, X2, and labels.
    """
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined = np.hstack([X1_flat, X2_flat])

    combined_resampled, y_resampled = smote.fit_resample(combined, y)

    X1_resampled = combined_resampled[:, :X1_flat.shape[1]].reshape(-1, *X1.shape[1:])
    X2_resampled = combined_resampled[:, X1_flat.shape[1]:].reshape(-1, *X2.shape[1:])

    return X1_resampled, X2_resampled, y_resampled

# Apply ESRGAN-based preprocessing only for Enhanced Model
def preprocess_with_esrgan(img_array):
    return esrgan.enhance(img_array)

# Load dataset (Same as Baseline but with ESRGAN applied)
def load_data_with_esrgan_and_smote(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={ dataset_name: { "path": dataset_config["path"],
                                  "train_writers": dataset_config["train_writers"],
                                  "test_writers": dataset_config["test_writers"] }},
        img_height=155, img_width=220
    )
    
    (train_X1, train_X2), train_labels = generator.get_train_data()
    (test_X1, test_X2), test_labels = generator.get_test_data()

    # Apply ESRGAN preprocessing to all images
    train_X1 = np.array([preprocess_with_esrgan(img) for img in train_X1])
    train_X2 = np.array([preprocess_with_esrgan(img) for img in train_X2])

    # ✅ Apply SMOTE after preprocessing
    print("Applying SMOTE for class balancing...")
    train_X1, train_X2, train_labels = apply_smote(train_X1, train_X2, train_labels, sampling_strategy=1.0)

    return (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels)

# Measure Inference Time
def measure_inference_time(model, test_pairs):
    start_time = time.time()
    _ = model.predict(test_pairs, batch_size=32)
    end_time = time.time()

    avg_time = (end_time - start_time) / len(test_pairs[0])
    print(f"Average inference time per signature pair: {avg_time:.6f} seconds")
    return avg_time

# Measure FAISS Search Time
def measure_faiss_performance(faiss_index, query_embedding, k=5):
    start_time = time.time()
    nearest_neighbors, distances = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
    end_time = time.time()

    query_time = end_time - start_time
    print(f"FAISS Query Time: {query_time:.6f} seconds")
    return query_time

# Main Script - Load Enhanced Data
all_train_X1, all_train_X2, all_train_labels = [], [], []
all_test_X1, all_test_X2, all_test_labels = [], [], []

for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset with ESRGAN + SMOTE: {dataset_name} ---")
    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_esrgan_and_smote(dataset_name, dataset_config)

    all_train_X1.append(train_X1)
    all_train_X2.append(train_X2)
    all_train_labels.append(train_labels)

    all_test_X1.append(test_X1)
    all_test_X2.append(test_X2)
    all_test_labels.append(test_labels)

# Combine Datasets
train_X1 = np.concatenate(all_train_X1, axis=0)
train_X2 = np.concatenate(all_train_X2, axis=0)
train_labels = np.concatenate(all_train_labels, axis=0)
test_X1 = np.concatenate(all_test_X1, axis=0)
test_X2 = np.concatenate(all_test_X2, axis=0)
test_labels = np.concatenate(all_test_labels, axis=0)

# Train Model with FAISS + Triplet Loss
train_dataset = tf.data.Dataset.from_tensor_slices(((train_X1, train_X2), train_labels)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_X1, test_X2), test_labels)).batch(8)

model = create_siamese_network(input_shape=(155, 220, 1))

# ✅ Only Enhanced Model Uses Triplet Loss
model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

start_time = time.time()
model.fit(train_dataset, epochs=5, verbose=1)
end_time = time.time()

# Evaluate Model
print("\nEvaluating Enhanced Model...")
y_pred = model.predict(test_dataset)
measure_inference_time(model, (test_X1, test_X2))
query_embedding = np.random.rand(1, 128)  # Dummy embedding
measure_faiss_performance(faiss_index, query_embedding, k=5)
