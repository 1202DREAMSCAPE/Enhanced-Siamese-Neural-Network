import os
import sys
import psutil
import time
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy, LossScaleOptimizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SigNet_v1 import create_siamese_network

# Enable mixed precision
set_global_policy('mixed_float16')
print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy()}")

# Define datasets
datasets = {
    "CEDAR": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 316))
    },
    "BHSig260_Bengali": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 100))
    },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 213)),
        "test_writers": list(range(213, 260))
    }
}

# Function to load datasets with paired inputs
def create_pairs(data, labels):
    pairs = []
    pair_labels = []
    class_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    for idx1 in range(len(data)):
        x1 = data[idx1]
        label1 = labels[idx1]

        # Positive pair (same label)
        idx2 = np.random.choice(class_indices[label1])
        x2 = data[idx2]
        pairs.append([x1, x2])
        pair_labels.append(1)

        # Negative pair (different label)
        label2 = np.random.choice([l for l in class_indices.keys() if l != label1])
        idx2 = np.random.choice(class_indices[label2])
        x2 = data[idx2]
        pairs.append([x1, x2])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)

# Load and preprocess data
def load_and_preprocess_data(datasets, img_height, img_width):
    data, labels = [], []

    for dataset_name, config in datasets.items():
        print(f"Loading data for {dataset_name}...")
        dataset_path = config["path"]
        train_writers = config["train_writers"]
        for writer in train_writers:
            writer_folder = os.path.join(dataset_path, f"writer_{writer:03d}")
            if not os.path.isdir(writer_folder):
                print(f"Warning: Writer folder {writer_folder} not found. Skipping...")
                continue

            genuine_folder = os.path.join(writer_folder, "genuine")
            forged_folder = os.path.join(writer_folder, "forged")

            if os.path.isdir(genuine_folder) and os.path.isdir(forged_folder):
                for label, folder in [(1, genuine_folder), (0, forged_folder)]:
                    for img_name in os.listdir(folder):
                        if img_name.endswith((".png", ".jpg")):
                            img_path = os.path.join(folder, img_name)
                            try:
                                img = load_img(img_path, target_size=(img_height, img_width), color_mode="grayscale")
                                data.append(img_to_array(img) / 255.0)
                                labels.append(label)
                            except Exception as e:
                                print(f"Error loading image {img_path}: {e}")
            else:
                print(f"Skipping writer folder {writer_folder} due to invalid structure.")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Prepare TensorFlow dataset
def create_tf_dataset(pairs, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((pairs[:, 0], pairs[:, 1]), labels))
    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Check if the device can handle the batch size
def can_handle_batch_size(batch_size):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 3)  # Initial memory in GB
    memory_threshold = psutil.virtual_memory().total * 0.9 / (1024 ** 3)  # 90% of total memory

    print(f"Testing batch size {batch_size}: Initial memory usage: {initial_memory:.2f} GB")
    if initial_memory > memory_threshold:
        print(f"Batch size {batch_size} exceeds memory limit. Skipping...")
        return False
    return True

# Train model function
def train_model(model, dataset, epochs, batch_size):
    metrics = {"memory": [], "runtime": [], "loss": []}
    for epoch in range(epochs):
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

        history = model.fit(dataset, epochs=1, verbose=1)

        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        epoch_time = time.time() - start_time

        metrics["memory"].append({"before": memory_before, "after": memory_after})
        metrics["runtime"].append(epoch_time)
        metrics["loss"].append(history.history["loss"][0])

        print(f"Batch Size: {batch_size}, Epoch {epoch + 1}: Memory (Before: {memory_before:.2f} GB, After: {memory_after:.2f} GB), Runtime: {epoch_time:.2f}s, Loss: {metrics['loss'][-1]:.4f}")
    return metrics

# List of batch sizes to test
batch_sizes = [8, 16, 32, 64, 128]
epochs = 3  # Train for 3 epochs
results = {}

# Load and preprocess data
img_height, img_width = 155, 220  # Resize to SigNet's original input size
train_data, train_labels = load_and_preprocess_data(datasets, img_height, img_width)
pairs_train, labels_train = create_pairs(train_data, train_labels)

# Loop through each batch size
for batch_size in batch_sizes:
    print(f"\n--- Testing with batch size: {batch_size} ---")
    
    # Check if the device can handle the batch size
    if not can_handle_batch_size(batch_size):
        continue

    # Create a fresh dataset
    train_dataset = create_tf_dataset(pairs_train, labels_train, batch_size)
    
    # Create a fresh model
    model = create_siamese_network((img_height, img_width, 1))
    model.compile(optimizer=LossScaleOptimizer(tf.keras.optimizers.Adam(1e-4), dynamic=True), loss="binary_crossentropy")
    
    # Train and collect metrics
    results[batch_size] = train_model(model, train_dataset, epochs, batch_size)

# Compare results
for batch_size, metrics in results.items():
    print(f"Batch Size: {batch_size} -> Last Loss: {metrics['loss'][-1]:.4f}, Avg Runtime per Epoch: {np.mean(metrics['runtime']):.2f}s")
