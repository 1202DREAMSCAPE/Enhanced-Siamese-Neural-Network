import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.optimizers import RMSprop
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

# Function for Triplet Loss
def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

# Dataset Configuration
datasets = {
    "CEDAR": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 315))
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

# Function to load data
def load_data(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={
            dataset_name: {
                "path": dataset_config["path"],
                "train_writers": dataset_config["train_writers"],
                "test_writers": dataset_config["test_writers"]
            }
        },
        img_height=155,
        img_width=220
    )
    train_data, train_labels = generator.get_train_data()
    test_data, test_labels = generator.get_test_data()

    return train_data, train_labels, test_data, test_labels

# Compute scalability issues
def compute_scalability_metrics(start_time, end_time, dataset_name):
    execution_time = end_time - start_time
    memory_usage = psutil.virtual_memory().percent  # Get memory usage
    print(f"\n--- Scalability Metrics for {dataset_name} ---")
    print(f"Training & Evaluation Time: {execution_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f}%")

# Compute noise sensitivity
def compute_noise_sensitivity(y_true, y_pred, dataset_name):
    y_pred_labels = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred)

    # Biometric-specific metrics
    gar = recall  # Genuine Acceptance Rate
    frr = 1 - gar  # False Rejection Rate
    far = 1 - precision  # False Acceptance Rate

    print("\n--- Noise Sensitivity Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (GAR): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Genuine Acceptance Rate (GAR): {gar:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], "k--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.show()

# Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")

    # Load dataset
    train_data, train_labels, test_data, test_labels = load_data(dataset_name, dataset_config)

    # Create and compile model
    model = create_siamese_network(input_shape=(155, 220, 1))
    loss_function = triplet_loss(margin=1.0)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=loss_function)

    # Train model
    print(f"Training on {dataset_name}...")
    start_time = time.time()
    model.fit(train_data, train_labels, epochs=5, batch_size=8, validation_split=0.2, verbose=1)
    end_time = time.time()

    # Evaluate model
    print(f"Evaluating on {dataset_name}...")
    y_pred = model.predict(test_data)

    # Compute scalability & noise sensitivity metrics
    compute_scalability_metrics(start_time, end_time, dataset_name)
    compute_noise_sensitivity(test_labels, y_pred, dataset_name)

    # Plot ROC curve
    plot_roc_curve(test_labels, y_pred, dataset_name)
