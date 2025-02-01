import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.optimizers import RMSprop
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras import backend as K
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Define Contrastive Loss (Baseline)
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = K.cast(y_true, y_pred.dtype)
    positive_loss = (1 - y_true) * K.square(y_pred)
    negative_loss = y_true * K.square(K.maximum(margin - y_pred, 0))
    return K.mean(positive_loss + negative_loss)

# ✅ Dataset Configuration (Same as Enhanced)
datasets = {
    # "CEDAR": {
    #     "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
    #     "train_writers": list(range(261, 300)),
    #     "test_writers": list(range(300, 315))
    # },
    # "BHSig260_Bengali": {
    #     "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
    #     "train_writers": list(range(1, 71)),
    # #     "test_writers": list(range(71, 100))
    # },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 169)),
        "test_writers": list(range(170, 260))
    }
}

# ✅ Function to Load Data
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

# ✅ Compute Class Imbalance (Train & Test)
def compute_class_distribution(train_labels, test_labels, dataset_name):
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_labels, return_counts=True)

    train_distribution = dict(zip(train_unique, train_counts))
    test_distribution = dict(zip(test_unique, test_counts))

    print(f"\n--- Class Imbalance in {dataset_name} ---")
    print(f"Training Data - Genuine: {train_distribution.get(1, 0)}, Forged: {train_distribution.get(0, 0)}")
    print(f"Test Data - Genuine: {test_distribution.get(1, 0)}, Forged: {test_distribution.get(0, 0)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Training Data Distribution
    axes[0].bar(["Genuine", "Forged"], [train_distribution.get(1, 0), train_distribution.get(0, 0)], color=["blue", "red"])
    axes[0].set_title(f"Train Data - {dataset_name}")
    axes[0].set_xlabel("Signature Type")
    axes[0].set_ylabel("Count")

    # Test Data Distribution
    axes[1].bar(["Genuine", "Forged"], [test_distribution.get(1, 0), test_distribution.get(0, 0)], color=["blue", "red"])
    axes[1].set_title(f"Test Data - {dataset_name}")
    axes[1].set_xlabel("Signature Type")

    plt.tight_layout()
    plt.show()


# ✅ Compute Scalability Metrics (Training Time & Memory Usage)
def compute_scalability_metrics(start_time, end_time, dataset_name):
    execution_time = end_time - start_time
    memory_usage = psutil.virtual_memory().percent  # Get memory usage
    print(f"\n--- Scalability Metrics for {dataset_name} ---")
    print(f"Training & Evaluation Time: {execution_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f}%")

# ✅ Compute Noise Sensitivity (Verifies Impact of Low-Quality Data)
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

# ✅ Function to Plot ROC Curve
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

# ✅ Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")

    # Load dataset
    train_data, train_labels, test_data, test_labels = load_data(dataset_name, dataset_config)

    # ✅ Compute Class Distribution to Show Imbalance
    compute_class_distribution(train_labels, test_labels, dataset_name)

    # Create and compile model
    model = create_siamese_network(input_shape=(155, 220, 1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=contrastive_loss)

    # Train model
    print(f"Training on {dataset_name}...")
    start_time = time.time()
    model.fit(train_data, train_labels, epochs=5, batch_size=8, validation_split=0.2, verbose=1)
    end_time = time.time()

    # Evaluate model
    print(f"Evaluating on {dataset_name}...")
    y_pred = model.predict(test_data)

    # ✅ Compute Scalability & Noise Sensitivity Metrics
    compute_scalability_metrics(start_time, end_time, dataset_name)
    compute_noise_sensitivity(test_labels, y_pred, dataset_name)

    # ✅ Plot ROC Curve
    plot_roc_curve(test_labels, y_pred, dataset_name)