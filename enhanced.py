import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import RMSprop
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.losses import Loss
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import time

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define Contrastive Loss
class ContrastiveLoss(Loss):
    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        positive_loss = (1 - y_true) * tf.square(y_pred)
        negative_loss = y_true * tf.square(tf.maximum(self.margin - y_pred, 0))
        return tf.reduce_mean(positive_loss + negative_loss)

# Apply SMOTE (adjusted for paired data)
# Apply SMOTE with controlled sampling
def apply_smote(X1, X2, y, sampling_strategy=1.0):
    """
    Apply SMOTE to oversample the minority class with controlled limits.

    Args:
        X1 (np.array): First input array (e.g., left branch of Siamese input).
        X2 (np.array): Second input array (e.g., right branch of Siamese input).
        y (np.array): Labels array.
        sampling_strategy (float): Fraction of the majority class to oversample.

    Returns:
        X1_resampled, X2_resampled, y_resampled: Resampled data and labels.
    """
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined = np.hstack([X1_flat, X2_flat])
    
    # Apply SMOTE with controlled sampling
    combined_resampled, y_resampled = smote.fit_resample(combined, y)
    
    # Restore the paired input shape
    X1_resampled = combined_resampled[:, :X1_flat.shape[1]].reshape(-1, *X1.shape[1:])
    X2_resampled = combined_resampled[:, X1_flat.shape[1]:].reshape(-1, *X2.shape[1:])
    
    return X1_resampled, X2_resampled, y_resampled

# Dataset Configuration
datasets = {
    "CEDAR": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 316))
    },
}

# Load dataset with SMOTE
def load_data_with_smote(dataset_name, dataset_config):
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
    (train_X1, train_X2), train_labels = generator.get_train_data()
    (test_X1, test_X2), test_labels = generator.get_test_data()

    print("Class distribution before SMOTE:")
    display_class_distribution(train_labels)

    print("Applying SMOTE to balance the training data...")
    train_X1, train_X2, train_labels = apply_smote(
        train_X1, train_X2, train_labels, sampling_strategy=0.5
    )  # Oversample to 50% of the majority class

    print("Class distribution after SMOTE:")
    display_class_distribution(train_labels)

    # Ensure data type compatibility with mixed precision
    train_X1 = train_X1.astype('float16')
    train_X2 = train_X2.astype('float16')
    test_X1 = test_X1.astype('float16')
    test_X2 = test_X2.astype('float16')

    return (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels)


# Create datasets for TensorFlow
def create_tf_dataset(X1, X2, labels, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), labels))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Function to compute biometric-specific metrics
def compute_biometric_metrics(y_true, y_pred):
    y_pred_labels = (y_pred < 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    # Biometric metrics: Genuine Acceptance Rate (GAR) and False Rejection Rate (FRR)
    gar = recall  # Equivalent to TPR for genuine samples
    frr = 1 - gar

    print("\n--- Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Genuine Acceptance Rate (GAR): {gar:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")

    return accuracy, precision, recall, f1, gar, frr

# Function to display class distribution
def display_class_distribution(labels, label_names=['Genuine', 'Forged']):
    """
    Displays the class distribution for given labels.

    Args:
        labels (np.array): Array of labels (e.g., 1 for Genuine, 0 for Forged).
        label_names (list): List of class names corresponding to labels.

    Returns:
        counts (list): List of counts for each class.
    """
    # Count occurrences for each class
    counts = [np.sum(labels == i) for i in range(len(label_names))]

    # Print class distribution
    print("\n--- Class Distribution ---")
    for label_name, count in zip(label_names, counts):
        print(f"{label_name}: {count}")

    return counts

# Example dataset labels (Replace with actual labels)
# Assuming '1' is for Genuine and '0' is for Forged


# Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")

    # Load dataset with SMOTE
    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_smote(dataset_name, dataset_config)

    # Display class distribution for training labels
    print(f"Class distribution for training data in {dataset_name}:")
    train_class_counts = display_class_distribution(train_labels)

    # Display class distribution for testing labels
    print(f"Class distribution for testing data in {dataset_name}:")
    test_class_counts = display_class_distribution(test_labels)

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_X1, train_X2, train_labels, batch_size=8)
    test_dataset = create_tf_dataset(test_X1, test_X2, test_labels, batch_size=8)

    # Create and compile model
    model = create_siamese_network(input_shape=(155, 220, 1))
    contrastive_loss = ContrastiveLoss(margin=1.0)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=contrastive_loss)

    # Train model
    print(f"Training on {dataset_name}...")
    start_time = time.time()
    model.fit(train_dataset, epochs=5, verbose=1)
    end_time = time.time()

    # Evaluate model
    print(f"Evaluating on {dataset_name}...")
    y_pred = model.predict(test_dataset)
    compute_biometric_metrics(test_labels, y_pred)

    print(f"Training and Evaluation Time: {end_time - start_time:.2f} seconds")
