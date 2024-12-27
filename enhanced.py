import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Apply SMOTE with controlled sampling
def apply_smote(X1, X2, y, sampling_strategy=1.0):
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined = np.hstack([X1_flat, X2_flat])
    
    combined_resampled, y_resampled = smote.fit_resample(combined, y)
    
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
    "BHSig260_Bengali": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 36)),
        "test_writers": list(range(36, 50))
    },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 160)),
        "test_writers": list(range(160, 185))
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
    train_X1, train_X2, train_labels = apply_smote(train_X1, train_X2, train_labels, sampling_strategy=1.0)

    print("Class distribution after SMOTE:")
    display_class_distribution(train_labels)

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
    counts = [np.sum(labels == i) for i in range(len(label_names))]
    print("\n--- Class Distribution ---")
    for label_name, count in zip(label_names, counts):
        print(f"{label_name}: {count}")
    return counts

# Main Script
all_train_X1, all_train_X2, all_train_labels = [], [], []
all_test_X1, all_test_X2, all_test_labels = [], [], []

for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")
    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_smote(dataset_name, dataset_config)

    all_train_X1.append(train_X1)
    all_train_X2.append(train_X2)
    all_train_labels.append(train_labels)

    all_test_X1.append(test_X1)
    all_test_X2.append(test_X2)
    all_test_labels.append(test_labels)

# Combine all datasets
train_X1 = np.concatenate(all_train_X1, axis=0)
train_X2 = np.concatenate(all_train_X2, axis=0)
train_labels = np.concatenate(all_train_labels, axis=0)
test_X1 = np.concatenate(all_test_X1, axis=0)
test_X2 = np.concatenate(all_test_X2, axis=0)
test_labels = np.concatenate(all_test_labels, axis=0)

print("\n--- Unified Dataset ---")
print("Training Data Distribution:")
display_class_distribution(train_labels)
print("Testing Data Distribution:")
display_class_distribution(test_labels)

# Create TensorFlow datasets
train_dataset = create_tf_dataset(train_X1, train_X2, train_labels, batch_size=8)
test_dataset = create_tf_dataset(test_X1, test_X2, test_labels, batch_size=8)

# Create and compile model
model = create_siamese_network(input_shape=(155, 220, 1))
contrastive_loss = ContrastiveLoss(margin=1.0)
model.compile(optimizer=RMSprop(learning_rate=0.001), loss=contrastive_loss)

# Train model
print(f"\nTraining on Unified Dataset...")
start_time = time.time()
model.fit(train_dataset, epochs=5, verbose=1)
end_time = time.time()

# Evaluate model
print("\nEvaluating on Unified Dataset...")
y_pred = model.predict(test_dataset)
compute_biometric_metrics(test_labels, y_pred)

print(f"\nTraining and Evaluation Time: {end_time - start_time:.2f} seconds")
