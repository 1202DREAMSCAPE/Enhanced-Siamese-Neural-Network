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

# Function to apply SMOTE
def apply_smote(X, y):
    X_flat = X.reshape(X.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    return X_resampled.reshape(-1, X.shape[1], X.shape[2], X.shape[3]), y_resampled

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
    train_data, train_labels = generator.get_train_data()
    test_data, test_labels = generator.get_test_data()

    print("Applying SMOTE to balance the training data...")
    train_data, train_labels = apply_smote(np.array(train_data[0]), train_labels)

    # Ensure data type compatibility with mixed precision
    train_data = train_data.astype('float16')
    test_data = np.array(test_data[0]).astype('float16')

    return train_data, train_labels, test_data, test_labels

# Create datasets for TensorFlow
def create_tf_dataset(data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Function to compute biometric-specific metrics
def compute_biometric_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
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

# Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")

    # Load dataset with SMOTE
    train_data, train_labels, test_data, test_labels = load_data_with_smote(dataset_name, dataset_config)

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_data, train_labels, batch_size=32)
    test_dataset = create_tf_dataset(test_data, test_labels, batch_size=32)

    # Create and compile model
    model = create_siamese_network(input_shape=(155, 220, 1))
    contrastive_loss = ContrastiveLoss(margin=1.0)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=contrastive_loss)

    # Train model
    print(f"Training on {dataset_name}...")
    start_time = time.time()
    model.fit(train_dataset, epochs=10, verbose=1)
    end_time = time.time()

    # Evaluate model
    print(f"Evaluating on {dataset_name}...")
    y_pred = model.predict(test_dataset)
    compute_biometric_metrics(test_labels, y_pred)

    print(f"Training and Evaluation Time: {end_time - start_time:.2f} seconds")
