import matplotlib.pyplot as plt
import numpy as np
import time
import faiss  # FAISS for efficient similarity search
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE  # ✅ SMOTE for class balancing
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network  # Enhanced SigNet with Triplet Loss
from esrgan import ESRGAN  # ESRGAN for super-resolution enhancement
from tensorflow.keras.models import load_model

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load ESRGAN for enhancement
esrgan = ESRGAN()

# ✅ Triplet Loss Function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return K.mean(loss)

# ✅ Dataset Configuration
datasets = {
    "CEDAR": { "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
               "train_writers": list(range(261, 300)), "test_writers": list(range(300, 315)) },
    "BHSig260_Bengali": { "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
                          "train_writers": list(range(1, 71)), "test_writers": list(range(71, 100)) },
    "BHSig260_Hindi": { "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
                        "train_writers": list(range(101, 213)), "test_writers": list(range(213, 260)) }
}

# ✅ Apply SMOTE for Class Balancing
def apply_smote(X1, X2, y, sampling_strategy=1.0):
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined = np.hstack([X1_flat, X2_flat])
    combined_resampled, y_resampled = smote.fit_resample(combined, y)

    X1_resampled = combined_resampled[:, :X1_flat.shape[1]].reshape(-1, *X1.shape[1:])
    X2_resampled = combined_resampled[:, X1_flat.shape[1]:].reshape(-1, *X2.shape[1:])

    print(f"SMOTE applied: {len(y_resampled) - len(y)} new samples added.")
    return X1_resampled, X2_resampled, y_resampled

# ✅ ESRGAN Preprocessing
def preprocess_with_esrgan(img_array):
    return esrgan.enhance(img_array)

# ✅ Load Dataset with ESRGAN + SMOTE
def load_data_with_esrgan_and_smote(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={ dataset_name: dataset_config },
        img_height=155, img_width=220
    )

    (train_X1, train_X2), train_labels = generator.get_train_data()
    (test_X1, test_X2), test_labels = generator.get_test_data()

    # Apply ESRGAN preprocessing
    train_X1 = np.array([preprocess_with_esrgan(img) for img in train_X1])
    train_X2 = np.array([preprocess_with_esrgan(img) for img in train_X2])

    # Apply SMOTE after ESRGAN
    train_X1, train_X2, train_labels = apply_smote(train_X1, train_X2, train_labels)

    return (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels)

# ✅ Train & Save Separate Models
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Training Model on {dataset_name} ---")

    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_esrgan_and_smote(dataset_name, dataset_config)

    train_dataset = tf.data.Dataset.from_tensor_slices(((train_X1, train_X2), train_labels)).batch(8)
    test_dataset = tf.data.Dataset.from_tensor_slices(((test_X1, test_X2), test_labels)).batch(8)

    model = create_siamese_network(input_shape=(155, 220, 1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

    model.fit(train_dataset, epochs=10, verbose=1)
    model.save(f"{dataset_name}_siamese_model.h5")
    print(f"Model saved as {dataset_name}_siamese_model.h5")

# ✅ Train Unified Model on All Data
print("\n--- Training Unified Model on All Datasets ---")

all_train_X1, all_train_X2, all_train_labels = [], [], []
all_test_X1, all_test_X2, all_test_labels = [], [], []

for dataset_name, dataset_config in datasets.items():
    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_esrgan_and_smote(dataset_name, dataset_config)
    
    all_train_X1.append(train_X1)
    all_train_X2.append(train_X2)
    all_train_labels.append(train_labels)

    all_test_X1.append(test_X1)
    all_test_X2.append(test_X2)
    all_test_labels.append(test_labels)

train_X1 = np.concatenate(all_train_X1, axis=0)
train_X2 = np.concatenate(all_train_X2, axis=0)
train_labels = np.concatenate(all_train_labels, axis=0)

test_X1 = np.concatenate(all_test_X1, axis=0)
test_X2 = np.concatenate(all_test_X2, axis=0)
test_labels = np.concatenate(all_test_labels, axis=0)

train_dataset = tf.data.Dataset.from_tensor_slices(((train_X1, train_X2), train_labels)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_X1, test_X2), test_labels)).batch(8)

unified_model = create_siamese_network(input_shape=(155, 220, 1))
unified_model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

unified_model.fit(train_dataset, epochs=10, verbose=1)
unified_model.save("unified_siamese_model.h5")
print("Unified model saved as unified_siamese_model.h5")
