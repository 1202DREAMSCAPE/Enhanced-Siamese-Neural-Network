import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import time
import psutil

from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score

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

# Initialize generator
generator = SignatureDataGenerator(
    dataset=datasets,
    img_height=155,
    img_width=220
)

# Load data
train_data, train_labels = generator.get_train_data()
test_data, test_labels = generator.get_test_data()

# Create Siamese network
model = create_siamese_network(input_shape=(155, 220, 1))
model.compile(optimizer=RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_data, train_labels, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate model
y_pred = model.predict(test_data)
y_pred_labels = (y_pred > 0.5).astype(int)

# Function to evaluate and log metrics
def log_metrics(y_true, y_pred, y_prob):
    # Calculate classification metrics
    report = classification_report(y_true, y_pred, target_names=["Genuine", "Forged"])
    cm = confusion_matrix(y_true, y_pred)
    auc_pr = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print("AUC-PR: {:.4f}".format(auc_pr))
    
# Simulate noise for noise sensitivity testing
def simulate_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

# Monitor computational scalability
def monitor_computation(model, data):
    process = psutil.Process()
    start_memory = process.memory_info().rss
    start_time = time.time()
    
    # Simulate a batch process
    predictions = model.predict(data)
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    memory_used = (end_memory - start_memory) / (1024 ** 2)  # Convert to MB
    elapsed_time = end_time - start_time
    
    print(f"Memory Used: {memory_used:.2f} MB")
    print(f"Time Taken: {elapsed_time:.2f} seconds")
    return predictions

# # Include hyperparameter sensitivity testing
# def evaluate_hyperparameters(model, train_data, val_data):
#     for lr in [0.01, 0.001, 0.0001]:
#         for batch_size in [16, 32, 64]:
#             print(f"Evaluating with Learning Rate: {lr}, Batch Size: {batch_size}")
#             model.compile(optimizer=RMSprop(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
#             start_time = time.time()
#             model.fit(train_data, batch_size=batch_size, epochs=5, validation_data=val_data, verbose=0)
#             end_time = time.time()
#             print(f"Training Time: {end_time - start_time:.2f} seconds")

# Log metrics
log_metrics(test_labels, y_pred_labels, y_pred)

# Simulate noise and re-evaluate
noisy_test_data = simulate_noise(test_data, noise_factor=0.1)
noisy_predictions = model.predict(noisy_test_data)
log_metrics(test_labels, (noisy_predictions > 0.5).astype(int), noisy_predictions)

# Monitor computation
monitor_computation(model, test_data)

# Hyperparameter evaluation (example)
# evaluate_hyperparameters(model, train_data, (test_data, test_labels))
