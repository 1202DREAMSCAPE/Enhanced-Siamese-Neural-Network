import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from SignatureDataGenerator import SignatureDataGenerator  # Your data generator
from SigNet_v1 import euclidean_distance  # Import custom distance function

# Custom contrastive loss (if used in your model)
def contrastive_loss(y_true, y_pred):
    margin = 1
    return np.mean(y_true * np.square(y_pred) + (1 - y_true) * np.square(np.maximum(margin - y_pred, 0)))

# Load the model
model_path = input("Enter the model file path (e.g., siamese_model.keras): ")
test_data_path = input("Enter the test dataset path (e.g., /path/to/Dataset): ")
model = load_model(model_path, custom_objects={"contrastive_loss": contrastive_loss, "euclidean_distance": euclidean_distance})

# Initialize test data generator
generator = SignatureDataGenerator(
    dataset=test_data_path,
    num_train_writers=0,  # No training writers needed
    num_test_writers=5,   # Number of test writers
    img_height=155,
    img_width=220
)

# Load test data
test_data, test_labels = generator.get_test_data()

# Predict distances
distances = model.predict(test_data)

# Evaluate thresholds
thresholds = np.arange(0.1, 0.9, 0.1)  # Thresholds from 0.1 to 0.9
metrics = []

for t in thresholds:
    predictions = (distances < t).astype(int)
    cm = confusion_matrix(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    metrics.append((t, precision, recall, f1, accuracy))
    
    print(f"Threshold: {t:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}, Accuracy: {accuracy:.3f}")
    print("-" * 50)

# Plot F1-Score and Accuracy vs Threshold
thresholds, precisions, recalls, f1_scores, accuracies = zip(*metrics)

plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label="F1-Score", marker="o")
plt.plot(thresholds, accuracies, label="Accuracy", marker="o")
plt.axvline(x=0.5, color='r', linestyle='--', label="Threshold=0.5")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs F1-Score and Accuracy")
plt.legend()
plt.grid()
plt.show()
