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

# Apply the optimal threshold for classification
threshold = 0.4 
predictions = (distances < threshold).astype(int)

# Evaluate metrics
roc_auc = roc_auc_score(test_labels, distances)
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

# Print metrics
print(f"Threshold: {threshold}")
print(f"ROC-AUC: {roc_auc}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Forged", "Genuine"], yticklabels=["Forged", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.show()
