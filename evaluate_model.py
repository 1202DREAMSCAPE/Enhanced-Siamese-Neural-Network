import numpy as np
from SigNet_v1 import euclidean_distance
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Custom contrastive loss (if used in your model)
def contrastive_loss(y_true, y_pred):
    margin = 1
    return np.mean(y_true * np.square(y_pred) + (1 - y_true) * np.square(np.maximum(margin - y_pred, 0)))

# Load the saved model
model_path = input("Enter the model file path (e.g., siamese_model.keras): ")
model = load_model(model_path, custom_objects={"contrastive_loss": contrastive_loss, "euclidean_distance": euclidean_distance})

# Specify test dataset path
test_data_path = "/Users/christelle/Downloads/Thesis/Dataset"  # Replace with your test dataset path

# Import your data generator or dataset loading logic
from SignatureDataGenerator import SignatureDataGenerator

# Initialize the generator for test data
generator = SignatureDataGenerator(
    dataset=test_data_path,
    num_train_writers=0,  # No training writers needed
    num_test_writers=5,   # Number of test writers
    img_height=155,
    img_width=220
)

# Load test data
test_data, test_labels = generator.get_test_data()

# Evaluate the model
distances = model.predict(test_data)

# Apply a threshold to classify distances
threshold = 0.5  # Adjust based on validation results
predictions = (distances < threshold).astype(int)

# Compute metrics
roc_auc = roc_auc_score(test_labels, distances)
accuracy = accuracy_score(test_labels, predictions)

print(f"ROC-AUC: {roc_auc}")
print(f"Accuracy: {accuracy}")

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Forged", "Genuine"], yticklabels=["Forged", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
