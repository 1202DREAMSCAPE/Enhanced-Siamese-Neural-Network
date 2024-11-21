from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score

# Initialize the generator
generator = SignatureDataGenerator(
    dataset="/Users/christelle/Downloads/Thesis/Dataset",  # Absolute path to the dataset
    num_train_writers=4,
    num_test_writers=1,
    img_height=155,
    img_width=220
)

# Load training and testing data
train_data, train_labels = generator.get_train_data()
test_data, test_labels = generator.get_test_data()

# Model setup
input_shape = (155, 220, 1)  # Image dimensions
model = create_siamese_network(input_shape)

# Compile the model with contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

model.compile(loss=contrastive_loss, optimizer=RMSprop(learning_rate=1e-4))
model.summary()

# Train the model
history = model.fit(
    x=train_data, y=train_labels,
    validation_data=(test_data, test_labels),
    batch_size=4, epochs=20
)

# Evaluate the model
distances = model.predict(test_data)  # Predicted distances

# Apply a threshold to classify distances
threshold = 0.5  
predictions = (distances < threshold).astype(int)

# Compute metrics
roc_auc = roc_auc_score(test_labels, distances)
accuracy = accuracy_score(test_labels, predictions)

print(f"ROC-AUC: {roc_auc}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save("siamese_model.keras")

# # when you want to load the model again, you can use the following code:
# from tensorflow.keras.models import load_model

# # Load the model saved in .keras format
# model = load_model("siamese_model.keras", custom_objects={"contrastive_loss": contrastive_loss})
