from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score

datasets = {
    "CEDAR": {
        "path": "/path_to_dataset/CEDAR",
        "train_writers": list(range(261, 300)),  # CEDAR train: writer_261 to writer_299
        "test_writers": list(range(300, 317))   # CEDAR test: writer_300 to writer_316
    },
    "BHSig260_Bengali": {
        "path": "/path_to_dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),    # Bengali train: writer_001 to writer_070
        "test_writers": list(range(71, 101))   # Bengali test: writer_071 to writer_100
    },
    "BHSig260_Hindi": {
        "path": "/path_to_dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 213)),  # Hindi train: writer_101 to writer_212
        "test_writers": list(range(213, 261))   # Hindi test: writer_213 to writer_260
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
model.save("siamese_model1.keras")

# # when you want to load the model again, you can use the following code:
# from tensorflow.keras.models import load_model

# # Load the model saved in .keras format
# model = load_model("siamese_model.keras", custom_objects={"contrastive_loss": contrastive_loss})
