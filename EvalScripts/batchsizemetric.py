import time
import os
import psutil
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
import sys

# Add the parent directory to sys.path to locate SigNet_v1 and SignatureDataGenerator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SigNet_v1 import create_siamese_network
from SignatureDataGenerator import SignatureDataGenerator

# Enable mixed precision
set_global_policy('mixed_float16')

# Print mixed precision status
from tensorflow.keras.mixed_precision import global_policy
print(f"Mixed precision policy: {global_policy()}")

# Define datasets
datasets = {
    "CEDAR": {
        "path": "/content/drive/My Drive/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 317))
    },
    "BHSig260_Bengali": {
        "path": "/content/drive/My Drive/Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 101))
    },
    "BHSig260_Hindi": {
        "path": "/content/drive/My Drive/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 213)),
        "test_writers": list(range(213, 261))
    }
}

# Initialize the generator
try:
    generator = SignatureDataGenerator(dataset=datasets, img_height=155, img_width=220)
    train_data, train_labels = generator.get_train_data()
    test_data, test_labels = generator.get_test_data()

    # Validate dataset sizes
    if train_data[0].shape[0] == 0 or test_data[0].shape[0] == 0:
        raise ValueError("Train or Test data is empty! Please check the dataset or the generator.")
except Exception as e:
    print(f"Error initializing SignatureDataGenerator: {e}")
    sys.exit(1)

# Print dataset shapes
print(f"Train Data Shape: X1={train_data[0].shape}, X2={train_data[1].shape}, Labels={train_labels.shape}")
print(f"Test Data Shape: X1={test_data[0].shape}, X2={test_data[1].shape}, Labels={test_labels.shape}")

# Create and compile the model
input_shape = (155, 220, 1)
model = create_siamese_network(input_shape)

# Define contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4), loss=contrastive_loss)

# Monitor system usage
def log_system_usage():
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"Memory Usage: {memory:.2f} GB")

# Test batch sizes
def test_batch_sizes(model, train_data, train_labels, test_data, test_labels, batch_sizes):
    training_results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        try:
            start_time = time.time()

            # Log memory usage before training
            print("Before Training:")
            log_system_usage()

            # Train the model
            history = model.fit(
                x=train_data, y=train_labels,
                validation_data=(test_data, test_labels),
                batch_size=batch_size,
                epochs=5,  # Short training for testing
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: (
                            print(f"After Epoch {epoch + 1}:"),
                            log_system_usage()
                        )
                    )
                ]
            )

            # Log memory usage after training
            print("After Training:")
            log_system_usage()

            # Record elapsed time
            elapsed_time = time.time() - start_time
            print(f"Batch size {batch_size} completed in {elapsed_time:.2f} seconds!")

            # Save results
            training_results[batch_size] = elapsed_time

        except tf.errors.ResourceExhaustedError:
            print(f"Batch size {batch_size} failed: Out of memory")
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")

    # Print summary
    print("\nTraining Results:")
    for batch_size, time_taken in training_results.items():
        print(f"Batch size {batch_size}: {time_taken:.2f} seconds")

# Define batch sizes to test
batch_sizes = [16, 32, 64, 128]

# Run the batch size test
test_batch_sizes(model, train_data, train_labels, test_data, test_labels, batch_sizes)
