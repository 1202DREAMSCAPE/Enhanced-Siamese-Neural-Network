import time
import os
import psutil
from tensorflow.keras.mixed_precision import set_global_policy
from SigNet_v1 import create_siamese_network
from SignatureDataGenerator import SignatureDataGenerator

# Mixed precision training for Mac M1 (optional)
set_global_policy('mixed_float16')

# Define datasets and generator
datasets = {
    "CEDAR": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 317))
    },
    "BHSig260_Bengali": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 101))
    },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 213)),
        "test_writers": list(range(213, 261))
    }
}

generator = SignatureDataGenerator(dataset=datasets, img_height=155, img_width=220)
train_data, train_labels = generator.get_train_data()
test_data, test_labels = generator.get_test_data()

# Create the model
input_shape = (155, 220, 1)
model = create_siamese_network(input_shape)

# Monitor system usage
def log_system_usage():
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"Memory Usage: {memory:.2f} GB")

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
                batch_size=batch_size, epochs=5  # Short training
            )

            # Log memory usage after training
            print("After Training:")
            log_system_usage()

            # Record elapsed time
            elapsed_time = time.time() - start_time
            print(f"Batch size {batch_size} completed in {elapsed_time:.2f} seconds!")

            # Save results
            training_results[batch_size] = elapsed_time

        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")

    # Print summary
    print("\nTraining Results:")
    for batch_size, time_taken in training_results.items():
        print(f"Batch size {batch_size}: {time_taken:.2f} seconds")

# Define batch sizes to test
batch_sizes = [32, 64, 128]

# Run the batch size test
test_batch_sizes(model, train_data, train_labels, test_data, test_labels, batch_sizes)
