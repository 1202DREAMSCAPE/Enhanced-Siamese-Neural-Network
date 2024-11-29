import numpy as np
import os
from tensorflow.keras.preprocessing import image
import random

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    """
    Data generator for multiple signature datasets with writers,
    each having genuine and forged signatures.
    """

    def __init__(self, dataset, img_height=155, img_width=220, batch_sz=4):
        """
        Initialize the generator with dataset parameters.

        Args:
            dataset: Dictionary of datasets with paths and writer splits.
            img_height: Target height for resizing images.
            img_width: Target width for resizing images.
            batch_sz: Batch size for training/testing.
        """
        self.dataset = dataset
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz

        # Store data and labels
        self.train_writers = []
        self.test_writers = []

        # Iterate over multiple datasets
        for dataset_name, dataset_info in dataset.items():
            dataset_path = dataset_info["path"]
            train_writers = dataset_info["train_writers"]
            test_writers = dataset_info["test_writers"]

            # Store writers for training and testing
            self.train_writers.extend([(dataset_path, writer) for writer in train_writers])
            self.test_writers.extend([(dataset_path, writer) for writer in test_writers])

    def preprocess_image(self, img_path):
        """
        Load and preprocess an image from a file path.
        Args:
            img_path: Path to the image file.
        Returns:
            Preprocessed image as a NumPy array.
        """
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width), color_mode='grayscale')
        img = image.img_to_array(img)
        img /= 255.0  # Normalize pixel values to [0, 1]
        return img

    def generate_pairs(self, dataset_path, writer):
        """
        Generate positive and negative pairs for a given writer.

        Args:
            dataset_path: Path to the dataset.
            writer: Name or ID of the writer folder (e.g., "writer_001").
        Returns:
            A list of pairs (image1, image2) and their corresponding labels.
        """
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        # Check if the folders exist
        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            print(f"Missing required directory: {genuine_path} or {forged_path}")
            return []

        # Get files from the directories
        genuine_files = [f for f in sorted(os.listdir(genuine_path)) if os.path.isfile(os.path.join(genuine_path, f))]
        forged_files = [f for f in sorted(os.listdir(forged_path)) if os.path.isfile(os.path.join(forged_path, f))]

        # Create positive pairs (genuine-genuine)
        positive_pairs = []
        for i in range(len(genuine_files)):
            for j in range(i + 1, len(genuine_files)):
                img1 = self.preprocess_image(os.path.join(genuine_path, genuine_files[i]))
                img2 = self.preprocess_image(os.path.join(genuine_path, genuine_files[j]))
                positive_pairs.append((img1, img2, 1))

        # Create negative pairs (genuine-forged)
        negative_pairs = []
        for genuine_img in genuine_files:
            for forged_img in forged_files:
                img1 = self.preprocess_image(os.path.join(genuine_path, genuine_img))
                img2 = self.preprocess_image(os.path.join(forged_path, forged_img))
                negative_pairs.append((img1, img2, 0))

        return positive_pairs + negative_pairs

    def get_train_data(self):
        """
        Generate training data pairs from train writers.

        Returns:
            Pair data and labels for training.
        """
        train_pairs = []
        for dataset_path, writer in self.train_writers:
            train_pairs.extend(self.generate_pairs(dataset_path, writer))

        # Shuffle the training pairs
        random.shuffle(train_pairs)

        # Split pairs into inputs and labels
        X1 = np.array([pair[0] for pair in train_pairs])
        X2 = np.array([pair[1] for pair in train_pairs])
        y = np.array([pair[2] for pair in train_pairs])

        return [X1, X2], y

    def get_test_data(self):
        """
        Generate testing data pairs from test writers.

        Returns:
            Pair data and labels for testing.
        """
        test_pairs = []
        for dataset_path, writer in self.test_writers:
            test_pairs.extend(self.generate_pairs(dataset_path, writer))

        # Split pairs into inputs and labels
        X1 = np.array([pair[0] for pair in test_pairs])
        X2 = np.array([pair[1] for pair in test_pairs])
        y = np.array([pair[2] for pair in test_pairs])

        return [X1, X2], y
