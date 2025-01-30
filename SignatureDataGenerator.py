import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
import random
from itertools import combinations

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    """
    A data generator for multiple signature datasets containing genuine and forged signatures.
    """

    def __init__(self, dataset, img_height=155, img_width=220, batch_sz=8, apply_esrgan=False, esrgan_model=None):
        """
        Initialize the generator with dataset parameters.

        Args:
            dataset: Dictionary containing dataset paths and writer splits.
            img_height: Target height for resizing images.
            img_width: Target width for resizing images.
            batch_sz: Batch size for training/testing.
            apply_esrgan: Whether to use ESRGAN for super-resolution.
            esrgan_model: The ESRGAN model instance (Only used in enhanced model).
        """
        self.dataset = dataset
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz
        self.apply_esrgan = apply_esrgan
        self.esrgan = esrgan_model if apply_esrgan else None

        # Storage for train and test writers
        self.train_writers = []
        self.test_writers = []

        # Validate dataset paths and load writer information
        self._load_writers()

    def _load_writers(self):
        """
        Load writer directories and validate existence.
        """
        for dataset_name, dataset_info in self.dataset.items():
            dataset_path = dataset_info["path"]
            train_writers = dataset_info["train_writers"]
            test_writers = dataset_info["test_writers"]

            for writer in train_writers + test_writers:
                writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
                if os.path.exists(writer_path):
                    if writer in train_writers:
                        self.train_writers.append((dataset_path, writer))
                    else:
                        self.test_writers.append((dataset_path, writer))
                else:
                    print(f"⚠ Warning: Writer directory not found: {writer_path}")

    def preprocess_image(self, img_path):
        """
        Load and preprocess an image.

        Args:
            img_path: Path to the image file.

        Returns:
            Preprocessed image as a NumPy array.
        """
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Missing image file {img_path}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

        img = image.load_img(img_path, target_size=(self.img_height, self.img_width), color_mode='grayscale')
        img = image.img_to_array(img)

        # Apply ESRGAN Super-Resolution if enabled
        if self.apply_esrgan and self.esrgan:
            img = self.esrgan.enhance(img)

        img /= 255.0  # Normalize pixel values to [0, 1]
        return img

    def generate_pairs(self, dataset_path, writer):
        """
        Generate positive (genuine-genuine) and negative (genuine-forged) pairs.

        Args:
            dataset_path: Path to dataset.
            writer: Writer ID.

        Returns:
            List of (image1, image2, label) pairs.
        """
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            print(f"⚠ Warning: Missing required directories for writer {writer}")
            return []

        # Load file names
        genuine_files = sorted([os.path.join(genuine_path, f) for f in os.listdir(genuine_path) if f.endswith(('.png', '.jpg'))])
        forged_files = sorted([os.path.join(forged_path, f) for f in os.listdir(forged_path) if f.endswith(('.png', '.jpg'))])

        if len(genuine_files) < 2 or len(forged_files) == 0:
            print(f"⚠ Warning: Not enough images for writer {writer} (genuine: {len(genuine_files)}, forged: {len(forged_files)})")
            return []

        # Create pairs using NumPy for speed
        genuine_imgs = np.array([self.preprocess_image(f) for f in genuine_files])
        forged_imgs = np.array([self.preprocess_image(f) for f in forged_files])

        # Generate positive pairs (genuine-genuine)
        positive_pairs = [(genuine_imgs[i], genuine_imgs[j], 1) for i, j in combinations(range(len(genuine_imgs)), 2)]

        # Generate negative pairs (genuine-forged)
        negative_pairs = [(genuine_imgs[i], forged_imgs[j], 0) for i in range(len(genuine_imgs)) for j in range(len(forged_imgs))]

        return positive_pairs + negative_pairs

    def get_data(self, writers_list):
        """
        Generate data pairs from the given list of writers.

        Args:
            writers_list: List of (dataset_path, writer_id) tuples.

        Returns:
            (X1, X2), y - Processed input pairs and labels.
        """
        pairs = []
        for dataset_path, writer in writers_list:
            pairs.extend(self.generate_pairs(dataset_path, writer))

        if not pairs:
            print("⚠ Warning: No data pairs generated!")
            return None, None

        # Shuffle data for randomness
        random.shuffle(pairs)

        # Convert to NumPy arrays
        X1 = np.array([pair[0] for pair in pairs], dtype=np.float32)
        X2 = np.array([pair[1] for pair in pairs], dtype=np.float32)
        y = np.array([pair[2] for pair in pairs], dtype=np.int8)

        return [X1, X2], y

    def get_train_data(self):
        """
        Generate training data pairs.
        """
        return self.get_data(self.train_writers)

    def get_test_data(self):
        """
        Generate testing data pairs.
        """
        return self.get_data(self.test_writers)
