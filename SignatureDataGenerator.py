import numpy as np
import os
from tensorflow.keras.preprocessing import image
import random

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    """
    Data generator for a signature dataset with 5 writers,
    each having 10 genuine and 10 forged signatures.
    """

    def __init__(self, dataset, num_train_writers=4, num_test_writers=1,
                 img_height=155, img_width=220, batch_sz=4):
        """
        Initialize the generator with dataset parameters.
        
        Args:
            dataset: Path to the dataset directory.
            num_train_writers: Number of writers for training.
            num_test_writers: Number of writers for testing.
            img_height: Target height for resizing images.
            img_width: Target width for resizing images.
            batch_sz: Batch size for training/testing.
        """
        self.dataset = dataset
        self.num_train_writers = num_train_writers
        self.num_test_writers = num_test_writers
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz
        
        # Get all writers in the dataset
# Ignore non-directory files such as .DS_Store
        self.writers = [d for d in sorted(os.listdir(self.dataset)) if not d.startswith('.')]
        random.shuffle(self.writers)  # Shuffle for randomness
        
        # Split writers into training and testing groups
        self.train_writers = self.writers[:self.num_train_writers]
        self.test_writers = self.writers[self.num_train_writers:]
    
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

    def generate_pairs(self, writer):
        """
        Generate positive and negative pairs for a given writer.
        Args:
            writer: Name of the writer folder (e.g., "writer1").
        Returns:
            A list of pairs (image1, image2) and their corresponding labels.
        """
        writer_path = os.path.join(self.dataset, writer)
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        # Check if the folders exist
        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            raise FileNotFoundError(f"Missing required directory: {genuine_path} or {forged_path}")

        # Get files from the directories
        genuine_files = [f for f in sorted(os.listdir(genuine_path)) if os.path.isfile(os.path.join(genuine_path, f))]
        forged_files = [f for f in sorted(os.listdir(forged_path)) if os.path.isfile(os.path.join(forged_path, f))]

        # Create positive pairs (genuine-genuine)
        positive_pairs = []
        for i in range(len(genuine_files)):
            for j in range(i + 1, len(genuine_files)):
                img1 = self.preprocess_image(os.path.join(writer_path, "genuine", genuine_files[i]))
                img2 = self.preprocess_image(os.path.join(writer_path, "genuine", genuine_files[j]))
                positive_pairs.append((img1, img2, 1))
        
        # Create negative pairs (genuine-forged)
        negative_pairs = []
        for genuine_img in genuine_files:
            for forged_img in forged_files:
                img1 = self.preprocess_image(os.path.join(writer_path, "genuine", genuine_img))
                img2 = self.preprocess_image(os.path.join(writer_path, "forged", forged_img))
                negative_pairs.append((img1, img2, 0))
        
        return positive_pairs + negative_pairs

    def get_train_data(self):
        """
        Generate training data pairs from train writers.
        Returns:
            Pair data and labels for training.
        """
        train_pairs = []
        for writer in self.train_writers:
            train_pairs.extend(self.generate_pairs(writer))
        
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
        for writer in self.test_writers:
            test_pairs.extend(self.generate_pairs(writer))
        
        # Split pairs into inputs and labels
        X1 = np.array([pair[0] for pair in test_pairs])
        X2 = np.array([pair[1] for pair in test_pairs])
        y = np.array([pair[2] for pair in test_pairs])
        
        return [X1, X2], y
