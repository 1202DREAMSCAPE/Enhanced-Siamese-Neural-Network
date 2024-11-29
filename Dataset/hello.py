import os
import shutil

# Set the path to your dataset
dataset_path = "/Users/christelle/Downloads/Thesis/Dataset"

# Define the dataset folders
datasets = ["CEDAR", "BHSig260_Bengali", "BHSig260_Hindi"]

# Function to organize files into genuine and forged folders
def organize_files(dataset_name):
    dataset_dir = os.path.join(dataset_path, dataset_name)

    # Traverse all writer folders
    for writer_folder in os.listdir(dataset_dir):
        writer_path = os.path.join(dataset_dir, writer_folder)

        # Check if it's a valid directory
        if os.path.isdir(writer_path) and writer_folder.startswith("writer_"):
            # Create genuine and forged folders
            genuine_folder = os.path.join(writer_path, "genuine")
            forged_folder = os.path.join(writer_path, "forged")
            os.makedirs(genuine_folder, exist_ok=True)
            os.makedirs(forged_folder, exist_ok=True)

            # Move files based on their prefix
            for file in os.listdir(writer_path):
                file_path = os.path.join(writer_path, file)
                if os.path.isfile(file_path):
                    if file.lower().startswith("genuine"):
                        shutil.move(file_path, os.path.join(genuine_folder, file))
                    elif file.lower().startswith("forged"):
                        shutil.move(file_path, os.path.join(forged_folder, file))

# Organize each dataset
for dataset in datasets:
    organize_files(dataset)

print("Files have been organized into genuine and forged folders!")
