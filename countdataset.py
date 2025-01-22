import os

# Dataset Configuration
datasets = {
    "BHSig260_Bengali": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
    "BHSig260_Hindi": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
    "CEDAR": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR"
}

# Function to count files in subdirectories
def count_signatures(writer_path):
    forged_path = os.path.join(writer_path, "forged")
    genuine_path = os.path.join(writer_path, "genuine")
    forged_count = len(os.listdir(forged_path)) if os.path.exists(forged_path) else 0
    genuine_count = len(os.listdir(genuine_path)) if os.path.exists(genuine_path) else 0
    return forged_count, genuine_count

# Traverse datasets and count signatures
for dataset_name, dataset_path in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    total_forged = 0
    total_genuine = 0

    for writer in os.listdir(dataset_path):
        writer_path = os.path.join(dataset_path, writer)
        if os.path.isdir(writer_path):
            forged_count, genuine_count = count_signatures(writer_path)
            total_forged += forged_count
            total_genuine += genuine_count
            print(f"Writer {writer}: Forged = {forged_count}, Genuine = {genuine_count}")

    print(f"\nTotal in {dataset_name}: Forged = {total_forged}, Genuine = {total_genuine}")
