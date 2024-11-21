import os

dataset_path = "/Users/christelle/Downloads/Thesis/Dataset"

for writer in sorted(os.listdir(dataset_path)):
    writer_path = os.path.join(dataset_path, writer)
    genuine_path = os.path.join(writer_path, "genuine")
    forged_path = os.path.join(writer_path, "forged")
    
    if not os.path.isdir(genuine_path) or not os.path.isdir(forged_path):
        print(f"ERROR: Missing directory in {writer}")
        continue
    
    genuine_files = os.listdir(genuine_path)
    forged_files = os.listdir(forged_path)
    
    print(f"{writer}: {len(genuine_files)} genuine, {len(forged_files)} forged")
