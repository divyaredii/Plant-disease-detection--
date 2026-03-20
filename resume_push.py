import os
import subprocess
import time

def run(cmd, check=False):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"FAILED: {cmd}")
        exit(1)
    return result

print("Resuming Incremental Git Push...")

def commit_and_push(path, msg):
    run(f'git add "{path}"')
    # If nothing added, return
    if run("git diff-index --quiet HEAD").returncode == 0:
        return True
    
    run(f'git commit -m "Add {msg}"')
    
    max_retries = 3
    for attempt in range(max_retries):
        if run('git push origin incremental_deploy:main').returncode == 0:
            return True
        print(f"Push failed, waiting 5 seconds and retrying... ({attempt+1}/{max_retries})")
        time.sleep(5)
    return False

# 1. Push standard directories individually (skip if already pushed because git knows)
print("\n--- Resuming push of main directories ---")
for d in ['Flask Deployed App', 'Model', 'downloaded_models', 'test', 'test_images']:
    if os.path.exists(d):
        commit_and_push(d, f"folder: {d}")

# 2. Push the massive image dataset ONE folder at a time, picking up where it left off!
print("\n--- Resuming push of Image Dataset sequentially ---")
dataset_path = "New Plant Diseases Dataset(Augmented)"

if os.path.exists(dataset_path):
    for split in ["train", "valid"]:
        split_path = os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)", split)
        if not os.path.exists(split_path):
            split_path = os.path.join(dataset_path, split)
        
        if os.path.exists(split_path):
            total_classes = os.listdir(split_path)
            for i, class_name in enumerate(total_classes):
                class_dir = os.path.join(split_path, class_name)
                if os.path.isdir(class_dir):
                    print(f"\nChecking Dataset Folder {i+1}/{len(total_classes)}: {split} - {class_name}")
                    success = commit_and_push(class_dir, f"{split} images: {class_name[:15]}")

print("\nSUCCESS! Resume completed and all files pushed safely.")
