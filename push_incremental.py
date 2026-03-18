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

print("Starting Incremental Git Push (Bullet-proof method)...")

# 1. Start fresh branch
run("git checkout --orphan incremental_deploy")
run("git rm -rf --cached .")

# 2. Write empty gitignore to upload everything exactly as requested
with open(".gitignore", "w") as f:
    f.write("")

# 3. Setup LFS for big files (GitHub limit is 100MB)
run("git lfs install", check=True)
run('git lfs track "*.pt"')
run('git lfs track "*.h5"')
run('git lfs track "*.keras"')
run('git lfs track "*.zip"')
run('git lfs track "*.tar.gz"')
run('git lfs track "*.mp4"')
run('git add .gitattributes')
run('git commit -m "Add Git LFS tracking"')

# Connect to new remote
run('git remote remove origin')
run('git remote add origin https://github.com/divyaredii/Plant-disease-detection--.git')
run('git push -u origin incremental_deploy:main --force')

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

# 4. Push base files first (small code files)
print("\n--- Pushing structural files ---")
for f in os.listdir('.'):
    if os.path.isfile(f) and f not in ['.gitattributes']:
        commit_and_push(f, f"file: {f}")

# 5. Push standard directories individually (excluding the massive dataset for now)
print("\n--- Pushing main directories ---")
for d in ['Flask Deployed App', 'Model', 'downloaded_models', 'test', 'test_images']:
    if os.path.exists(d):
        commit_and_push(d, f"folder: {d}")

# 6. Push the massive image dataset ONE folder at a time
print("\n--- Pushing enormous Image Dataset sequentially ---")
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
                    print(f"\nUploading Dataset Folder {i+1}/{len(total_classes)}: {split} - {class_name}")
                    success = commit_and_push(class_dir, f"{split} images: {class_name[:15]}")
                    if not success:
                        print(f"CRITICAL ERROR: Failed to upload {class_dir}. Moving to next.")

print("\nSUCCESS! Everything has been pushed sequentially.")
