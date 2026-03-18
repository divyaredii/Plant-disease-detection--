import os
import subprocess
import time

def run_cmd(cmd, check=False):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"FAILED: {cmd}")
        exit(1)
    return result.returncode

print("Setting up Git LFS to bypass GitHub's 100MB file limits and timeout issues...")

# Create empty gitignore because the user wants everything uploaded
with open(".gitignore", "w") as f:
    f.write("")

# Initialize LFS and track large files so GitHub doesn't reject them
run_cmd("git lfs install", check=True)
run_cmd('git lfs track "*.pt"')
run_cmd('git lfs track "*.h5"')
run_cmd('git lfs track "*.hdf5"')
run_cmd('git lfs track "*.keras"')
run_cmd('git lfs track "*.zip"')
run_cmd('git lfs track "*.tar.gz"')
run_cmd('git lfs track "*.mp4"')
run_cmd('git lfs track "*.part"')
run_cmd('git add .gitattributes')
run_cmd('git commit -m "Add Git LFS tracking for ultra-large files"')

# Increase postBuffer to prevent HTTP 408 (Timeout) or 413 (Payload Too Large) during git push
run_cmd('git config http.postBuffer 2147483648')

# Now we will add directories sequentially to avoid hitting limits again
directories_to_add = [
    # Top-level code
    '*.py', '*.md', '*.txt', '*.csv', '*.json', '*.html', '*.css', '*.png', '*.jpg', '*.JPG',
    
    # Virtual environment (User requested this)
    'venv', 'env', '__pycache__', '.idea', '.vscode',
    
    # App code and big models (Will go through LFS now)
    '"Flask Deployed App"',
    'downloaded_models',
]

for d in directories_to_add:
    print(f"\n-- Adding {d}...")
    run_cmd(f'git add {d}')
    run_cmd(f'git commit -m "Add {d}"')

# Push the base codebase + environments + big models first
print("\n-- Pushing codebase, environment, and models to remote...")
max_retries = 3
for attempt in range(max_retries):
    res = run_cmd('git push origin deploy_branch:main')
    if res == 0:
        break
    else:
        print(f"Push failed. Retrying... ({attempt + 1}/{max_retries})")
        time.sleep(5)
else:
    print("FAILED to push codebase after multiple attempts.")
    exit(1)

# Now iterate over the extremely huge dataset classes and push them individually
dataset_path = "New Plant Diseases Dataset(Augmented)"

if os.path.exists(dataset_path):
    print(f"\nNow processing dataset at: {dataset_path}")
    
    dirs_to_push = []
    for split in ["train", "valid"]:
        split_path = os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)", split)
        if not os.path.exists(split_path):
            split_path = os.path.join(dataset_path, split) # fallback
            
        if os.path.exists(split_path):
            for class_name in os.listdir(split_path):
                class_dir = os.path.join(split_path, class_name)
                if os.path.isdir(class_dir):
                    dirs_to_push.append((split, class_name, class_dir))
    
    total_dirs = len(dirs_to_push)
    print(f"Found {total_dirs} directories to push sequentially.")
    
    count = 0
    batch_size = 2 # Lowered batch size even more to prevent limits
    
    for i in range(0, total_dirs, batch_size):
        batch = dirs_to_push[i:i+batch_size]
        for split, class_name, class_dir in batch:
            print(f"-- Adding {split} / {class_name}...")
            run_cmd(f'git add "{class_dir}"')
        
        batch_names = ", ".join([b[1][:10]+"..." for b in batch])
        run_cmd(f'git commit -m "Add dataset batch: {batch_names}"')
        
        print(f"-- Pushing batch {i//batch_size + 1} of {total_dirs//batch_size + 1}...")
        for attempt in range(max_retries):
            res = run_cmd('git push origin deploy_branch:main')
            if res == 0:
                break
            else:
                print(f"Push failed! Retrying in 5 seconds... ({attempt + 1}/{max_retries})")
                time.sleep(5)
        
        count += len(batch)
        print(f"Progress: {count}/{total_dirs} directories uploaded.")

print("\nAll done! Everything pushed successfully.")
