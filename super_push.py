import os
import subprocess
import time

def run(cmd, check=False):
    # print(f"\n> {cmd}") 
    result = subprocess.run(cmd, capture_output=True, shell=True)
    if check and result.returncode != 0:
        print(f"FAILED: {cmd}\nError: {result.stderr.decode()}")
        exit(1)
    return result

print("Starting Super-Granular Push (Splitting images into small ~25MB batches)...")

# Setup safe config
run('git config http.postBuffer 2147483648')
run('git config http.lowSpeedLimit 0')
run('git config http.lowSpeedTime 999999')

dataset_path = "New Plant Diseases Dataset(Augmented)"

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit(1)

# List all image files recursively
all_images = []
for root, dirs, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_images.append(os.path.join(root, f))

print(f"Found {len(all_images)} total images to scan.")

# Use chunks of 40 images per push (~20 MB per push)
chunk_size = 40
count = 0 
for i in range(0, len(all_images), chunk_size):
    chunk = all_images[i:i + chunk_size]
    
    # Check if any in chunk need adding
    for img in chunk:
        rel_path = os.path.relpath(img, os.getcwd())
        run(f'git add "{rel_path}"')
    
    # See if we have anything to commit
    if run("git diff-index --quiet HEAD").returncode != 0:
        run(f'git commit -m "Incremental push batch {i//chunk_size}"')
        
        max_retries = 3
        for attempt in range(max_retries):
            res = run('git push origin incremental_deploy:main')
            if res.returncode == 0:
                count += len(chunk)
                if (i // chunk_size) % 5 == 0:
                    print(f"✓ Pushed {count} new images... ({i}/{len(all_images)})")
                break
            else:
                print(f"Push failed (batch {i//chunk_size}). Retry {attempt+1}/{max_retries}...")
                time.sleep(5)
        else:
            print(f"CRITICAL: Failed to push batch {i//chunk_size}. Skipping folder chunk...")
            # We skip to the next batch rather than stopping entirely
            run('git reset --soft HEAD~1')
            run('git reset HEAD .')
            continue
    else:
        pass # Already pushed

print("\nSUCCESS! Script finished scanning all files.")
