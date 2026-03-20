import os
import subprocess
import time

def run(cmd):
    # print(f"\n> {cmd}")
    result = subprocess.run(cmd, capture_output=True, shell=True)
    return result

print("Starting Fast Super-Push (Optimized for 130KB/s upload)...")

# Setup safe config
run('git config http.postBuffer 2147483648')
run('git config http.lowSpeedLimit 0')
run('git config http.lowSpeedTime 999999')

# Get ONLY the untracked/modified files (saves scanning time!)
print("Scanning for missing files on your computer...")
stdout = run('git ls-files --others --exclude-standard').stdout.decode('utf-8', 'ignore')
missing_files = [line.strip() for line in stdout.strip().split('\n') if line.strip()]

if not missing_files:
    print("All files are already pushed! (Checking modified files next...)")
    stdout = run('git ls-files --modified').stdout.decode('utf-8', 'ignore')
    missing_files = [line.strip() for line in stdout.strip().split('\n') if line.strip()]

if not missing_files:
    print("Everything is completely up to date.")
    exit(0)

print(f"Detected {len(missing_files)} missing files.")

# Push in 150-image batches (~25-35MB) to prevent Timeouts
batch_size = 150
total_batches = (len(missing_files) + batch_size - 1) // batch_size

for i in range(0, len(missing_files), batch_size):
    current_batch_num = (i // batch_size) + 1
    chunk = missing_files[i:i + batch_size]
    
    print(f"\n--- Processing Batch {current_batch_num}/{total_batches} ---")
    
    # Batch adding is MUCH faster
    for file_path in chunk:
        run(f'git add "{file_path}"')
    
    # Check if there's anything to commit
    if run("git diff-index --quiet HEAD").returncode != 0:
        run(f'git commit -m "Incremental push batch {current_batch_num}"')
        
        max_retries = 3
        for attempt in range(max_retries):
            # Try to push
            print(f"Uploading batch ({current_batch_num}/{total_batches})...")
            res = run('git push origin incremental_deploy:main')
            if res.returncode == 0:
                print(f"✓ Batch {current_batch_num} successfully uploaded!")
                break
            else:
                print(f"⚠ Upload failed! Retrying in 5 seconds... ({attempt+1}/{max_retries})")
                time.sleep(5)
        else:
            print(f"CRITICAL: Failed to push batch {current_batch_num}. Stopping.")
            exit(1)
    else:
        print(f"✓ Batch {current_batch_num} already synchronised.")

print("\nSUCCESS! Every missing file has been successfully uploaded to GitHub.")
