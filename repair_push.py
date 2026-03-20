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

print("Starting Push Repair Script (Splitting many commits)...")

# Get list of commits on local branch that aren't on remote
# Note: Reversed to push oldest first
res = run('git log origin/main..incremental_deploy --oneline --reverse', True)
stdout = subprocess.check_output('git log origin/main..incremental_deploy --oneline --reverse', shell=True).decode()
commits = [line.split(' ')[0] for line in stdout.strip().split('\n') if line.strip()]

if not commits:
    print("No commits to push! Everything is synchronised.")
    exit(0)

print(f"Found {len(commits)} commits to push one by one...")

for i, commit_hash in enumerate(commits):
    print(f"\n--- Pushing Commit {i+1}/{len(commits)}: {commit_hash} ---")
    
    max_retries = 3
    for attempt in range(max_retries):
        # We push the specific commit hash to the main branch
        res = run(f'git push origin {commit_hash}:main')
        if res.returncode == 0:
            print(f"✓ Commit {commit_hash} pushed successfully!")
            break
        else:
            print(f"Push failed! Retrying in 5 seconds... ({attempt+1}/{max_retries})")
            time.sleep(5)
    else:
        print(f"FAILED to push commit {commit_hash} after multiple attempts.")
        # If one fails, we might still be able to push the next ones if it's a transient error
        # but usually it's better to pause and see why

print("\nRepair script complete!")
