#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration (adjust these paths if necessary) ---
MARIN_REPO_DIR="../../"
PAGES_REPO_DIR="../speedrun" # Your local clone of marin-community/speedrun
COMMIT_MESSAGE="Automated leaderboard update: $(date +'%Y-%m-%d %H:%M:%S %Z')"
PYTHON_EXECUTABLE="python" # Or the specific path to python in your conda env if needed

echo "=== Starting Speedrun Leaderboard Deployment ==="

# 1. Generate latest leaderboard data from the Marin repository
echo "[1/5] Navigating to Marin repository: $MARIN_REPO_DIR"
cd "$MARIN_REPO_DIR"
echo "[1/5] Generating leaderboard data..."
$PYTHON_EXECUTABLE marin/speedrun/create_leaderboard.py
echo "Leaderboard data generated."

# 2. Navigate to the GitHub Pages repository
echo "[2/5] Navigating to GitHub Pages repository: $PAGES_REPO_DIR"
cd "$PAGES_REPO_DIR"

# 3. Update local Pages repo from remote to avoid conflicts
echo "[3/5] Pulling latest changes from remote for Pages repository (git pull origin main)..."
git pull origin main
echo "Local Pages repository updated."

# 4. Copy updated static files from Marin repo to Pages repo
SOURCE_STATIC_DIR="$MARIN_REPO_DIR/marin/speedrun/static"
echo "[4/5] Copying index.html from $SOURCE_STATIC_DIR to $PAGES_REPO_DIR/"
cp "$SOURCE_STATIC_DIR/index.html" "$PAGES_REPO_DIR/"

echo "[4/5] Removing old data directory: $PAGES_REPO_DIR/data/"
rm -rf "$PAGES_REPO_DIR/data" # Remove old directory to ensure clean copy
echo "[4/5] Copying data directory from $SOURCE_STATIC_DIR/data to $PAGES_REPO_DIR/"
cp -R "$SOURCE_STATIC_DIR/data" "$PAGES_REPO_DIR/"
echo "Static files copied."

# 5. Commit and push changes to the GitHub Pages repository
echo "[5/5] Preparing to commit and push changes..."
git add .

# Check if there are changes to commit
if git diff-index --quiet HEAD --; then
    echo "No changes to deploy. Leaderboard is already up-to-date."
else
    echo "Committing changes with message: '$COMMIT_MESSAGE'"
    git commit -m "$COMMIT_MESSAGE"
    echo "Pushing changes to origin main..."
    git push origin main
    echo "Leaderboard deployed successfully to https://marin.community/speedrun/"
fi

echo "=== Speedrun Leaderboard Deployment Finished ==="
