#!/usr/bin/env bash
# Test script to verify step-1.sh replay works correctly
# This tests that replaying the migration produces the same tree as the original

set -e

echo "Testing workspace migration replay..."
echo ""

# Get current branch (should be ws or similar with migration already applied)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Verify we're on a branch with the migration already applied
if [ ! -d "lib/marin" ]; then
    echo "ERROR: Migration not yet applied on current branch"
    echo "Please run this test from a branch where step-1.sh has already been applied"
    exit 1
fi

# Create ephemeral test branch from parent commit
PARENT_COMMIT=$(git rev-parse "$CURRENT_BRANCH^")
TEST_BRANCH="ws-test-$(date +%s)"
echo "Creating test branch $TEST_BRANCH from $PARENT_COMMIT..."
git checkout -b "$TEST_BRANCH" "$PARENT_COMMIT"

# Remove migration artifacts that would conflict
echo "Removing migration artifacts if present..."
rm -rf lib/

# Copy workspace-migration/ from the original branch (untracked)
echo "Copying workspace-migration/ from $CURRENT_BRANCH..."
git checkout "$CURRENT_BRANCH" -- workspace-migration/
git restore --staged workspace-migration/

# Run migration script with current branch as reference
echo "Running migration script..."
./workspace-migration/step-1.sh "$CURRENT_BRANCH"

# Stage workspace-migration/ and commit
echo "Staging workspace-migration/ and committing..."
git add workspace-migration/
git commit -m '`uv`-workspace-ification, pt 1: marin, data_browser'

# Compare trees
echo ""
echo "Comparing trees..."
TREE0=$(git rev-parse "$CURRENT_BRANCH^{tree}")
TREE1=$(git rev-parse HEAD^{tree})

if [ "$TREE0" = "$TREE1" ]; then
    echo "✓ SUCCESS: Worktrees match!"
    echo "  Original ($CURRENT_BRANCH): $TREE0"
    echo "  Replayed ($TEST_BRANCH):    $TREE1"
    echo ""
    echo "Cleaning up: deleting test branch and returning to $CURRENT_BRANCH..."
    git checkout "$CURRENT_BRANCH"
    git branch -D "$TEST_BRANCH"
    exit 0
else
    echo "✗ FAILURE: Worktrees differ"
    echo "  Original ($CURRENT_BRANCH): $TREE0"
    echo "  Replayed ($TEST_BRANCH):    $TREE1"
    echo ""
    echo "Differences:"
    git diff-tree -r "$CURRENT_BRANCH" HEAD
    echo ""
    echo "Leaving you on $TEST_BRANCH for inspection."
    exit 1
fi
