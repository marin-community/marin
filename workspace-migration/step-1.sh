#!/usr/bin/env bash
# Workspace Migration - Step 1
#
# Initializes uv workspace and moves marin/data_browser into lib/
# Run from repo root: ./workspace-migration/step-1.sh
#
# This script can be replayed on different git states to apply the restructuring.

set -e

# Change to repo root (parent of workspace-migration/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Reference branch containing the workspace structure changes
# Can be overridden via argument or falls back to rw/ws on a marin-community/marin remote
REFERENCE_BRANCH="${1:-}"
if [ -z "$REFERENCE_BRANCH" ]; then
    # Find a remote pointing to marin-community/marin
    REMOTE=$(git remote -v | grep -E 'github\.com[:/]marin-community/marin' | head -1 | awk '{print $1}')
    if [ -n "$REMOTE" ]; then
        REFERENCE_BRANCH="$REMOTE/rw/ws"
    else
        echo "ERROR: No reference branch specified and no remote found pointing to marin-community/marin"
        echo "Please either:"
        echo "  1. Pass a reference branch as an argument: ./workspace-migration/step-1.sh <branch>"
        echo "  2. Add a remote pointing to https://github.com/marin-community/marin"
        exit 1
    fi
fi

echo "Workspace Migration - Step 1"
echo "Initializing uv workspace: marin + data_browser → lib/"
echo "Reference branch: $REFERENCE_BRANCH"
echo ""

# Check if lib/ already exists
if [ -d "lib" ]; then
    echo "ERROR: lib/ directory already exists. Migration may have already been applied."
    exit 1
fi

# Update .gitignore to allow lib/ and CLAUDE.md
echo "Updating .gitignore..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '/^lib\/$/d; /^CLAUDE\.md$/d' .gitignore
else
    sed -i '/^lib\/$/d; /^CLAUDE\.md$/d' .gitignore
fi

# Create lib directory structure
echo "Creating lib/ directory structure..."
mkdir -p lib/marin

# Move packages to lib/
echo "Moving src/ to lib/marin/src/..."
git mv src lib/marin/

echo "Moving marin/tools/ to lib/marin/tools/..."
git mv marin/tools lib/marin/

# Remove now-empty marin/ directory
rmdir marin

echo "Moving data_browser/ to lib/data_browser/..."
git mv data_browser lib/

# Save original root pyproject.toml (for extracting content, will be deleted)
echo "Saving original pyproject.toml..."
mkdir -p tmp
cp pyproject.toml tmp/pyproject.toml.migration-backup

# Create new workspace root pyproject.toml from reference branch
echo "Creating workspace root pyproject.toml..."
git show "$REFERENCE_BRANCH:pyproject.toml" > pyproject.toml || {
    echo "ERROR: Could not extract pyproject.toml from $REFERENCE_BRANCH"
    exit 1
}

# Create lib/marin/pyproject.toml by extracting from reference branch
echo "Creating lib/marin/pyproject.toml..."
if git cat-file -e "$REFERENCE_BRANCH:lib/marin/pyproject.toml" 2>/dev/null; then
    git show "$REFERENCE_BRANCH:lib/marin/pyproject.toml" > lib/marin/pyproject.toml
else
    echo "WARNING: Could not extract lib/marin/pyproject.toml from $REFERENCE_BRANCH"
    echo "Creating from template..."
    # Fallback: extract from backed up root and modify
    sed -e '1,/\[project\]/{ /^requires-python/d; }' \
        -e 's/packages = \["src\/marin", "experiments"\]/packages = ["src\/marin"]/' \
        -e 's|license = { file = "LICENSE" }|license = { file = "../../LICENSE" }|' \
        tmp/pyproject.toml.migration-backup > lib/marin/pyproject.toml
fi

# Create CLAUDE.md from reference branch
echo "Creating CLAUDE.md..."
if git cat-file -e "$REFERENCE_BRANCH:CLAUDE.md" 2>/dev/null; then
    git show "$REFERENCE_BRANCH:CLAUDE.md" > CLAUDE.md
else
    echo "WARNING: Could not extract CLAUDE.md from $REFERENCE_BRANCH"
fi

# lib/data_browser/pyproject.toml should already have correct requires-python
# No changes needed

# Update paths in critical files for workspace structure
echo "Updating file paths for workspace structure..."

# Use MARIN_DOC_BRANCH env var for GitHub doc URLs, default to main
DOC_BRANCH="${MARIN_DOC_BRANCH:-main}"
# URL-encode slashes
DOC_BRANCH_ENCODED=$(echo "$DOC_BRANCH" | sed 's|/|%2F|g')

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's|src/marin/cluster/config\.py|lib/marin/src/marin/cluster/config.py|g' Makefile
    sed -i '' 's|src/marin/cluster/config\.py|lib/marin/src/marin/cluster/config.py|g' .github/workflows/build-docker-images.yaml
    sed -i '' 's|paths: \["\.", "src"\]|paths: [".", "lib/marin/src"]|' mkdocs.yml
    sed -i '' 's|src/marin/speedrun/|lib/marin/src/marin/speedrun/|g' .github/workflows/update-leaderboard.yml
    sed -i '' "s|/blob/main/src/marin/|/blob/$DOC_BRANCH_ENCODED/lib/marin/src/marin/|g" docs/tutorials/submitting-speedrun.md docs/tutorials/datashop.md docs/explanations/executor.md docs/explanations/evaluation.md docs/reports/markdownified-datasets.md docs/explanations/speedrun-flops-accounting.md
else
    sed -i 's|src/marin/cluster/config\.py|lib/marin/src/marin/cluster/config.py|g' Makefile
    sed -i 's|src/marin/cluster/config\.py|lib/marin/src/marin/cluster/config.py|g' .github/workflows/build-docker-images.yaml
    sed -i 's|paths: \["\.", "src"\]|paths: [".", "lib/marin/src"]|' mkdocs.yml
    sed -i 's|src/marin/speedrun/|lib/marin/src/marin/speedrun/|g' .github/workflows/update-leaderboard.yml
    sed -i "s|/blob/main/src/marin/|/blob/$DOC_BRANCH_ENCODED/lib/marin/src/marin/|g" docs/tutorials/submitting-speedrun.md docs/tutorials/datashop.md docs/explanations/executor.md docs/explanations/evaluation.md docs/reports/markdownified-datasets.md docs/explanations/speedrun-flops-accounting.md
fi

# Copy uv.lock from reference branch
echo "Copying uv.lock from reference branch..."
git show "$REFERENCE_BRANCH:uv.lock" > uv.lock || {
    echo "WARNING: Could not extract uv.lock from $REFERENCE_BRANCH, generating new one..."
    rm -f uv.lock
    echo 'Running `RUST_LOG=warn uv sync` to generate lockfile...'
    RUST_LOG=warn uv sync
}

# Stage all changes (git mv and git rm --cached already staged, add new files)
echo "Staging changes..."
git add .gitignore pyproject.toml CLAUDE.md lib/ Makefile mkdocs.yml .github/workflows/build-docker-images.yaml .github/workflows/update-leaderboard.yml docs/ uv.lock

# Clean up temp file
rm -f tmp/pyproject.toml.migration-backup

echo ""
echo "Migration complete! Next steps:"
echo "1. Review the changes with: git status"
echo "2. Commit the changes: git commit -m 'Your message'"
echo "3. Test imports: uv run python -c 'import marin; print(\"Success\")'"
echo "4. Update CI/docs as needed"
