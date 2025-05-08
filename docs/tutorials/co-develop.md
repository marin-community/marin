# Co-develop Marin and Levanter (as a Submodule)

If you need to co-develop [Levanter](https://github.com/stanford-crfm/levanter) and marin, you can create levanter as a submodule within marin following these steps:
```
# Run from ROOT_DIR
git clone https://github.com/stanford-crfm/levanter
git clone git@github.com:stanford-crfm/marin.git
mkdir -p marin/submodules
cd levanter
# Create a new working tree for the LEVANTER_BRANCH branch of the levanter repository,
# located at ROOT_DIR/marin/submodules/levanter. This means ROOT_DIR/marin/submodules/levanter
# is now a working directory specifically for the LEVANTER_BRANCH branch of levanter.
git worktree add ROOT_DIR/marin/submodules/levanter LEVANTER_BRANCH
cd ..
```
Change `LEVANTER_BRANCH` to the branch name in `levanter` you are developing. Now, you can make changes to the `LEVANTER_BRANCH` of levanter by making edits in `ROOT_DIR/marin/submodules/levanter`. After making changes, you can run
```
cd marin/submodules/levanter
git add .
git commit -m "Update LEVANTER_BRANCH branch with new changes"
```
This will commit the change to the `LEVANTER_BRANCH` branch of levanter.
