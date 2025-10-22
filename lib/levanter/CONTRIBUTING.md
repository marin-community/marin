Contributing
============

Levanter is a growing code base, and we are excited for other folks to get involved. The instructions below walk you through our dev setup and how to submit a PR.

Dev Installation
----------------

First follow the same instructions as provided for the [Levanter README](README.md) to install Levanter.

The main addition for a dev environment is to install [`pre-commit`](https://pre-commit.com/):

    pre-commit install

This will set up git hook scripts that ensure your code is formatted in a manner consistent with
the repo. If any problems are found, the appropriate files will be updated with fixes. You will
need to review and commit the fixed files.

Forking The Repo
----------------

To submit changes, you will need to work off of a fork of the repo and issue a pull request.

There are two easy ways to fork the repo.

If you have installed the [GitHub CLI](https://cli.github.com/) you can issue this command:

    gh repo fork stanford-crfm/levanter --clone=true

This will create the fork and clone the repo into your current directory.

Alternatively you can fork the repo in your browser. While logged in to your GitHub account,
go to the [Levanter repo](https://github.com/stanford-crfm/levanter) and click on the Fork
button in the upper left hand corner.

You can then clone your forked version of the Levanter repo like any other GitHub repo.

Create A Branch For Your Submission
-----------------------------------

You will generally need to create a branch of `main` for your code changes.  In general every submission
should be focused on a specific set of bug fixes or new features that are coherently
related.  Changes that are not related belong in different submissions. So you should
be able to give your branch an informative name such as `checkpointer-time-bugfix` .

You can create a branch off of `main` with this command:

    git checkout -b checkpointer-time-bugfix main

Implement Your Changes
----------------------

As you implement your changes in your feature branch, the git hook scripts will check your
code for proper formatting as you make commits. Make sure you have run `pre-commit install`
before you start making commits.

You can also check all files in the current branch with this command:

    pre-commit run --all-files

When your changes are operational you should verify that the current tests are passing.

Set up your environment for running the tests:

    export PYTHONPATH=/path/to/levanter/src:path/to/levanter/tests:$PYTHONPATH
    wandb offline

You can run the tests with this command:

    pytest tests

You should add tests for any functionality you have added consistent with the [pytest](https://docs.pytest.org/en/6.2.x/) format
of the existing tests.

Submit Pull Request
-------------------

When your feature branch is ready you should submit a pull request.

Detailed instructions for submtting a pull request from a fork can be found on [Github Docs](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

The steps basically are:

1. While logged in to your GitHub, go to the original [Levanter repo pull request page](https://github.com/stanford-crfm/levanter/pulls)
2. Click on the highlighted text stating "compare across forks".
3. Set the base repository to `stanford-crfm/levanter` and the base branch to `main`.
4. Set the head repository to `your-org/levanter` and the compare branch to `your-feature-branch`.
5. Click on the "Create pull request" button and complete the pull request form.

When submitting your pull request, you should provide a detailed description of what you've done.

The following is a useful template:

    ## Description
    A brief and concise description of what your pull request is trying to accomplish.

    ## Fixes Issues
    A list of issues/bugs with # references. (e.g., #123)

    ## Unit test coverage
    Are there unit tests in place to make sure your code is functioning correctly?

    ## Known breaking changes/behaviors
    Does this break anything in Levanter's existing user interface? If so, what is it and how is it addressed?
