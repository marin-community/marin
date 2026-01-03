# Submitting a pull request

Suppose you have already checked out the Marin repository and made some changes to it (e.g., for the [Marin speedrun](submitting-speedrun.md)).
Now you want to submit your changes to the Marin repository.

## Prerequisites

We assume you have already followed the [installation guide](installation.md) to set up Marin.

## Develop your code

Assume that you have made your code changes in a branch called `<my-branch>`.

## Submitting a pull request

If you are a Marin maintainer, you have write access to the Marin repository and
can directly push your changes to a branch and open a pull request from that
branch to the main branch.

Otherwise, you will have to fork the Marin repository.

Go to [https://github.com/marin-community/marin/fork](https://github.com/marin-community/marin/fork) and fork the repository.
Assume the forked repository is at `https://github.com/<your-username>/marin`.

Add the original Marin repository as an upstream remote and your fork as an origin remote:

```bash
git remote remove origin
git remote add upstream https://github.com/marin-community/marin.git
git remote add origin https://github.com/<your-username>/marin.git
git remote -v  # Check that the remotes are set up correctly
```

Then pull changes from the upstream repository to your local repository and push your changes to your fork:

```bash
git pull upstream main
git push origin <my-branch>
```

Finally, go to
[https://github.com/<your-username>/marin/pulls](https://github.com/<your-username>/marin/pulls)
and click on the "New pull request" button to make the actual pull request.
