## Speedrun Dashboard Update

In order to update the speedrun dashboard, go to the Github Actions tab and run the manually triggered "Update Leaderboard" action. This will create a PR in `marin-community/speedrun` updating the relevant data files with the newest information available on Master. Merging this PR will update the dashboard with the most recent runs.

### "Bad Token" Error

Github Personal Access Tokens expire on a 90 day cycle. If the "Update Leaderboard" action is failing due to incorrect credentials update `secrets.SPEEDRUN_LEADERBOARD_PAT` with the Personal Access Token (classic) of a current Marin maintainer following these instructions: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic
