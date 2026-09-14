# Marin Slack sessions

Treat the Slack request and surrounding thread as the source conversation.
Follow the target repository's `AGENTS.md` and applicable skills.

- Answer a simple question directly in Slack without creating a repository
  change. Carry an explicit implementation or fix request through the
  repository's pull-request workflow.
- Keep the final Slack reply concise and self-contained. Include links to any
  pull request, issue, or durable artifact created for the request.
- Announce actions such as opening a pull request or issue, or starting a run
  or deployment, in the Slack thread before entering a wait or long-running
  monitor. Include the action's identifier and link when available. A status
  update does not replace this thread message.
- Add the `agent-generated` label to every pull request or issue you create.
- Use the repository's commit/landing skill for changes and remain with CI for
  as long as that workflow requires.
