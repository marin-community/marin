# Loom plugin for Claude

This plugin lets Claude hand code changes in `marin-community` repositories to
Loom sessions on `loom.oa.dev` and follow them from a chat. It contains:

- `.mcp.json`: the Loom connector at `https://loom.oa.dev/mcp`.
- `skills/use-loom/`: when to hand work to Loom, how to write the session goal,
  and which connector tool answers each follow-up.
- `commands/loom.md`: the `/loom <request>` command, which hands the rest of the
  message to Loom. In claude.ai chat it loads as a skill.

## Prerequisites

Loom must serve the connector, and claude.ai must be registered as its OAuth
client. [The Loom deployment README](../README.md#claudeai-connector) covers
both steps.

Making the plugin available does not add its bundled connector. An
organization Owner adds a custom connector in **Organization settings >
Connectors** with the URL `https://loom.oa.dev/mcp` and enters the OAuth client
ID and secret under **Advanced settings**. Each member then connects it and
signs in to Loom with GitHub.

## Publish to the organization

claude.ai organization sync reads only private or internal repositories, and
this repository is public. Publish the plugin in one of two ways:

- Zip this directory and upload it from **Organization settings > Plugins &
  skills > Add > Upload a plugin**. Upload a new version after each change.
- List this directory as a `git-subdir` plugin source in a private marketplace
  repository owned by the organization, and add that repository with **Sync
  from GitHub**.

```sh
cd infra/loom && zip -r loom-plugin.zip claude-plugin
```

Then set the plugin's **Default access**, for example to **Installed by
default**. Raise `version` in `.claude-plugin/plugin.json` with each release.

To try the plugin before publishing it, run
`claude --plugin-dir infra/loom/claude-plugin` in Claude Code, or upload the
zip to your own account from **Customize > Plugins > Add > Upload plugin**.
