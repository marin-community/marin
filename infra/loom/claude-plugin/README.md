# Loom plugin for Claude

This plugin lets Claude hand code changes in `marin-community` repositories to
Loom sessions on `loom.oa.dev` and follow them from a chat. It contains:

- `.mcp.json`: the Loom connector at `https://loom.oa.dev/mcp`.
- `skills/use-loom/`: when to hand work to Loom, how to write the session goal,
  and which connector tool answers each follow-up.
- `commands/loom.md`: the `/loom <request>` command, which hands the rest of the
  message to Loom. In claude.ai chat it loads as a skill.

## Prerequisites

The `connector.enabled` setting must be on in the [Loom deployment](../README.md)
for `loom.oa.dev`. A Loom administrator registers claude.ai as the connector's
OAuth client once, which prints a client ID and secret:

```sh
loom oauth-client add claude.ai \
  --redirect-uri https://claude.ai/api/mcp/auth_callback \
  --redirect-uri https://claude.com/api/mcp/auth_callback
```

Making the plugin available does not add its bundled connector. An
organization Owner adds a custom connector in **Organization settings >
Connectors** with the URL `https://loom.oa.dev/mcp`, selects **Use your own
OAuth client** under **Advanced settings**, and enters the client ID and
secret. Each member then connects it and signs in to Loom with GitHub.

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

To try the plugin before publishing it, upload the zip to your own account
from **Customize > Plugins > Add > Upload plugin**.

## Claude Code

Loom has no dynamic client registration, so Claude Code needs a client ID
before it can sign in. A Loom administrator registers Claude Code once as a
public client. Loom accepts its loopback callback on any port:

```sh
loom oauth-client add claude-code --public --redirect-uri http://localhost/callback
```

Each person then adds the server with that client ID and copies the skill:

```sh
claude mcp add --transport http --scope user --client-id <client-id> loom https://loom.oa.dev/mcp
cp -r infra/loom/claude-plugin/skills/use-loom ~/.claude/skills/
```
