@docs/dev-guide/guidelines-internal.md
@AGENTS.md

<!-- COMMS-MONITOR-START -->
## Inter-Agent Communications

You are session `claude-marin-inference`. You have comms MCP tools available (mcp__comms__*).

### On Session Start
- IMMEDIATELY call `mcp__comms__read_messages` to check for pending messages.
- Process any messages before doing anything else.

### After Every Task
- Call `mcp__comms__read_messages` after completing each user request.

### Idle Monitoring Loop (CRITICAL)
When you have NO active user task (you've finished responding and are waiting
for input), you MUST enter a monitoring loop:

1. Sleep 30 seconds: `sleep 30` via Bash
2. Call `mcp__comms__check_unread_count` (lightweight peek)
3. If count > 0: call `mcp__comms__read_messages` and process them
4. REPEAT from step 1 — do NOT stop

This loop ensures you receive messages from other agents even when idle.
Continue until the user sends a new message (which interrupts you naturally).

### Message Handling
- For quick questions/responses: reply inline via `mcp__comms__send_message`
- For substantial tasks: acknowledge receipt immediately ("Working on it..."),
  then spawn a background Task, continue monitoring, and send results when done
- Always identify yourself by session name in responses

### Priority
User input > agent messages. If the user sends something, handle it first,
then resume monitoring.
<!-- COMMS-MONITOR-END -->