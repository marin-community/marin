import { createHash } from "node:crypto";

// Recovery changes model context only. No malformed or synthetic call is executed.
const retryLimit = __PARSE_ERROR_RETRIES__;
const feedbackType = "harbor-hermes-parse-error";
const fatalMarker = "HARBOR_PI_PARSE_ERROR_PROJECTION_FAILURE";
const feedbackTool = "__parse_error_feedback__";
const feedback = "Tool call JSON could not be parsed. No tool was executed. Return a new tool call using valid JSON with name and arguments. Do not repeat the malformed text.";
const toolCalls = /<tool_call>([\s\S]*?)<\/tool_call>|<tool_call>([\s\S]*)/g;

export function hasInvalidToolJSON(message) {
  if (message.role !== "assistant" || message.content.some(block => block.type === "toolCall")) return false;
  const text = message.content.filter(block => block.type === "text").map(block => block.text).join("");
  for (const match of text.matchAll(toolCalls)) {
    try {
      JSON.parse(match[1] || match[2]);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      return true;
    }
  }
  return false;
}

export function parseErrorRetry(pi) {
  let retries = 0;
  pi.on("turn_end", event => {
    if (event.outcome !== "completed" || retries >= retryLimit || !hasInvalidToolJSON(event.message)) return;
    retries += 1;
    return {
      entries: [{
        type: "custom_message",
        customType: feedbackType,
        content: feedback,
        display: true,
        details: {
          retry: retries, sourceEntryId: event.messageEntryId,
          sourceResponseSha256: createHash("sha256").update(event.message.content.filter(block => block.type === "text").map(block => block.text).join("")).digest("hex"),
        },
      }],
      continue: true,
    };
  });
  pi.on("context", (event, ctx) => {
    const messages = [];
    for (const message of event.messages) {
      if (message.role !== "custom" || message.customType !== feedbackType) {
        messages.push(message);
        continue;
      }
      const previous = messages.at(-1);
      if (!previous || previous.role !== "assistant" || !hasInvalidToolJSON(previous) ||
          !Number.isInteger(message.details?.retry) || message.details.retry < 1 || message.details.retry > retryLimit ||
          message.details.sourceResponseSha256 !== createHash("sha256").update(previous.content.filter(block => block.type === "text").map(block => block.text).join("")).digest("hex")) {
        process.stderr.write(fatalMarker + "\n");
        ctx.abort();
        throw new Error("Parse-error feedback lost its original malformed response");
      }
      const id = `${feedbackType}-${message.details.retry}`;
      // This call exists only in the model-visible history, never in the execution queue.
      messages[messages.length - 1] = {
        ...previous,
        content: [...previous.content, { type: "toolCall", id, name: feedbackTool, arguments: {} }],
      };
      messages.push({
        role: "toolResult", toolCallId: id, toolName: feedbackTool,
        content: [{ type: "text", text: feedback }], isError: true, timestamp: message.timestamp,
      });
    }
    return { messages };
  });
}
