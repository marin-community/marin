const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const [extensionPath, servers, exclusions] = process.argv.slice(2);
fs.accessSync(extensionPath, fs.constants.R_OK);
const excluded = JSON.parse(exclusions);
const directory = path.join(os.homedir(), ".pi/agent");
fs.mkdirSync(directory, { recursive: true });
const settingsPath = path.join(directory, "settings.json");
const settings = fs.existsSync(settingsPath)
  ? JSON.parse(fs.readFileSync(settingsPath, "utf8"))
  : {};
const wrapperPath = path.join(directory, "harbor-mcp-filter.js");
if (excluded.length) {
  // Intercept rediscovery too: the Tau3 relay owns conversation control.
  fs.writeFileSync(wrapperPath, `
import mcp from ${JSON.stringify(extensionPath)};
const excluded = new Set(${JSON.stringify(excluded)});
export default async function (pi) {
  await mcp({
    ...pi,
    registerTool(tool) {
      if (!excluded.has(tool.name)) pi.registerTool(tool);
    },
    setActiveTools(names) {
      pi.setActiveTools(names.filter(name => !excluded.has(name)));
    },
  });
}
`);
} else {
  fs.rmSync(wrapperPath, { force: true });
}
settings.extensions = [
  ...(settings.extensions ?? []).filter(entry => entry !== extensionPath && entry !== wrapperPath),
  excluded.length ? wrapperPath : extensionPath,
];
fs.writeFileSync(settingsPath, JSON.stringify(settings));
fs.writeFileSync(path.join(directory, "mcp.json"), servers);
