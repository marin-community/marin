# Bun 1.1 | Bun Blog



Bun is a fast, all-in-one toolkit for running, building, testing, and debugging JavaScript and TypeScript, from a single script to a full-stack application. If you're new to Bun, you can learn more in the [Bun 1.0](/blog/bun-v1.0#bun-is-an-all-in-one-toolkit "") blog post.

VIDEO

Bun 1.1 is huge update.

There's been over [1,700 commits](https://github.com/oven-sh/bun/compare/bun-v1.0.0...bun-v1.1.0 "") since Bun 1.0, and we've been working hard to make Bun more stable and more compatible with Node.js. We've fixed over a thousand bugs, added tons of new features and APIs, and now, Bun supports Windows!

You can now run Bun on Windows 10 and later! This is a huge milestone for us, and we're excited to bring Bun to a whole new group of developers.

Bun on Windows passes [98%](https://twitter.com/bunjavascript/status/1774036956474978507 "") of our own test suite for Bun on macOS and Linux. That means everything from the runtime, test runner, package manager, bundler — it all works on Windows.

To get started with Bun on Windows, run the following command in your terminal:

```
> powershell -c "irm bun.sh/install.ps1 | iex"

```

Bun has a built-in, npm-compatible package manager that installs packages. When installing a Vite React App, `bun install` runs 18x faster than `yarn` and 30x faster than `npm` on Windows.

[![](/images/bun-install-on-windows.png "")](/images/bun-install-on-windows.png "")

Time spent installing dependencies in a vite react app using \`--ignore-scripts\` on Windows.

You can also run scripts using `bun run`, which is a faster alternative to `npm run`. To make `bun run` even faster on Windows, we engineered a new file-format: `.bunx`.

The `.bunx` file is a cross-filesystem symlink that is able to start scripts or executables using Bun or Node.js. We decided to create this for several reasons:

* Symlinks are not guaranteed to work on Windows.
* Shebangs at the top of a file (`#!/usr/bin/env bun`) are not read on Windows.
* We want to avoid creating three permutations of each executable: `.cmd`, `.sh`, and `.ps1`.
* We want to avoid confusing "Terminate batch job? (Y/n)" prompts.

The end result is that `bun run` is 11x faster than `npm run`, and `bunx` is also 11x faster than `npx`.

[![](/images/bun-run-on-windows.png "")](/images/bun-run-on-windows.png "")

Time spent running \`bunx cowsay\` vs \`npx cowsay\` on Windows.

Even if you only use Bun as a package manager and not a runtime, `.bunx` just works with Node.js. This also solves the annoying "Terminate batch job?" prompt that Windows developers are used to when sending ctrl-c to a running script.

[![](/images/terminate-batch-job-bun.gif "")](/images/terminate-batch-job-bun.gif "")

[![](/images/terminate-batch-job-npm.gif "")](/images/terminate-batch-job-npm.gif "")

Bun has built-in support `--watch` mode. This gives you a fast iteration cycle between making changes and having those changes affect your code. On Windows, we made to sure to optimize the time it takes between control-s and process reload.

[![](/images/bun-test-watch-windows.gif "")](/images/bun-test-watch-windows.gif "")

On the left, making changes to a test file. On the right, \`bun test --watch\` on Windows.

We've also spent time optimizing Node.js APIs to use the fastest syscalls available on Windows. For example, [`fs.readdir()`](https://nodejs.org/api/fs.html#fsreaddirpath-options-callback "") on Bun is [58% faster](https://twitter.com/jarredsumner/status/1745742065647132748 "") than Node.js on Windows.

[![](/images/fs-readdir-windows.png "")](/images/fs-readdir-windows.png "")

Time spent listing files in a directory, 1000 times on Windows.

While we haven't optimized *every* API, if you notice something on Windows that is slow or slower than Node.js, [file an issue](/issues "") and we will figure out how to make it faster.

Windows support is just one anecdote when compared to the dozens of new features, APIs, and improvements we've made since Bun 1.0.

Bun has built-in support for JavaScript, TypeScript, and JSX, powered by Bun's very own transpiler written in highly-optimized native code.

Since Bun 1.0, we've implemented a content-addressable cache for files larger than 50KB to avoid the performance overhead of transpiling the same files repeatedly.

This makes command-line tools, like `tsc`, run up to [2x faster](/blog/bun-v1.0.15#transpiler-cache-makes-clis-like-tsc-up-to-2x-faster "") than in Bun 1.0.

[![](/images/tsc-2x-faster.png "")](/images/tsc-2x-faster.png "")

Time spent running \`tsc --help\` in Bun and Node.js. 

Bun is now a cross-platform shell — like bash, but also on Windows.

JavaScript is the world's most popular scripting language. So, why is running shell scripts so complicated?

```
import { spawnSync } from "child_process";

// this is a lot more work than it could be
const { status, stdout, stderr } = spawnSync("ls", ["-l", "*.js"], {
  encoding: "utf8",
});

```

Different platforms also have different shells, each with slightly different syntax rules, behavior, and even commands. For example, if you want to run a shell script using `cmd` on Windows:

* `rm -rf` doesn't work.
* `FOO=bar <command>` doesn't work.
* `which` doesn't exist. (it's called `where` instead)

The [Bun Shell](/docs/runtime/shell "") is a lexer, parser, and interpreter that implements a bash-like programming language, along with a selection of core utilities like `ls`, `rm`, and `cat`.

The shell can also be run from JavaScript and TypeScript, using the `Bun.$` API.

```
import { $ } from "bun";

// pipe to stdout:
await $`ls *.js`;

// pipe to string:
const text = await $`ls *.js`.text();

```

The syntax makes it easy to pass arguments, buffers, and pipes between the shell and JavaScript.

```
const response = await fetch("https://example.com/");

// pipe a response as stdin,
// pipe the stdout back to JavaScript:
const stdout = await $`gzip -c < ${response}`.arrayBuffer();

```

Variables are also escaped to prevent command injection.

```
const filename = "foo.js; rm -rf /";

// ls: cannot access 'foo.js; rm -rf /':
// No such file or directory
await $`ls ${filename}`;

```

You can run shell scripts using the Bun Shell by running `bun run`.

The Bun Shell is enabled by default on Windows when running `package.json` scripts with `bun run`. To learn more, check out the [documentation](/docs/runtime/shell "") or the announcement [blog post](/blog/the-bun-shell "").

Bun now has a built-in [Glob API](/docs/api/glob "") for matching files and strings using glob patterns. It's similar to popular Node.js libraries like `fast-glob` and `micromatch`, except it matches strings [3x faster](/blog/bun-v1.0.14#bun-glob "").

Use `glob.match()` to match a string against a glob pattern.

```
import { Glob } from "bun";

const glob = new Glob("**/*.ts");
const match = glob.match("src/index.ts"); // true

```

Use `glob.scan()` to list files that match a glob pattern, using an `AsyncIterator`.

```
const glob = new Glob("**/*.ts");

for await (const path of glob.scan("src")) {
  console.log(path); // "src/index.ts", "src/utils.ts", ...
}

```

Bun has a new [Semver API](/docs/api/semver "") for parsing and sorting semver strings. It's similar to the popular `node-semver` package, except it's [20x faster](https://twitter.com/jarredsumner/status/1722225679570473173 "").

Use `semver.satisfies()` to check if a version satisfies a range.

```
import { semver } from "bun";

semver.satisfies("1.0.0", "^1.0.0"); // true
semver.satisfies("1.0.0", "^2.0.0"); // false

```

Use `semver.order()` to compare two versions, or sort an array of versions.

```
const versions = ["1.1.0", "0.0.1", "1.0.0"];
versions.sort(semver.order); // ["0.0.1", "1.0.0", "1.1.0"]

```

Bun also supports a new string-width API for measuring the visible width of a string in a terminal. This is useful when you want to know how many columns a string will take up in a terminal.

It's similar to the popular `string-width` package, except it's [6000x faster](/blog/bun-v1.0.29#bun-stringwidth-6-756x-faster-string-width-replacement "").

```
import { stringWidth } from "bun";

stringWidth("hello"); // 5
stringWidth("👋"); // 2
stringWidth("你好"); // 4
stringWidth("👩‍👩‍👧‍👦"); // 2
stringWidth("\u001b[31mhello\u001b[39m"); // 5

```

It supports ANSI escape codes, fullwidth characters, graphemes, and emojis. It also supports Latin1, UTF-16, and UTF-8 encodings, with optimized implementations for each.

When you create an HTTP server using `Bun.serve()`, you can now get the URL of the server using the `server.url` property. This is useful for getting the formatted URL of a server in tests.

```
import { serve } from "bun";

const server = serve({
  port: 0, // random port
  fetch(request) {
    return new Response();
  },
});

console.log(`${server.url}`); // "http://localhost:1234/"

```

You can also get the IP address of a HTTP request using the `server.requestIP()` method. This does not read headers like `X-Forwarded-For` or `X-Real-IP`. It simply returns the IP address of the socket, which may correspond to the IP address of a proxy.

```
import { serve } from "bun";

const server = serve({
  port: 0,
  fetch(request) {
    console.log(server.requestIP(request)); // "127.0.0.1"
    return new Response();
  },
});

```

When you spawn a subprocess using `Bun.spawn()`, you can now access the CPU and memory usage of a process using the `resourceUsage()` method. This is useful for monitoring the performance of a process.

```
import { spawnSync } from "bun";

const { resourceUsage } = spawnSync([
  "bun",
  "-e",
  "console.log('Hello world!')",
]);

console.log(resourceUsage);
// {
//   cpuTime: { user: 5578n, system: 4488n, total: 10066n },
//   maxRSS: 22020096,
//   ...
// }

```

Bun now supports environment variables using `import.meta.env`. It's an alias of [`process.env`](https://nodejs.org/api/process.html#processenv "") and `Bun.env`, and exists for compatibility with other tools in the JavaScript ecosystem, such as Vite.

```
import.meta.env.NODE_ENV; // "development"

```

Bun aims to be a drop-in replacement for Node.js.

Node.js compatibility continues to remain a top priority for Bun. We've made a lot of improvements and fixes to Bun's support for Node.js APIs. Here are some of the highlights:

Bun now supports the [`node:http2`](https://nodejs.org/api/http2.html#http2 "") client APIs, which allow you to make outgoing HTTP/2 requests. This also means you can also use packages like `@grpc/grpc-js` to send gRPC requests over HTTP/2.

```
import { connect } from "node:http2";

const client = connect("https://example.com/");
const request = client.request({ ":path": "/" });

request.on("response", (headers, flags) => {
  for (const name in headers) {
    console.log(`${name}: ${headers[name]}`);
    // "cache-control: max-age=604800", ...
  }
});

request.on("end", () => {
  client.close();
});

request.end();

```

We're still working on adding support for the HTTP/2 server, you can track our progress in this [issue](https://github.com/oven-sh/bun/issues/8823 "").

Bun uses [JavaScriptCore](https://docs.webkit.org/Deep%20Dive/JSC/JavaScriptCore.html "") as its JavaScript engine, as opposed to Node.js which uses [V8](https://v8.dev/ ""). [`Date`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date "") parsing is complicated, and its behaviour varies greatly between engines.

For example, in Bun 1.0 the following `Date` would work in Node.js, but not in Bun:

```
const date = "2020-09-21 15:19:06 +00:00";

Date.parse(date); // Bun: Invalid Date
Date.parse(date); // Node.js: 1600701546000

```

To fix these inconsistencies, we ported the `Date` parser from V8 to Bun. This means that `Date.parse` and `new Date()` behave the same in Bun as it does in Node.js.

In Bun 1.0, we did not have support for the `recursive` option in [`fs.readdir()`](https://nodejs.org/api/fs.html#fsreaddirpath-options-callback ""). This was an oversight and caused subtle bugs with many packages.

Not only did we add support for the `recursive` option, but we also made is 22x faster than Node.js.

[![](/images/fs-readdir-recursive.png "")](/images/fs-readdir-recursive.png "")

Time spent listing files using recursive \`fs.readdir()\` in a large directory.

You can now send IPC messages between Bun and Node.js processes using the `ipc` option. This also fixed a bug that would have caused Bun to hang when using Next.js 14.1.

```
if (typeof Bun !== "undefined") {
  const prefix = `[bun ${process.versions.bun} 🐇]`;
  const node = Bun.spawn({
    cmd: ["node", __filename],
    ipc({ message }) {
      console.log(message);
      node.send({ message: `${prefix} 👋 hey node` });
      node.kill();
    },
    stdio: ["inherit", "inherit", "inherit"],
    serialization: "json",
  });

  node.send({ message: `${prefix} 👋 hey node` });
} else {
  const prefix = `[node ${process.version}]`;
  process.on("message", ({ message }) => {
    console.log(message);
    process.send({ message: `${prefix} 👋 hey bun` });
  });
}

```

Node.js has *lots* of undocumented APIs that you wouldn't find from reading its documentation.

There are millions of npm packages, inevitably some of them will depend on obscure or undocumented APIs. Instead of leaving these packages to be broken or forgotten, we actually add these APIs to Bun so you don't need to rewrite your code.

For example, [`ServerResponse`](https://nodejs.org/api/http.html#class-httpserverresponse "") has an undocumented `_headers` property that allows the HTTP headers to be modified as an object.

```
import { createServer } from "node:http";

createServer((req, res) => {
  const { _headers } = res;
  delete _headers["content-type"];
  res._implicitHeader();
  res.end();
});

```

This API was used in recent Astro release, which we then fixed in Bun. There was also an `_implicitHeader()` function, which was used by Express, and we also fixed that.

There's also a ton of other Node.js APIs that we've either added or fixed, including:

| Module | APIs |
| --- | --- |
| *Top-level* | Added [`import.meta.filename`](https://nodejs.org/api/esm.html#importmetafilename ""), [`import.meta.dirname`](https://nodejs.org/api/esm.html#importmetadirname ""), [`module.parent`](https://nodejs.org/api/modules.html#moduleparent "") |
| [`process`](https://nodejs.org/api/process.html "") | Added [`getReport()`](https://nodejs.org/api/process.html#processreport ""), `binding("tty_wrap")` |
| [`node:util`](https://nodejs.org/api/util.html "") | Added [`domainToASCII()`](https://nodejs.org/api/url.html#urldomaintoasciidomain ""), [`domainToUnicode()`](https://nodejs.org/api/url.html#urldomaintounicodedomain ""), and [`styleText()`](https://nodejs.org/api/util.html#utilstyletextformat-text ""). Changed [`inspect()`](https://nodejs.org/api/util.html#utilinspectobject-options "") to be more consistent with Node.js |
| [`node:crypto`](https://nodejs.org/api/crypto.html "") | Added [`KeyObject`](https://nodejs.org/api/crypto.html#static-method-keyobjectfromkey ""), [`createPublicKey()`](https://nodejs.org/api/crypto.html#cryptocreatepublickeykey ""), [`createPrivateKey()`](https://nodejs.org/api/crypto.html#cryptocreateprivatekeykey ""),[`generateKeyPair()`](https://nodejs.org/api/crypto.html#cryptogeneratekeypairtype-options ""), [`generateKey()`](https://nodejs.org/api/crypto.html#cryptogeneratekeytype-options ""), [`sign()`](https://nodejs.org/api/crypto.html#signsignprivatekey-outputencoding ""), [`verify()`](https://nodejs.org/api/crypto.html#verifyverifyobject-signature-signatureencoding ""), and more |
| [`node:fs`](https://nodejs.org/api/fs.html "") | Added [`openAsBlob()`](https://nodejs.org/api/fs.html#fsopenasblobpath-options ""), [`opendir()`](https://nodejs.org/api/fs.html#fsopendirpath-options-callback ""), [`fdatasync()`](https://nodejs.org/api/fs.html#fsfdatasyncfd-callback ""). Fixed [`FileHandle`](https://nodejs.org/api/fs.html#class-filehandle "") not being returned with various APIs |
| [`node:console`](https://nodejs.org/api/console.html "") | Added [`Console`](https://nodejs.org/api/console.html#class-console "") |
| [`node:dns`](https://nodejs.org/api/dns.html "") | Added [`lookupService()`](https://nodejs.org/api/dns.html#dnslookupserviceaddress-port-callback "") |
| [`node:http`](https://nodejs.org/api/http.html "") | Unix domain sockets via [`request()`](https://nodejs.org/api/http.html#httprequesturl-options-callback "") |
| [`node:events`](https://nodejs.org/api/events.html "") | Added [`on()`](https://nodejs.org/api/events.html#eventsonemitter-eventname-options "") |
| [`node:path`](https://nodejs.org/api/path.html "") | Fixed many bugs with Windows paths. |
| [`node:vm`](https://nodejs.org/api/vm.html "") | Added [`createScript()`](https://nodejs.org/api/vm.html#vm-executing-javascript "") |
| [`node:os`](https://nodejs.org/api/os.html "") | Added [`availableParallelism()`](https://nodejs.org/api/os.html#os_os_availableparallelism "") |

Bun also supports the Web standard APIs, including [`fetch()`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API "") and [`Response`](https://developer.mozilla.org/en-US/docs/Web/API/Response ""). This makes it easier to write code that works in both the browser and Bun.

Since Bun 1.0, we've made a lot of improvements and fixes to the Web APIs.

Previously, [`WebSocket`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/WebSocket "") was marked as experimental due to protocol bugs, such as early disconnects and fragmentation issues.

In Bun 1.1, `WebSocket` is now stable and passes the industry-standard [Autobahn](https://www.google.com/search?q=autobahn+websocket&sourceid=chrome&ie=UTF-8 "") conformance test suite. This fixes [dozens](https://github.com/oven-sh/bun/issues/6686 "") of bugs the `WebSocket` client and makes it more reliable for production use.

```
const ws = new WebSocket("wss://echo.websocket.org/");

ws.addEventListener("message", ({ data }) => {
  console.log("Received:", data);
});

ws.addEventListener("open", () => {
  ws.send("Hello!");
});

```

Bun now supports the [user-timings](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API/User_timing "") APIs, which includes APIs like `performance.mark()` and `performance.measure()`. This is useful for measuring the performance of your application.

```
performance.mark("start");
while (true) {
  // ...
}
performance.mark("end");
performance.measure("task", "start", "end");

```

You can now use [`fetch()`](https://developer.mozilla.org/en-US/docs/Web/API/fetch "") to make requests with the `br` encoding. This is useful for making requests to servers that support Brotli compression.

```
const response = await fetch("https://example.com/", {
  headers: {
    "Accept-Encoding": "br",
  },
});

```

Bun now supports the recently-added [`URL.canParse()`](https://developer.mozilla.org/en-US/docs/Web/API/URL/canParse_static "") API. This makes it possible to check if a string is a valid URL without throwing an error.

```
URL.canParse("https://example.com:8080/"); // true
URL.canParse("apoksd!"); // false

```

Bun now supports sending [`fetch()`](https://developer.mozilla.org/en-US/docs/Web/API/fetch "") requests over a Unix socket.

```
const response = await fetch("http://localhost/info", {
  unix: "/var/run/docker.sock",
});

const { ID } = await response.json();
console.log("Docker ID:", ID); // <uuid>

```

While this not an API that browsers will support, it's useful for server-side applications that need to communicate with services over a Unix socket, like the Docker daemon.

You can now pass an [`AsyncIterator`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/AsyncIterator "") to the [`Response`](https://developer.mozilla.org/en-US/docs/Web/API/Response/Response "") constructor. This is useful for streaming data from a source that doesn't support `ReadableStream`.

```
const response = new Response({
  async *[Symbol.asyncIterator]() {
    yield "Hello, ";
    yield Buffer.from("world!");
  },
});
await response.text(); // "Hello, world!"

```

This is a non-standard extension to the `fetch()` API that Node.js supported, and was added to Bun for compatibility reasons.

Even if you don't use Bun as a runtime, you can still use `bun install` as a package manager. Bun is an npm-compatible package manager that installs packages up to 29x faster than `npm`.

Since Bun 1.0, we have significantly improved the stability and performance of `bun install`. We've fixed hundreds of bugs, added new features, and improved the overall developer experience.

If you ran into a bug using `bun install` in Bun 1.0, chances are it was related to lifecycle scripts. [Lifecycle scripts](https://docs.npmjs.com/cli/v10/using-npm/scripts#life-cycle-scripts "") are scripts that run during the installation of a package, such as `postinstall`.

In Bun 1.1, we've fixed many of those bugs and completely overhauled how lifecycle scripts work.

#### Lifecycle scripts on Windows

On Windows, lifecycle scripts use the Bun Shell. That means you don't need helper libraries like:

* `rimraf`
* `cross-env`
* `node-which`

By default, Bun does not run lifecycle scripts for packages that are not trusted. This is a security feature to prevent malicious scripts from running on your machine. Bun will only run scripts that are defined in the [`trustedDependencies`](https://bun.sh/guides/install/trusted "") list in your `package.json`.

When you first add a package, Bun will tell you if the package had a lifecycle script that did not run.

```
bun add v1.1.0  
 Saved lockfile  
  
 installed @biomejs/biome@1.6.1 with binaries:  
  - biome  
  
 1 package installed [55.00ms]  
  
 Blocked 1 postinstall. Run `bun pm untrusted` for details.  

```

If you want to see which scripts were blocked, you can run `bun pm untrusted`.

```
bun pm untrusted v1.1.0  
  
./node_modules/@biomejs/biome @1.6.1  
 » [postinstall]: node scripts/postinstall.js  
  
These dependencies had their lifecycle scripts blocked during install.  
  
If you trust them and wish to run their scripts, use `bun pm trust`.  

```

If you trust the package, you can run `bun pm trust [package]`. If you want to trust every package, you can also run `bun pm trust --all`.

```
bun pm trust v1.1.0  
  
./node_modules/@biomejs/biome @1.6.1  
 ✓ [postinstall]: node scripts/postinstall.js  
  
 1 script ran across 1 package [71.00ms]  

```

If you already know you want to trust a dependency, you can add it using `bun add --trust [package]`. This will add the package and it's transitive dependencies to your `trustedDependencies` list so you don't need to run `bun pm trust` for that package.

```
{
  "dependencies": {
    "@biomejs/biome": "1.6.1"
  },
   "trustedDependencies": [
     "@biomejs/biome"
   ]
}
```

Bun includes a default allowlist of popular packages containing lifecycle scripts that are known to be safe. You can see the full list by running `bun pm default-trusted`.

Also, to reduce the performance impact of lifecycle scripts, we've made them run in parallel. This means that lifecycle scripts will run concurrently, which reduces the time it takes to install packages.

[![](https://github.com/oven-sh/bun/assets/709451/4ad72f2f-6152-4e4e-b1d1-d832cefdd8f8 "")](https://github.com/oven-sh/bun/assets/709451/4ad72f2f-6152-4e4e-b1d1-d832cefdd8f8 "")

Bun uses a binary lockfile, `bun.lockb`, for faster installs with `bun install`.

You can now run `bun pm migrate` to convert a `package-lock.json` file to a `bun.lockb` file. This is useful if you want to do a one-time migration from npm to Bun.

```
[5.67ms] migrated lockfile from package-lock.json

 21 packages installed [54.00ms]
```

You don't need to run this command if you're using `bun install`, as it will automatically migrate the lockfile if it detects a `package-lock.json` file.

Bun is a JavaScript and TypeScript bundler, transpiler, and minifier that can be used to bundle code for the browser, Node.js, and other platforms.

Bun can bundle code to run on Node.js using the `--target=node` flag.

```
var { promises } = require("node:fs");
var { join } = require("node:path");

promises.readFile(join(__dirname, "data.txt"));

```

In Bun 1.0, there were several bugs that prevented this from working correctly, such as not being able to require built-in modules like `node:fs` and `node:path`. Here's what that looked like in Bun 1.0:

```
bun-1.0 build --target=node app.ts --outfile=dist.mjs
```

```
TypeError: (intermediate value).require is not a function
    at __require (file:///app.mjs:2:22)
    at file:///app.mjs:7:20
    at ModuleJob.run (node:internal/modules/esm/module_job:218:25)
    at async ModuleLoader.import (node:internal/modules/esm/loader:329:24)
    at async loadESM (node:internal/process/esm_loader:28:7)
    at async handleMainPromise (node:internal/modules/run_main:113:12)
```

In Bun 1.1, these bugs have now been fixed.

```
bun build --target=node app.ts --outfile=dist.mjs
```

Bun can compile TypeScript and JavaScript files to a single-file executable using the `--compile` flag.

```
bun build --compile app.ts
```

In Bun 1.1, you can also embed NAPI (n-api) addons `.node` files. This is useful for bundling native Node.js modules, like `@anpi-rs/canvas`.

```
import { promises } from "fs";
import { createCanvas } from "@napi-rs/canvas";

const canvas = createCanvas(300, 320);
const data = await canvas.encode("png");

await promises.writeFile("empty.png", data);

```

Then you can compile and run your application as a single-file executable.

```
bun build --compile canvas.ts
```

Bun has a powerful macro system that allows you to transform your code at compile-time. Macros can be used to generate code, optimize code, and even run code at compile-time.

In Bun 1.1, you can now import built-in modules at bundle-time.

#### Read a file into a string at bundle-time

```
import { readFileSync } from "node:fs"
  with { type: "macro" };

export const contents = readFileSync("hello.txt", "utf8");

```

#### Spawn a process at bundle-time

```
import { spawnSync } from "node:child_process"
  with { type: "macro" };

const result = spawnSync("echo", ["Hello, world!"], {encoding: "utf-8"}).stdout;
console.log(result); // "Hello, world!"

```

Bun has a built-in test module that makes it easy to write and run tests in JavaScript, TypeScript, and JSX. It supports the same APIs as Jest, which includes the [`expect()`](https://jestjs.io/docs/expect#expectvalue "")-style APIs.

Matchers are assertions that you can use to test your code. Since Bun 1.0, we've added dozens of new `expect()` matchers, including:

```
import { expect } from "bun:test";

expect.hasAssertions();
expect.assertions(9);
expect({}).toBeObject();
expect([{ foo: "bar" }]).toContainEqual({ foo: "bar" });
expect(" foo ").toEqualIgnoringWhitespace("foo");
expect("foo").toBeOneOf(["foo", "bar"]);
expect({ foo: Math.PI }).toEqual({ foo: expect.closeTo(3.14) });
expect({ a: { b: 1 } }).toEqual({ a: expect.objectContaining({ b: 1 }) });
expect({ a: Promise.resolve("bar") }).toEqual({ a: expect.resolvesTo("bar") });
expect({ b: Promise.reject("bar") }).toEqual({ b: expect.rejectsTo("bar") });
expect.unreachable();

```

If there's a matcher that Bun doesn't support, you can create your own using [`expect.extend()`](https://jestjs.io/docs/expect#expectextendmatchers ""). This is useful when you want to define a custom matcher that is reusable across multiple tests.

```
import { test, expect } from "bun:test";

expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `Expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `Expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

test("toBeWithinRange()", () => {
  expect(1).toBeWithinRange(1, 99); // ✅
  expect(100).toBeWithinRange(1, 99); // ❌ Expected 100 to be within range 1 - 99
});

```

Bun now supports module mocking.

* Unlike Jest, Bun is able to mock both ESM and CommonJS modules.
* If a module has already been imported, Bun is able to update the module *in-place*, which means that mocks work at runtime. Other test runners can't do this, since they set mocks at build time.
* You can override anything: local files, npm packages, and built-in modules.

file.js

```
import { mock, test, expect } from "bun:test";
import { fn } from "./mock";

test("mocking a local file", async () => {
  mock.module("./mock", () => {
    return {
      fn: () => 42,
    };
  });

  // fn is already imported, so it will be updated in-place
  expect(fn()).toBe(42);

  // also works with cjs
  expect(require("./mock").fn()).toBe(42);
});

```

package.js

```
import { mock, test, expect } from "bun:test";
import stringWidth from "string-width";

test("mocking an npm package", async () => {
  mock.module("string-width", () => {
    return {
      default: Bun.stringWidth,
    };
  });

  const string = "hello";
  expect(stringWidth()).toBe(5);
  expect(require("string-width")()).toBe(5);
});

```

built-in.js

```
import { mock, test, expect } from "bun:test";
import { readFileSync } from "node:fs";

test("mocking a built-in module", async () => {
  mock.module("node:fs", () => {
    return {
      readFileSync: () => "mocked!",
    };
  });

  expect(readFileSync("./foo.txt")).toBe("mocked!");
  expect(require("fs").readFileSync("./bar.txt")).toBe("mocked!");
});

```

Since 1.0, Bun has had built-in support for [SQLite](https://www.sqlite.org/ ""). It has an API that's inspired by `better-sqlite3`, but is written in native code to be faster.

```
import { Database } from "bun:sqlite";

const db = new Database(":memory:");
const query = db.query("select 'Bun' as runtime;");
query.get(); // { runtime: "Bun" }

```

Since then, there's been lots of new features and improvements to `bun:sqlite`.

We've added support for multi-statement queries, which allows multiple SQL statements to be run in a single call to `run()` or `exec()` delimited by a `;`.

```
import { Database } from "bun:sqlite";

const db = new Database(":memory:");

db.run(`
  CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT
  );

  INSERT INTO users (name) VALUES ("Alice");
  INSERT INTO users (name) VALUES ("Bob");
`);

```

When an error is thrown from `bun:sqlite`, you'll now see more detailed errors, including the table and column name. In Bun 1.0, the error message was terse and did not contain extra details.

```
- error: constraint failed
+ SQLiteError: UNIQUE constraint failed: foo.bar
+   errno: 2067
+   code: "SQLITE_CONSTRAINT_UNIQUE"
      at run (bun:sqlite:185:11)
      at /index.js:7:1

```

In Bun 1.1, you can now import a SQLite database using the `import` syntax. This uses the new [import attributes](https://nodejs.org/api/esm.html#import-attributes "") feature to specify the type of import.

```
import db from "./users.db"
  with { type: "sqlite" };

const { n } = db
  .query("SELECT COUNT(id) AS n FROM users")
  .get();

console.log(`Found ${n} users!`); // "Found 42 users!"

```

You can also compile your application and SQLite database into a [single-file executable](/docs/bundler/executables#sqlite ""). To enable this, specify the `embed` property on the import, then use `bun build --compile` to build your app.

```
import db from "./users.db"
  with { type: "sqlite", embed: "true" };

```

```
bun build --compile ./app.ts
```

We spent a lot of time thinking about the developer experience in Bun. We want to make it easy to write, run, and debug JavaScript and TypeScript code.

There are tons of improvements to commands, output, and error messages to make Bun easier to use.

When an error is thrown in Bun, it prints a stack trace to the console with a multi-line source code preview. Now that source code preview gets syntax highlighted, which makes it easier to read.

[![](https://github.com/oven-sh/bun/assets/709451/a95254d0-2652-433c-80e7-7faac1e38c2a "")](https://github.com/oven-sh/bun/assets/709451/a95254d0-2652-433c-80e7-7faac1e38c2a "")

A preview of errors with and without syntax highlighting.

Stack traces from `Error.stack` now include less noise, such as internal functions that are not relevant to the error. This makes it easier to see where the error occurred.

```
1 | throw new Error("Oops");
          ^
error: Oops
    at /oops.js:1:7
    at globalThis (/oops.js:3:14)
    at overridableRequire (:1:20)
    at /index.js:3:8
    at globalThis (/index.js:3:8)
```

You can run `bun --eval`, or `bun -e` for short, to evaluate a script without creating a file. Just like the rest of Bun, it supports top-level await, ESM, CommonJS, TypeScript, and JSX.

You can pass the script as a string.

```
bun -e 'console.log(Bun.version)'
```

Or you can pipe the script through stdin using `bun -`.

```
echo 'console.log(await fetch("https://example.com/"))' | bun -
```

```
Response (1.26 KB) {
  status: 200, ...
}
```

You can also use `bun --print`, which is the same as `bun -e`, except it prints the last statement using `console.log()`.

```
bun --print 'await Bun.file("package.json").json()'
```

```
{
  name: "bun",
  dependencies: { ... },
}
```

You can also omit `await` since Bun will detect dangling promises.

```
bun --print 'fetch("https://example.com/").then(r => r.text())'
```

Bun detects and loads [`.env`](https://bun.sh/docs/runtime/env "") files by default, but now you can use `bun --env-file` to load a custom `.env` file. This is useful for testing different environments.

```
bun --env-file=custom.env src/index.ts
```

```
bun --env-file=.env.a --env-file=.env.b run build
```

You can use `--env-file` when running JavaScript files or when running package.json scripts.

Bun 1.1 contains a few, minor tweaks in behaviour that you should be aware of, but we think are highly unlikely to break your code.

In Bun 1.0, the default network timeout for `fetch()` and `bun install` was 30 seconds.

Since [Bun 1.0.4](https://bun.sh/blog/bun-v1.0.4#potentially-breaking-changes ""), the default network timeout has been increased to **5 minutes**. This aligns the default with Google Chrome and should help with high-latency connections.

You can also disable the timeout with `fetch()` using:

```
const response = await fetch("https://example.com/", {
  timeout: false,
});

```

Previously, `Bun.write()` would throw an error if the parent directory didn't exist.

```
import { write } from "bun";

await write("does/not/exist/hello.txt", "Hello!");
// ENOENT: No such file or directory

```

Since [Bun 1.0.16](https://bun.sh/blog/bun-v1.0.16#bun-write-now-creates-the-parent-directory-if-it-doesn-t-exist ""), Bun will create the parent directory if it doesn't exist.

While this does not match the behaviour of APIs like `fs.writeFileSync()`, developers asked us to make this change so the API would be more intuituve to use and lead to a better developer experience.

Without this change, developers would have to write the following boilterplate code:

```
import { write } from "bun";
import { mkdir } from "node:fs/promises";

try {
  await write("does/not/exist/hello.txt", "Hello!");
} catch (error) {
  if (error.code === "ENOENT") {
    await mkdir("does/not/exist", { recursive: true });
    await write("does/not/exist/hello.txt", "Hello!");
  } else {
    throw error;
  }
}

```

If you want to restore the old behavior, you can specify the `createPath` property.

```
import { write } from "bun";

await write("does/not/exist/hello.txt", "Hello, world!", { createPath: false });
// ENOENT: No such file or directory

```

Packages can use [conditional exports](https://nodejs.org/api/packages.html#packages_conditional_exports "") to specify different entry files for different environments. For example, a package might define a `browser` export for the browser and a `node` export for Node.js.

```
{
  "exports": {
    "node": "./node.js",
    "browser": "./browser.js",
    "worker": "./worker.js"
  }
}

```

In Bun 1.0, Bun would select the first export using the following order: `bun`, `worker`, `node`.

In Bun 1.1, Bun will no longer select the `worker` export, since that is associated with [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API ""), which often assume a browser-like environment.

This change only applies when Bun is being used as a runtime, and fixes various bugs where a `worker` export would be selected before a more-applicable `node` export.

In Node.js, [`process.env.NODE_ENV`](https://nodejs.org/en/learn/getting-started/nodejs-the-difference-between-development-and-production#nodejs-the-difference-between-development-and-production "") is set to `undefined` by default.

Early in Bun's development, we set the default to be `development` which turned out to be a mistake. This is because developers often forget to set `NODE_ENV` to production, which can lead to development functionality being included in production builds.

In Bun 1.1, we changed the default `NODE_ENV` to `undefined` to match Node.js.

```
bun --print 'process.env.NODE_ENV'
```

```
NODE_ENV=development bun --print 'process.env.NODE_ENV'
```

Previously, if you installed a package using the `latest` tag, it would write the literal string `latest` to your `package.json`. This was not intended and does not match the behavior of other package managers.

In Bun 1.1, the `latest` tag is resolved before being written to the `package.json`.

```
bun install lodash@latest
```

package.json

```
{
  "dependencies": {
     "lodash": "latest"
     "lodash": "^4.17.21"
  }
}
```

The Bun shell was introduced in [Bun 1.0.24](https://bun.sh/blog/bun-v1.0.24#bun-shell ""). When a subprocess exited, the promise resolved even if the exit code was non-zero.

```
import { $ } from "bun";

await $`cd /does/not/exist`;
// does not throw

```

This is often not the desired behavior, and would cause bugs to go unnoticed where a command failed but the promise resolved.

In Bun 1.1, the Bun shell will now reject with an error when the subprocess exits with a non-zero exit code.

```
import { $ } from "bun";

await $`cd /does/not/exist`;
// ShellError: cd /does/not/exist: No such file or directory

```

If you want to revert back to the previous behavior, you can call the `throws()` function.

```
import { $ } from "bun";

const { exitCode, stderr } = await $`cd /does/not/exist`.throws(false);
console.log(exitCode); // 1
console.log(stderr); // "cd: /does/not/exist: No such file or directory"

```

In Bun 1.0, [`import.meta.resolve()`](https://nodejs.org/api/esm.html#importmetaresolvespecifier "") would asynchronously resolve to an absolute file path.

This matched the behavior of Node.js' original implementation. However, for Web API compatibility reasons, Node.js changed the API to be synchronous. And so, Bun has done the same.

```
import.meta.resolve("./foo.js"); // Before: Promise { "/path/to/foo.js" }
import.meta.resolve("./foo.js"); // After: "file:///path/to/foo.js"

```

Since Bun 1.0, we've fixed over a thousand bugs.

If you ran into an error using Bun 1.0, we'd encourage you to try again with Bun 1.1. And if there's still something we haven't fixed, please feel free to create a [new issue](/issues "") or bump an existing one.

You can upgrade to Bun 1.1 with the following command:

Out of the thousands of bugs fixed, here are a few of the most common issues that you may have encountered, that have now been fixed.

* `Module not found` after running `bun install`.
* `WebSocket` errors or early disconnects.
* `Bun.file` would sometimes cause an `EBADF: Bad file descriptor` error.
* `bun install` would resolve an incorrect version if certain pre/post tags were present.
* `bun install --yarn` would sometimes generate invalid YAML.
* `Failed to start server. Is port in use?` inside Docker containers.
* `"pidfd_open(2)" system call is not supported` on Vercel and Google Cloud.
* `Bun.serve()` [not responding](https://github.com/oven-sh/bun/issues/3580 "") to HTTP requests with an `_` header.
* Listening to `0.0.0.0` would bind to IPv6 as well.
* `process.nextTick()` and `setImmediate()` would execute in a different order than Node.js.

We are constantly making changes to Bun so it can be faster and more efficient. Follow [@bunjavascript](https://twitter.com/bunjavascript "") to get the latest digest of "In the next version of Bun."

Here is a snippet of some of the performance improvements made since Bun 1.0:

* `Bun.write()` and `Bun.file().text()` are [3x faster](/blog/bun-v1.0.16#bun-write-and-bun-file-text-gets-3x-faster-under-concurrent-load "") under concurrent load.
* `bunx esbuild` is [50x faster](/blog/bun-v1.0.17#bunx-esbuild-starts-50x-faster-than-npx-esbuild "") than `npx esbuild`.
* `fs.cp()` and `fs.copyFile()` are [50% faster](/blog/bun-v1.0.34#50-faster-cross-mount-fs-cp-fs-copyfile-on-linux "") on Linux across filesystems.
* `Bun.peek()` is [90x faster](/blog/bun-v1.0.19#bun-peek-gets-90x-faster "") with a new implementation.
* `expect().toEqual()` is [100x faster](/blog/bun-v1.0.19#expect-map1-toequal-map2-gets-100x-faster "") using `Map` and `Set` objects.
* `setTimeout()` and `setInterval()` are [4x faster](/blog/bun-v1.0.19#settimeout-setinterval-get-4x-higher-throughput "") on Linux.
* `Bun.spawnSync()` is [50% faster](/blog/bun-v1.0.19#optimized-bun-spawnsync-for-large-input-on-linux "") at buffering stdout on Linux.
* `node:http` is [14% faster](/blog/bun-v1.0.10#node-http-gets-14-faster "") using a hello world benchmark.
* `fs.readdir()` is [40x faster](/blog/bun-v1.0.15#recursive-in-fs-readdir-is-40x-faster-than-node-js "") than Node.js using the `recursive` option.
* `bun:sqlite` uses [4x less](/blog/bun-v1.0.21#bun-sqlite-uses-less-memory "") memory.
* `fs.readlink()` uses [2x less](/blog/bun-v1.0.20#fs-readlink-uses-up-to-2x-less-memory "") memory.
* `fs.stat()` uses [2x less](/blog/bun-v1.0.20#fs-stat-uses-up-to-2x-less-memory "") memory.
* `FormData` uses [less memory](/blog/bun-v1.0.21#copy-on-write-file-uploads-on-linux "") with copy-on-write file upload on Linux

That's it — that's Bun 1.1, and this is still just the beginning for Bun.

We've made Bun faster, more reliable, fixed a thousand of bugs, added tons of new features and APIs, and now Bun supports Windows. To get started, run any of the following commands in your terminal.

curl

```
curl -fsSL https://bun.sh/install | bash
```

powershell

```
> powershell -c "irm bun.sh/install.ps1 | iex"

```

docker

```
docker run --rm --init --ulimit memlock=-1:-1 oven/bun
```

We're hiring engineers, designers, and past or present contributors to JavaScript engines like V8, WebKit, Hermes, and SpiderMonkey to join our team in-person in San Francisco to build the future of JavaScript.

You can check out our [careers](/careers "") page or send us an [email](mailto:jobs@bun.sh "").

Bun is free, open source, and MIT-licensed.

As such, that also means that we receive a wide range of open source contributions from the community. We'd like to thank everyone who has opened an issue, fixed a bug, or even contributed a feature.


