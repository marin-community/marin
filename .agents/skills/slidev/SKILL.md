---
name: slidev
description: Use when asked to create, edit, preview, or export a slide deck with Slidev (Markdown slides with KaTeX, Mermaid, and code blocks); covers install, authoring, export, and visual checks.
---

# Slidev decks

Slidev renders one Markdown file into a web slide deck (Vue + Vite). On this machine the CLI is
installed globally (`/opt/homebrew/bin/slidev`, `@slidev/cli` 52.x) together with
`@slidev/theme-default` and `playwright-chromium`, which `slidev export` needs. Reinstall or upgrade with:

```bash
npm install -g @slidev/cli @slidev/theme-default
npm install -g --allow-scripts=playwright-chromium playwright-chromium
```

The theme is a separate global package; without it every command fails with `The theme
"@slidev/theme-default" was not found`. Other themes install the same way (`@slidev/theme-seriph`).
The Playwright command must carry `--allow-scripts`; npm otherwise skips the Chromium download and
export fails with a missing browser. Check the install with `slidev --version`.

## Where decks live

- Research handoff decks: `.agents/handoffs/slides/<deck_name>/slides.md` with figures beside it.
  The deck folder needs no `package.json` when the global CLI is used; keep `node_modules/`, `dist/`,
  and exported PDFs out of git unless the user asks to commit an export.
- User-facing documentation decks belong under `docs/` and must be linked from MkDocs.

## Commands

```bash
slidev slides.md --open                      # dev server with hot reload (default http://localhost:3030)
slidev export slides.md --output deck.pdf    # PDF; add --with-toc for an outline, --dark for dark theme
slidev export slides.md --format png --output png_dir   # one PNG per slide, for visual checks
slidev build slides.md --out dist            # static site
```

Run commands from the deck folder so relative asset paths resolve. If `rm` is aliased to `trash` in the
user's shell, clear an old export directory with `command rm -rf <dir>` or skip the removal. Export renders every slide in
Chromium; add `--timeout 120000` for decks with many KaTeX or Mermaid blocks.

## Authoring rules

- Start the file with the deck frontmatter (`theme: default`, `title`, `info`, `mdc: true`). Separate
  slides with a line containing only `---`; a slide may start with its own frontmatter block
  (`layout: two-cols`, `layout: center`, `layout: section`, `class: text-sm`).
- `two-cols` layouts use `::right::` to start the second column. `image-right` takes `image: ./file.svg`.
- KaTeX is built in: `$...$` inline, `$$ ... $$` on their own lines for display math. Use `\\` inside
  `aligned` environments and `\text{}` for words. Backslashes need no extra escaping in Markdown.
- Mermaid diagrams go in ```` ```mermaid ```` fences. Code fences accept line highlights, for example
  ```` ```python {2,4} ````.
- `<v-click>` or `v-click` on a list reveals items step by step; avoid it in decks meant for PDF review.
- UnoCSS utility classes are available: `text-sm`, `text-xs`, `grid grid-cols-2 gap-4`, `h-80`,
  `mt-4`. Wrap dense tables in `<div class="text-xs">`.
- Reference figures relative to `slides.md` (`![](./figure.svg)` or `<img src="./figure.svg" class="h-72">`).
  Generate figures as SVG with matplotlib (`uv run --with matplotlib python ...`) beside the deck.
- Keep one idea per slide, at most about twelve lines of body text or one table of at most ten rows.
  A slide that overflows renders clipped in the PDF.
- Prose follows `.agents/skills/writing-style/SKILL.md` (reports guide) and the final
  `ai-writing-donts.md` pass: lead with results, numbers with units, no hype, no emoji.

## Verify before delivering

1. `slidev export slides.md --format png --output <dir>` and read a sample of the PNGs (title, the
   densest table, every math slide) to catch overflow, unrendered `$`, or broken images.
2. Export the PDF and report its path and page count (`mdls -name kMDItemNumberOfPages deck.pdf` on macOS).
3. Do not commit exports or `node_modules/`: give every deck folder a `.gitignore` with `png/`, `*.pdf`,
   `node_modules/`, and `dist/`.

## Deliver in the side panel

The user reviews decks in the Claude desktop app's side panel, so every "make slides" request ends with
the live deck open there, in addition to the attached PDF:

1. Add or update an entry in `.claude/launch.json` (git-ignored, so re-create it per checkout):

   ```json
   {
     "name": "slidev-<deck-name>",
     "runtimeExecutable": "slidev",
     "runtimeArgs": [".agents/handoffs/slides/<deck-name>/slides.md", "--port", "3030"],
     "port": 3030
   }
   ```

2. Call `preview_start` with that name; it opens the Browser pane at `http://localhost:3030`. Take a
   screenshot to confirm the title slide rendered. Arrow keys and clicks navigate; edits to `slides.md`
   hot-reload.
3. Send the exported PDF with `SendUserFile` as an attachment, and give the `slidev slides.md --open`
   command for viewing outside the app.
