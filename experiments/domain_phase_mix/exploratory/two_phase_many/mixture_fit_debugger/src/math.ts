import { browserAdaptor } from "@mathjax/src/js/adaptors/browserAdaptor.js";
import { RegisterHTMLHandler } from "@mathjax/src/js/handlers/html.js";
import { TeX } from "@mathjax/src/js/input/tex.js";
import "@mathjax/src/js/input/tex/ams/AmsConfiguration.js";
import { mathjax } from "@mathjax/src/js/mathjax.js";
import { SVG } from "@mathjax/src/js/output/svg.js";

const adaptor = browserAdaptor();
RegisterHTMLHandler(adaptor);

const input = new TeX({ packages: ["base", "ams"] });
const output = new SVG({ fontCache: "local" });
const mathDocument = mathjax.document(document, { InputJax: input, OutputJax: output });

export function renderMath(tex: string, display: boolean): string {
  const node = mathDocument.convert(tex, { display });
  return adaptor.outerHTML(node);
}
