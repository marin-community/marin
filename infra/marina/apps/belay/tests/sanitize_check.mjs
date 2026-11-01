// Renders hostile agent-log text through common.js's markdown() and link() on a minimal fake DOM, and prints the
// resulting tree as JSON for test_public_rendering.py to check. No HTML parser exists in the fake DOM: an element
// can only appear because common.js created it.
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

class Text {
  constructor(text) {
    this.text = text
  }
  toJSON() {
    return { text: this.text }
  }
}

class Element {
  constructor(tag) {
    this.tag = tag
    this.attrs = {}
    this.children = []
    this.dataset = {}
  }
  set className(value) {
    this.attrs.class = value
  }
  set textContent(value) {
    this.children = [new Text(value)]
  }
  setAttribute(key, value) {
    this.attrs[key] = String(value)
  }
  addEventListener() {}
  append(...nodes) {
    this.children.push(...nodes)
  }
  set innerHTML(_) {
    throw new Error('innerHTML used')
  }
  toJSON() {
    return { tag: this.tag, attrs: this.attrs, children: this.children }
  }
}

const document = {
  body: { dataset: {} },
  querySelector: () => ({ content: 'test' }),
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => new Text(text),
}
const context = vm.createContext({ document, location: { pathname: '/belay/status' }, Node: Object, console })
vm.runInContext(readFileSync(process.argv[2], 'utf8'), context)
const source = readFileSync(0, 'utf8')
const rendered = vm.runInContext(
  `[markdown(${JSON.stringify(source)}), link('javascript:alert(1)', 'a'), link('//evil.example/x', 'b'),
    link('/belay/status#X0001', 'c'), link('https://example.com/ok', 'd')]`,
  context,
)
process.stdout.write(JSON.stringify(rendered))
