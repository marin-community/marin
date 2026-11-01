// Loads the page's script: /belay/ front, /belay/status, /belay/mfu (hero-scale MFU) and /belay/admin.
const SCRIPTS = { front: 'front.js', status: 'status.js', mfu: 'mfu.js', admin: 'admin.js' }
const page = pageName()
document.getElementById('brand').href = PAGES.front
renderNav(page)
const script = document.createElement('script')
script.src = `${BELAY_BASE}${SCRIPTS[page]}`
document.body.append(script)
