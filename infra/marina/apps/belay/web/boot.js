// Loads the page's script: /belay/ the front page, /belay/status the status page, /belay/admin the admin page.
const SCRIPTS = { front: 'front.js', status: 'status.js', admin: 'admin.js' }
const page = pageName()
document.getElementById('brand').href = PAGES.front
renderNav(page)
const script = document.createElement('script')
script.src = `${BELAY_BASE}${SCRIPTS[page]}`
document.body.append(script)
