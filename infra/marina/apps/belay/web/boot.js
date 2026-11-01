// Loads the page's script under the app's root: front, status, mfu (hero-scale MFU) and admin.
const SCRIPTS = { front: 'front.js', status: 'status.js', mfu: 'mfu.js', admin: 'admin.js' }
const page = pageName()
document.getElementById('brand').href = PAGES.front
renderNav(page)
const script = document.createElement('script')
script.src = asset(SCRIPTS[page])
document.body.append(script)
