// Loads the view's script (front, status, mfu or admin, from the fragment), and reloads when a link moves to
// another view; a fragment within the same view (a front-page section, a status row) stays in place.
const SCRIPTS = { front: 'front.js', status: 'status.js', mfu: 'mfu.js', admin: 'admin.js' }
const page = pageName()
document.getElementById('brand').href = PAGES.front
renderNav(page)
const script = document.createElement('script')
script.src = asset(SCRIPTS[page])
document.body.append(script)
window.addEventListener('hashchange', () => {
  if (pageName() !== page) location.reload()
})
