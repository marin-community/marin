// Picks the view: `/belay/admin` loads the admin view (the server decides whether the caller may see it), anything
// else the public view.
const ADMIN_PATH = /\/admin\/?$/
const script = document.createElement('script')
script.src = `${BELAY_BASE}${ADMIN_PATH.test(location.pathname) ? 'admin.js' : 'public.js'}`
document.body.append(script)
