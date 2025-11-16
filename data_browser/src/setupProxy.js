const { createProxyMiddleware } = require('http-proxy-middleware');

const DEFAULT_TARGET = 'http://localhost:5000';
const TARGET_ENV_VARS = [
  'MARIN_DATA_BROWSER_PROXY_TARGET',
  'REACT_APP_MARIN_DATA_BROWSER_PROXY_TARGET',
];

function resolveTarget() {
  for (const envName of TARGET_ENV_VARS) {
    if (process.env[envName]) {
      return process.env[envName];
    }
  }
  return DEFAULT_TARGET;
}

module.exports = function setupProxy(app) {
  const target = resolveTarget();
  console.log(`[data-browser] Proxying /api to ${target}`);
  const debugProxy = process.env.MARIN_DATA_BROWSER_PROXY_DEBUG === '1';
  const logRequest = (req, res, next) => {
    if (debugProxy) {
      console.log(`[data-browser] proxy ${req.method} ${req.originalUrl} -> ${target}`);
    }
    next();
  };
  app.use(
    '/api',
    logRequest,
    createProxyMiddleware({
      target,
      changeOrigin: true,
      logLevel: 'warn',
      pathRewrite: (path, req) => req.originalUrl || path,
    }),
  );
};
