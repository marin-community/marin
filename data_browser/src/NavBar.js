import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { viewSingleUrl, viewUrl } from './utils';

function NavBar({ rootPaths }) {
  const location = useLocation();
  const navigate = useNavigate();

  // Parse current paths from URL
  const urlParams = new URLSearchParams(location.search);
  const rawPaths = urlParams.get("paths");
  let currentPaths = [];
  if (rawPaths) {
    try {
      const parsed = JSON.parse(rawPaths);
      if (Array.isArray(parsed)) currentPaths = parsed;
    } catch (e) { /* ignore */ }
  }

  // Which roots are currently active (have a path under them in the view)
  const activeRoots = new Set(
    rootPaths.filter(root => currentPaths.some(p => p.startsWith(root)))
  );

  const isOnViewPage = location.pathname.startsWith("/view");

  // Extract short label from gs://marin-us-central2 -> us-central2
  function label(path) {
    const match = path.match(/gs:\/\/marin-(.+)/);
    return match ? match[1] : path;
  }

  // Add a root path to the current paths list
  function addPath(rootPath) {
    if (currentPaths.some(p => p.startsWith(rootPath))) return;
    const newPaths = [...currentPaths, rootPath];
    navigate(viewUrl({ paths: newPaths }));
  }

  if (!rootPaths || rootPaths.length === 0) return null;

  return (
    <nav style={{
      display: "flex",
      alignItems: "center",
      gap: 0,
      borderBottom: "2px solid #e0e0e0",
      padding: "0 12px",
      backgroundColor: "#fafafa",
      flexWrap: "wrap",
    }}>
      <a
        href="/"
        onClick={(e) => { e.preventDefault(); navigate("/"); }}
        style={{
          padding: "10px 14px",
          textDecoration: "none",
          color: location.pathname === "/" ? "#1976d2" : "#555",
          fontWeight: location.pathname === "/" ? 700 : 500,
          fontSize: 14,
          borderBottom: location.pathname === "/" ? "2px solid #1976d2" : "2px solid transparent",
          marginBottom: -2,
        }}
      >
        Home
      </a>
      {rootPaths.map((path) => {
        const isActive = activeRoots.has(path);
        return (
          <span key={path} style={{ display: "flex", alignItems: "center", marginBottom: -2 }}>
            <a
              href={viewSingleUrl(path)}
              onClick={(e) => {
                e.preventDefault();
                navigate(viewSingleUrl(path));
              }}
              style={{
                padding: "10px 8px 10px 14px",
                textDecoration: "none",
                color: isActive ? "#1976d2" : "#555",
                fontWeight: isActive ? 700 : 500,
                fontSize: 14,
                borderBottom: isActive ? "2px solid #1976d2" : "2px solid transparent",
                whiteSpace: "nowrap",
              }}
            >
              {label(path)}
            </a>
            {isOnViewPage && !isActive && (
              <button
                onClick={() => addPath(path)}
                title={`Add ${label(path)} to current view`}
                style={{
                  background: "none",
                  border: "1px solid #ccc",
                  borderRadius: 4,
                  color: "#888",
                  cursor: "pointer",
                  fontSize: 12,
                  lineHeight: 1,
                  padding: "2px 5px",
                  marginRight: 6,
                }}
              >
                +
              </button>
            )}
          </span>
        );
      })}
    </nav>
  );
}

export default NavBar;
