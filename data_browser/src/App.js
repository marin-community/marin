import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import axios from 'axios';
import HomePage from './HomePage';
import ViewPage from './ViewPage';
import ExperimentPage from './ExperimentPage';
import NavBar from './NavBar';
import { apiConfigUrl, checkJsonResponse } from './utils';

// If we're in an iframe, send messages to the
// parent when the route changes. This allows the
// parent to update its own URL accordingly.
function RouteChangeNotifier() {
  const location = useLocation();

  useEffect(() => {
    // Check if we're in an iframe
    if (window !== window.parent) {
      // Send message to parent with the new path and query string
      window.parent.postMessage({
        type: 'ROUTE_CHANGE',
        path: location.pathname + location.search
      }, '*');
    }
  }, [location]);

  return null;
}

function AppContent() {
  const [rootPaths, setRootPaths] = useState([]);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await axios.get(apiConfigUrl());
        const payload = checkJsonResponse(response, () => {});
        if (payload && Array.isArray(payload.root_paths)) {
          setRootPaths(payload.root_paths);
        }
      } catch (e) {
        console.error("Failed to fetch config for nav bar:", e);
      }
    };
    fetchConfig();
  }, []);

  return (
    <>
      <NavBar rootPaths={rootPaths} />
      <div style={{ paddingTop: 8 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/view/*" element={<ViewPage />} />
          <Route path="/experiment/*" element={<ExperimentPage />} />
        </Routes>
      </div>
    </>
  );
}

function App() {
  return (
    <Router>
      <RouteChangeNotifier />
      <AppContent />
    </Router>
  );
}

export default App;
