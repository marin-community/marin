import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import HomePage from './HomePage';
import ViewPage from './ViewPage';
import ExperimentPage from './ExperimentPage';

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

function App() {
  return (
    <Router>
      <RouteChangeNotifier />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/view/*" element={<ViewPage />} />
        <Route path="/experiment/*" element={<ExperimentPage />} />
      </Routes>
    </Router>
  );
}

export default App;
