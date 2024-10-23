import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './HomePage';
import ViewPage from './ViewPage';
import ExperimentPage from './ExperimentPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/view/*" element={<ViewPage />} />
        <Route path="/experiment/*" element={<ExperimentPage />} />
      </Routes>
    </Router>
  );
}

export default App;
