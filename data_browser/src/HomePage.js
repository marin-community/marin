import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import Button from '@mui/material/Button';
import { apiConfigUrl, viewSingleUrl, navigateToUrl, renderError, checkJsonResponse } from "./utils";

function HomePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const urlParams = new URLSearchParams(location.search);

  // State
  const [error, setError] = useState(null);
  const [config, setConfig] = useState(null);

  // Fetch data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiConfigUrl());
        const payload = checkJsonResponse(response, setError);
        if (!payload) {
          return;
        }
        if (!payload || !Array.isArray(payload.root_paths)) {
          console.error("Invalid /api/config payload:", payload);
          setError("Backend config response is missing root_paths.");
          return;
        }
        setConfig(payload);
      } catch (error) {
        console.error(error);
        setError(error.message);
      }
    };
    fetchData();
  }, []);

  if (error) {
    return renderError(error);
  }

  if (!config) {
    return "Loading...";
  }

  const links = config.root_paths.map((path) => {
    return <a href={viewSingleUrl(path)} target="_blank" rel="noreferrer">{path}</a>;
  });

  function goToExperiment() {
    const path = prompt("Enter path to experiment JSON (e.g., gs://marin-us-central2/experiments/...):");
    if (!path) {
      return;
    }
    navigateToUrl(urlParams, {path}, {pathname: "/experiment"}, navigate);
  }

  return (
    <div>
      <h1>Data Browser</h1>
      <ul>
        {links.map((link, i) => <li key={i}>{link}</li>)}
      </ul>
      <Button variant="contained" onClick={goToExperiment}>Go to experiment</Button>
    </div>
  );
}

export default HomePage;
