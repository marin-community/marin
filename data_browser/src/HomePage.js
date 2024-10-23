import React from "react";
import { useLocation, useNavigate, Link } from 'react-router-dom';
import Button from '@mui/material/Button';
import { viewSingleUrl, navigateToUrl } from "./utils";

function HomePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const urlParams = new URLSearchParams(location.search);

  const rootPaths = [
    "gs://marin-us-central2",
    "gs://marin-us-west4",
    "gs://marin-eu-west4",
  ];

  const links = rootPaths.map((path) => {
    return <Link to={viewSingleUrl(path)}>{path}</Link>;
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
