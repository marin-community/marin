import React from "react";
import { Link } from "react-router-dom";
import { viewSingleUrl } from "./utils";

function HomePage() {
  const rootPaths = [
    "gs://marin-us-central2",
    "gs://marin-us-west4",
    "gs://marin-eu-west4",
  ];

  const links = rootPaths.map((path) => {
    return <Link to={viewSingleUrl(path)}>{path}</Link>;
  });

  return (
    <div>
      <h1>Data Browser</h1>
      <ul>
        {links.map((link, i) => <li key={i}>{link}</li>)}
      </ul>
    </div>
  );
}

export default HomePage;
