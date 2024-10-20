import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import { apiViewUrl, renderError, renderDuration, renderDate, viewSingleUrl } from './utils';

function ExperimentPage() {
  // Get URL parameters
  const location = useLocation();
  const urlParams = new URLSearchParams(location.search);

  // Path to the experiments JSON
  const path = urlParams.get("path");

  // State
  const [error, setError] = useState(null);
  const [experiment, setExperiment] = useState(null);
  const [statuses, setStatuses] = useState({});  // output path -> status (list of events)

  // Fetch data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get the main experiment JSON
        const response = await axios.get(apiViewUrl({path}));
        const experiment = response.data.data;
        setExperiment(experiment);

        // Fetch the status (events) files for each step
        const promises = experiment.steps.map(async (step, index) => {
          const statusPath = step.output_path + "/.executor_status";
          try {
            const response = await axios.get(apiViewUrl({path: statusPath, count: 100}));
            const events = response.data.items;
            setStatuses(statuses => Object.assign({}, statuses, {[step.output_path]: events}));
          } catch (error) {
            console.error(error);
            const events = [{status: "ERROR", message: error.message}];
            setStatuses(statuses => Object.assign({}, statuses, {[step.output_path]: events}));
          }
        });
        await Promise.all(promises);
      } catch (error) {
        console.error(error);
        setError(error.message);
      }
    };
    fetchData();
  }, [location, path]);

  if (error) {
    return renderError(error);
  }

  if (!experiment) {
    return "Loading...";
  }

  return renderExperiment({experiment, path, statuses});
}

export default ExperimentPage;

////////////////////////////////////////////////////////////

function extractRayRelativePath(path) {
  // Given caller path, extract the relative path
  // Input: /tmp/ray/session_2024-10-17_20-58-29_674266_488/runtime_resources/working_dir_files/_ray_pkg_38803023dcc3288a/experiments/scratch.py
  // Output: experiments/scratch.py
  return path.replace(/.*_ray_pkg_\w+\//, "");
}

/**
 * Render information about an experiment.
 */
function renderExperiment(args) {
  const {experiment, path, statuses} = args;

  const header = renderExperimentHeader({experiment, path});

  const steps = experiment.steps.map((_, index) => {
    return renderExperimentStep({experiment, statuses, index});
  });

  return (<div>
    {header}
    {steps}
  </div>);
}

function renderExperimentHeader(args) {
  const {experiment, path} = args;
  const relativePath = extractRayRelativePath(experiment.caller_path);

  const links = [];
  
  // Link to code on GitHub
  const githubUrl = "https://github.com/stanford-crfm/marin/blob/main/" + relativePath;
  links.push(<a href={githubUrl} target="_blank" rel="noreferrer">[GitHub]</a>);

  // Link to plain data browser
  links.push(<a href={viewSingleUrl(path)} target="_blank" rel="noreferrer">[JSON]</a>);

  // Link to Ray job
  const rayUrl = `http://localhost:8265/#/jobs/${experiment.ray_job_id}`;
  links.push(<a href={rayUrl} target="_blank" rel="noreferrer">[Ray]</a>);

  return (<div>
    <h3>Experiment: {relativePath}</h3>
    <div>Created: {renderDate(experiment.created_date)}</div>
    {links.map((link, i) => <span key={i}>{link}</span>)}
  </div>);
}

function renderExperimentStep(args) {
  const {experiment, statuses, index} = args;

  const step = experiment.steps[index];
  const events = statuses[step.output_path];

  const links = [];

  // Link to the JSON with the information (including config)
  const infoUrl = viewSingleUrl(step.output_path + "/.executor_info");
  links.push(<a href={infoUrl} target="_blank" rel="noreferrer">[info]</a>);

  // Link to the Ray task
  const rayTaskId = events && events[events.length - 1].ray_task_id;
  if (rayTaskId) {
    const rayUrl = `http://localhost:8265/#/jobs/${experiment.ray_job_id}/tasks/${rayTaskId}`;
    links.push(<a href={rayUrl} target="_blank" rel="noreferrer">[Ray]</a>);
  }

  // Link to the wandb page (if it's a training run)
  if (step.name.startsWith("checkpoints")) {  // Heuristically guess if this is a training run
    const name = step.output_path.split("/").pop();
    const wandbUrl = `https://wandb.ai/stanford-mercury/marin/runs/${name}`;
    links.push(<a href={wandbUrl} target="_blank" rel="noreferrer">[wandb]</a>);
  }

  const configRows = Object.entries(step.version.config).map(([key, value]) => {
    return <tr key={key}><td>{key}</td><td>: {value}</td></tr>;
  });

  // Map dependencies (represented by output paths) to the corresponding step indices
  const outputPaths = experiment.steps.map(step => step.output_path);
  const dependencies = step.dependencies.map(dep => `[${outputPaths.indexOf(dep)}]`).join(", ");

  return (<div key={index} className="experiment-step">
    <div>[{index}] <a href={viewSingleUrl(step.output_path)}>{step.output_path}</a> := {step.fn_name}({dependencies})</div>
    <table><tbody>{configRows}</tbody></table>
    {renderExperimentStatus(events)}
    {links.map((link, i) => <span key={i}>{link}</span>)}
  </div>);
}

function renderExperimentStatus(events) {
  if (!events) {
    return "[Loading status]";
  }

  const lastEvent = events[events.length - 1];

  // If last event is RUNNING, use the current time as the end time
  // Otherwise, use the time of the last event
  const startTime = new Date(events[0].date);
  const endTime = ["WAITING", "RUNNING"].includes(lastEvent.status) ? new Date() : new Date(lastEvent.date);

  const duration = (endTime.getTime() - startTime.getTime()) / 1000;

  const lastStatus = lastEvent.status;
  const lastMessage = lastEvent.message;
  return <div>Status: {lastStatus}{lastMessage ? ": " + lastMessage : ""} {renderDate(lastEvent.date)} &mdash; {renderDuration(duration)}</div>;
}