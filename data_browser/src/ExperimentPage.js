import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Paper from '@mui/material/Paper';
import { useLocation } from 'react-router-dom';
import { apiViewUrl, renderError, renderDuration, renderDate, viewSingleUrl, round, joinSpans, renderSciNotation } from './utils';

const wandbIcon = "📉";
const huggingfaceIcon = "🤗";
const infoIcon = "ℹ️";
const statusIcons = {
  "SUCCESS": "✅",
  "FAILED": "❌",
  "WAITING": "⏳",
  "RUNNING": "🔄",
};

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
  const [files, setFiles] = useState({});  // path -> contents (general file cache)

  // Fetch data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get the main experiment JSON
        const response = await axios.get(apiViewUrl({path}));
        const experiment = response.data.data;
        setExperiment(experiment);

        // Fetch the status (events) files for each step
        const statusPromises = experiment.steps.map(async (step, index) => {
          try {
            const response = await axios.get(apiStatusUrl(step));
            const events = response.data.items;
            setStatuses(statuses => Object.assign({}, statuses, {[step.output_path]: events}));
          } catch (error) {
            console.error(error);
            const events = [{status: "ERROR", message: error.message}];
            setStatuses(statuses => Object.assign({}, statuses, {[step.output_path]: events}));
          }
        });

        await Promise.all(statusPromises);
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

/**
 * Render information about an experiment.
 */
function renderExperiment({experiment, path, statuses}) {
  const header = renderExperimentHeader({experiment, path});
  const steps = renderExperimentSteps({experiment, statuses});

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
  const githubUrl = experiment.git_commit ?
    `https://github.com/stanford-crfm/marin/tree/${experiment.git_commit}/${relativePath}` :
    `https://github.com/stanford-crfm/marin/blob/main/${relativePath}`;
  links.push(<Button href={githubUrl} color="primary" target="_blank">Code</Button>);

  // Link to plain data browser
  links.push(<Button href={viewSingleUrl(path)} target="_blank">JSON</Button>);

  return (<div>
    <h3>Experiment: {relativePath}</h3>
    <div className="experiment-step-line">Created: {renderDate(experiment.created_date)}</div>
    <div className="description">{experiment.description}</div>
    <ButtonGroup>
      {links.map((link, i) => <span key={i}>{link}</span>)}
    </ButtonGroup>
  </div>);
}

function renderExperimentSteps({experiment, statuses}) {
  const rows = [];
  experiment.steps.forEach((step, index) => {
    const row = [];

    const events = statuses[step.output_path];
    row.push(<td className="experiment-step-table-cell" key="status">{renderExperimentStatus({step, events})}</td>);

    const info = <a href={viewInfoUrl(step)} target="_blank" title="View raw JSON specification of this step">{infoIcon}</a>;
    row.push(<td className="experiment-step-table-cell" key="info">{info}</td>);

    const stepName = <a href={viewOutputPathUrl(step)} target="_blank" title="View raw output path produced by this step">[{step.name}]</a>;
    row.push(<td className="experiment-step-table-cell" key="step-name">{stepName}</td>);

    row.push(<td className="experiment-step-table-cell" key="equals">:=</td>);

    const {name, description} = renderStepDescription({step, steps: experiment.steps});
    row.push(<td className="experiment-step-table-cell" key="name">{name}</td>);
    row.push(<td className="experiment-step-table-cell" key="description">{description}</td>);

    rows.push(<tr key={index}>{row}</tr>);
  });
  return (<table className="experiment-steps-table"><tbody>{rows}</tbody></table>);
}

function renderStepDescription({step, steps}) {
  // Downloading datasets
  if (step.fn_name === "operations.download.huggingface.download.download" ||
      step.fn_name === "operations.download.huggingface.download_hf.download_hf" || 
      step.fn_name === "operations.download.huggingface.download_gated_manual.download_and_upload_to_store") {
    const hfDatasetId = step.config.hf_dataset_id;
    const revision = step.config.revision;
    const hfUrl = `https://huggingface.co/datasets/${hfDatasetId}/tree/${revision}`;
    return {name: "download", description: <a href={hfUrl} target="_blank">{huggingfaceIcon}{hfDatasetId}</a>};
  }
  if (step.fn_name === "operations.download.nemotron_cc.download_nemotron_cc.download_nemotron_cc") {
    const url = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html";
    const description = "Nemotron-CC from Common Crawl";
    return {name: "download", description: <a href={url} target="_blank">{description}</a>};
  }

  // Tokenize
  if (step.fn_name === "marin.processing.tokenize.tokenize.tokenize") {
    return renderTokenizeStepDescription({step, steps});
  }

  // Train
  if (step.fn_name === "marin.training.training.run_levanter_train_lm") {
    return renderTrainStepDescription({step, steps});
  }

  // Evaluate
  if (step.fn_name === "marin.evaluation.run.evaluate") {
    return renderEvaluateStepDescription({step, steps});
  }

  return {name: step.fn_name, description: step.description};
}

function renderPath({path, steps}) {
  // If path contains a pattern (e.g., gs://marin-us-central2/.../val*.jsonl.gz), strip those out
  // This is what we link to since the data browser can't handle patterns
  let linkedPath = path;
  while (linkedPath.includes("*") || linkedPath.includes("{")) {
    // Go up to the parent
    const basename = linkedPath.split("/").pop();
    linkedPath = linkedPath.substring(0, linkedPath.length - basename.length - 1);
  }

  // What to show
  const replacedPath = replacePath({path, steps});

  return <a href={viewSingleUrl(linkedPath)} target="_blank">{replacedPath}</a>;
}
    
function replacePath({path, steps}) {
  // If the path is under an output path of some step, then link to the name of that step
  for (const step of steps) {
    if (!path || !step.output_path)
      console.log(path, step);
    if (path.startsWith(step.output_path)) {
      return path.replace(step.output_path, `[${step.name}]`);
    }
  }
  return path;
}

function renderTokenizeStepDescription({step, steps}) {
  const tokenizer = step.config.tokenizer;
  const hfUrl = `https://huggingface.co/${tokenizer}/raw/main/tokenizer.json`;
  const paths = step.config.train_paths.concat(step.config.validation_paths);
  const links = paths.map((path, index) => {
    return <div key={index}>{renderPath({path, steps})}</div>;
  });
  const description = <span>
    <a href={hfUrl} target="_blank">{tokenizer}</a>, <span>{links}</span>
  </span>;
  return {name: "tokenize", description};
}

function renderTrainStepDescription({step, steps}) {
  const dataConfig = step.config.train_config.data;
  const datasetSummary = renderDatasetSummary({dataConfig, steps});

  // Model
  const modelConfig = step.config.train_config.model;
  const architectureSummary = [
    `d_model = ${modelConfig.hidden_dim}`,
    `d_ff = ${modelConfig.intermediate_dim}`,
    `n_heads = ${modelConfig.num_heads} (${modelConfig.num_kv_heads} kv)`,
    `n_layers = ${modelConfig.num_layers}`,
    `seq_len = ${modelConfig.seq_len}`,
    `rope(${modelConfig.rope.factor},${modelConfig.rope.theta})`,
    modelConfig.activation_function,
  ].join(", ");

  // Optimizer
  const optimizerConfig = step.config.train_config.optimizer;
  const finalLearningRate = optimizerConfig.learning_rate * optimizerConfig.min_lr_ratio;
  const optimizerSummary = [
    `betas = ${optimizerConfig.beta1},${optimizerConfig.beta2}`,
    `warmup = ${optimizerConfig.warmup}`,
    `lr = ${renderSciNotation(optimizerConfig.learning_rate)} → ${renderSciNotation(finalLearningRate)} (${optimizerConfig.lr_schedule})`,
    `weight_decay = ${optimizerConfig.weight_decay}`,
  ].join(", ");

  // Initialization + training
  const trainerConfig = step.config.train_config.trainer;
  const initializationSummary = trainerConfig.initalize_from ?
    viewSingleUrl(trainerConfig.initialize_from) :
    `random(${modelConfig.initializer_range})`;

  const numSteps = trainerConfig.num_train_steps;
  const batchSize = trainerConfig.train_batch_size;
  const seqLen = modelConfig.seq_len;
  const numTokens = computeNumTokens({numSteps, batchSize, seqLen});
  const trainerSummary = <span>
    {Math.round(numSteps)} (steps) * {renderStagedValue(batchSize)} (batch_size) * {modelConfig.seq_len} (seq_len) = {renderSciNotation(numTokens)} (tokens)
  </span>;

  // Hardware
  const resources = step.config.resources;
  const resourcesSummary = `${resources.slice_count || 1} * ${resources.tpu_type}`;

  const wandbLink = <a href={getWandbUrl({step})} title="Go to the WandB page for this training run" target="_blank">{wandbIcon}</a>;
  const description = (
    <table className="train-table">
      <tbody>
        <tr><td>Dataset</td><td>{datasetSummary}</td></tr>
        <tr><td>Architecture</td><td>{architectureSummary}</td></tr>
        <tr><td>Optimizer</td><td>{optimizerSummary}</td></tr>
        <tr><td>Initialization</td><td>{initializationSummary}</td></tr>
        <tr><td>Training {wandbLink}</td><td>{trainerSummary}</td></tr>
        <tr><td>Hardware</td><td>{resourcesSummary}</td></tr>
      </tbody>
    </table>
  );
  return {name: "train", description: description};
}

function renderDatasetSummary({dataConfig, steps}) {
  // sources: {source: {train_urls: [url1, url2, ...], ...}}
  // weights:
  // - {source: weight}
  // - [[step, {source:weight}], [step, {source:weight}], ...]
  const sources = dataConfig.configs;
  const allWeights = Array.isArray(dataConfig.train_weights) ? dataConfig.train_weights : [[0, dataConfig.train_weights]];
  // To normalize the weights
  const sumWeights = allWeights.map(([step, weights]) => Object.values(weights).reduce((a, b) => a + b, 0));

  const datasetRows = [];

  // Header
  if (allWeights.length > 1) {
    const header = [<td key="source">Source \ Step</td>].concat(
      allWeights.map(([step, weights], stage) => <td key={step}>{step}</td>)
    );
    datasetRows.push(<tr key="header">{header}</tr>);
  }

  const nonzeroSources = Object.entries(sources).filter(([source, location]) => {
    // This happens for validation sets (listed, but not trained on)
    if (allWeights.every(([step, weights]) => weights[source] === 0)) {
      return false;
    }
    return true;
  });

  if (nonzeroSources.length === 1) {
    return renderPath({path: nonzeroSources[0][1].train_urls[0], steps});
  }

  nonzeroSources.forEach(([source, location]) => {
    const row = [<td key="source">{renderPath({path: location.train_urls[0], steps})}</td>].concat(
      allWeights.map(([step, weights], stage) => <td key={step}>{round(weights[source] / sumWeights[stage], 2)}</td>)
    );
    datasetRows.push(<tr key={source}>{row}</tr>);
  });
  return <table className="dataset-table"><tbody>{datasetRows}</tbody></table>;
}

function getWandbUrl({step}) {
  // Link to the wandb page (if it's a training run) - by inferring it
  const name = step.output_path.split("/").pop();
  const wandbUrl = `https://wandb.ai/marin-community/marin/runs/${name}`;
  return wandbUrl;
}

function renderEvaluateStepDescription({step, steps}) {
  const evalsSummary = step.config.evals.map(({task_alias}) => {
    const metric = 0.5;
    return <span>{task_alias}: {metric}</span>;
  });
  const description = joinSpans([
    renderPath({path: step.config.model_path, steps}),
    evalsSummary,
  ], ", ");
  return {name: "evaluate", description};
}

function renderExperimentStatus({step, events}) {
  if (!events) {
    return "??";
  }

  const lastEvent = events[events.length - 1];

  // If last event is RUNNING, use the current time as the end time
  // Otherwise, use the time of the last event
  const startTime = new Date(events[0].date);
  const endTime = ["WAITING", "RUNNING"].includes(lastEvent.status) ? new Date() : new Date(lastEvent.date);

  const duration = (endTime.getTime() - startTime.getTime()) / 1000;

  const statusIcon = statusIcons[lastEvent.status] || lastEvent.status;

  const lastStatus = <span className={"status-" + lastEvent.status}>{statusIcon}</span>;
  const temporalInfo = `last updated ${renderDate(lastEvent.date)}, lifetime is ${renderDuration(duration)}`;
  return <span className="status-container">
    <a href={viewStatusUrl(step)} target="_blank" title={`View raw JSON status of this step ${temporalInfo}`}>
      {lastStatus}
    </a>
  </span>;
}

function apiStatusUrl(step) {
  const statusPath = step.output_path + (step.output_path.endsWith("/") ? "" : "/") + ".executor_status";
  return apiViewUrl({path: statusPath, count: 100});
}

function apiResultsUrl(step) {
  const resultsPath = step.output_path + (step.output_path.endsWith("/") ? "" : "/") + ".results.json";
  return apiViewUrl({path: resultsPath, count: 100});
}

function viewStatusUrl(step) {
  const statusPath = step.output_path + (step.output_path.endsWith("/") ? "" : "/") + ".executor_status";
  return viewSingleUrl(statusPath);
}
          
function viewInfoUrl(step) {
  const infoPath = step.output_path + (step.output_path.endsWith("/") ? "" : "/") + ".executor_info";
  return viewSingleUrl(infoPath);
}
          
function viewOutputPathUrl(step) {
  return viewSingleUrl(step.output_path);
}

function extractRayRelativePath(path) {
  // Given caller path, extract the relative path
  // Input: /tmp/ray/session_2024-10-17_20-58-29_674266_488/runtime_resources/working_dir_files/_ray_pkg_38803023dcc3288a/experiments/scratch.py
  // Output: experiments/scratch.py
  return path.replace(/.*_ray_pkg_\w+\//, "");
}

function renderStagedValue(valueOrValues) {
  // valueOrValues could be a number or [{start, value}, {start, value}, ...]
  // e.g., for batch size
  if (typeof valueOrValues === "number") {
    return valueOrValues;
  }
  if (Array.isArray(valueOrValues)) {
    const content = joinSpans(valueOrValues.map((item) => <span><sub>{item.start}</sub>{item.value}</span>), "→");
    return <span>[{content}]</span>;
  }
  return "???";
}

function computeNumTokens({numSteps, batchSize, seqLen}) {
  // batchSize could be a number or [{start, value}, {start, value}, ...]
  if (Array.isArray(batchSize)) {
    let total = 0;
    for (let i = 0; i < batchSize.length; i++) {
      // Go through the stages and compute how many steps are in each stage
      const start = batchSize[i].start;
      const end = i + 1 < batchSize.length ? batchSize[i + 1].start : numSteps;
      total += (end - start) * batchSize[i].value * seqLen;
    }
    return total;
  }
  return numSteps * batchSize * seqLen;
}
