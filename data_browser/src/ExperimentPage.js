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
  "WAITING": "🧍",
  "RUNNING": "🔄",
};
const loadingIcon = "⏳";

function ExperimentPage() {
  // Get URL parameters
  const location = useLocation();
  const urlParams = new URLSearchParams(location.search);

  // Path to the experiments JSON
  const path = urlParams.get("path");

  // State
  const [error, setError] = useState(null);
  const [experiment, setExperiment] = useState(null);
  const [auxiliaryData, setAuxiliaryData] = useState({});  // url -> contents (general file cache)

  // Fetch data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get the main experiment JSON
        const response = await axios.get(apiViewUrl({path}));
        const experiment = response.data.data;
        setExperiment(experiment);

        // Prefetch all the urls (e.g., for status, results)
        const urls = getAllPrefetchUrls(experiment);

        // Fetch the status (events) files for each step
        const promises = urls.map(async (url) => {
          try {
            const response = await axios.get(url);
            setAuxiliaryData(auxiliaryData => Object.assign({}, auxiliaryData, {[url]: response.data}));
          } catch (error) {
            console.error(error);
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

  return renderExperiment({experiment, path, auxiliaryData});
}

function getAllPretchUrls(experiment) {
  // Return a list of paths that we need to fetch async to render different aspects of the experiments
  const paths = [];
  experiment.steps.forEach((step) => {
    paths.push(apiStatusUrl(step));
    if (step.fn_name === "marin.evaluation.run.evaluate") {
      paths.push(apiResultsUrl(step));
    }
  });
  return paths;
}

export default ExperimentPage;

////////////////////////////////////////////////////////////

/**
 * Render information about an experiment.
 */
function renderExperiment({experiment, path, auxiliaryData}) {
  const header = renderExperimentHeader({experiment, path});
  const steps = renderExperimentSteps({experiment, auxiliaryData});

  return (<div>
    <Paper elevation={3} style={{padding: 10}}>{header}</Paper>
    <Paper elevation={3} style={{padding: 10}}>{steps}</Paper>
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

function renderExperimentSteps({experiment, auxiliaryData}) {
  const rows = [];
  experiment.steps.forEach((step, index) => {
    const row = [];

    row.push(<td className="experiment-step-table-cell" key="status">{renderExperimentStatus({step, auxiliaryData})}</td>);

    const info = <a href={viewInfoUrl(step)} target="_blank" title="View raw JSON specification of this step">{infoIcon}</a>;
    row.push(<td className="experiment-step-table-cell" key="info">{info}</td>);

    const stepName = <a href={viewOutputPathUrl(step)} target="_blank" title="View raw output path produced by this step">[{step.name}]</a>;
    row.push(<td className="experiment-step-table-cell" key="step-name">{stepName}</td>);

    row.push(<td className="experiment-step-table-cell" key="equals">:=</td>);

    const {name, description} = renderStepDescription({step, steps: experiment.steps, auxiliaryData});
    row.push(<td className="experiment-step-table-cell" key="name" title={step.fn_name}>{name}</td>);
    row.push(<td className="experiment-step-table-cell" key="description">{description}</td>);

    rows.push(<tr key={index} id={step.name}>{row}</tr>);
  });
  return (<table className="experiment-steps-table"><tbody>{rows}</tbody></table>);
}

function renderStepDescription({step, steps, auxiliaryData}) {
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

  if (step.fn_name === "operations.download.filesystem.transfer.transfer_files") {
    const description = renderPath({path: step.config.input_path, steps});
    return {name: "copy", description};
  }

  // Tokenize
  if (step.fn_name === "marin.processing.tokenize.tokenize.tokenize") {
    return renderTokenizeStepDescription({step, steps});
  }

  // Run inference (e.g., for quality filtering)
  if (step.fn_name === "marin.processing.classification.inference.run_inference") {
    const description = <table>
      <tbody>
        <tr><td>Model:</td><td>{step.config.model_type}</td></tr>
        <tr><td>Input:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
        <tr><td>Output attribute:</td><td>{step.config.attribute_name}</td></tr>
      </tbody>
    </table>;
    return {name: "run_inference", description};
  }

  if (step.fn_name === "marin.generation.inference.run_inference") {
    const description = <table>
      <tbody>
        <tr><td>Model:</td><td>{getBasename(step.config.model_name)}</td></tr>
        <tr><td>Prompt:</td><td>{step.config.template.substring(0, 80)}... <span title={step.config.template}>{infoIcon}</span></td></tr>
        <tr><td>Input:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
      </tbody>
    </table>;
    return {name: "run_inference", description};
  }

  if (step.fn_name === "marin.datashop.pipeline.run_medu_dataset_sampling_pipeline") {
    const description = <table>
      <tbody>
        <tr><td>Input:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
        <tr><td>Processor:</td><td>{step.config.processor_type}</td></tr>
      </tbody>
    </table>;
    return {name: "sample", description};
  }

  if (step.fn_name === "marin.classifiers.hf.launch_ray_training.launch_training_with_ray") {
    const resources = step.config.resource_config;
    const hardwareSummary = `${resources.num_tpu} * ${resources.tpu_type}`;
    const trainingConfig = step.config.training_config;
    const modelName = trainingConfig.model_name;
    const modelLink = <a href={`https://huggingface.co/${modelName}`} target="_blank">{huggingfaceIcon}{modelName}</a>;
    const description = <table>
      <tbody>
        <tr><td>Base model:</td><td>{modelLink}</td></tr>
        <tr><td>Input data:</td><td>{renderPath({path: trainingConfig.train_dataset, steps})}</td></tr>
        <tr><td>Hardware:</td><td>{hardwareSummary}</td></tr>
      </tbody>
    </table>;
    return {name: "train", description};
  }

  // Consolidate (e.g., filter)
  if (step.fn_name === "marin.processing.classification.consolidate.consolidate") {
    const filters = step.config.filters.map((filter) => {
      return <div key={filter.name}>{renderPath({path: filter.attribute_path, steps})}.{filter.name}.{filter.label} (keep {filter.keep_fraction})</div>;
    });
    const description = <table>
      <tbody>
        <tr><td>Input data:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
        <tr><td>Filters:</td><td>{filters}</td></tr>
      </tbody>
    </table>;
    return {name: "consolidate", description};
  }

  // Train
  if (step.fn_name === "marin.training.training.run_levanter_train_lm") {
    return renderTrainStepDescription({step, steps});
  }

  // Evaluate
  if (step.fn_name === "marin.evaluation.run.evaluate") {
    return renderEvaluateStepDescription({step, steps, auxiliaryData});
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
  const {step, replacedPath} = replacePath({path, steps});

  function onMouseEnter(step) {
    console.log("onMouseEnter", step);
    document.getElementById(step.name).classList.add("highlight");
  }

  function onMouseLeave(step) {
    console.log("onMouseLeave", step);
    document.getElementById(step.name).classList.remove("highlight");
  }

  const link = <a href={viewSingleUrl(linkedPath)} target="_blank"
      className={step && "path-link"}
      onMouseEnter={step && (() => onMouseEnter(step))}
      onMouseLeave={step && (() => onMouseLeave(step))}>
    {replacedPath}
  </a>;

  return link;
}

function replacePath({path, steps}) {
  // If the path is under an output path of some step, then link to the name of that step
  for (const step of steps) {
    if (path.startsWith(step.output_path)) {
      return {step, replacedPath: path.replace(step.output_path, `[${step.name}]`)};
    }
  }
  return {step: null, replacedPath: path};
}

function renderTokenizeStepDescription({step, steps}) {
  const tokenizer = step.config.tokenizer;
  const hfUrl = `https://huggingface.co/${tokenizer}/raw/main/tokenizer.json`;
  const paths = step.config.train_paths.concat(step.config.validation_paths);
  const links = paths.map((path, index) => {
    return <div key={index}>{renderPath({path, steps})}</div>;
  });
  const description = <table>
    <tbody>
      <tr><td>Tokenizer:</td><td><a href={hfUrl} target="_blank">{tokenizer}</a></td></tr>
      <tr><td>Raw text:</td><td>{links}</td></tr>
    </tbody>
  </table>;
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
    `lr = ${renderSciNotation(optimizerConfig.learning_rate)} → ${renderSciNotation(finalLearningRate)} (${optimizerConfig.lr_schedule})`,
    `warmup = ${optimizerConfig.warmup}`,
    `betas = ${optimizerConfig.beta1},${optimizerConfig.beta2}`,
    `weight_decay = ${optimizerConfig.weight_decay}`,
  ].join(", ");

  // Initialization + training
  const trainerConfig = step.config.train_config.trainer;
  const initializationSummary = trainerConfig.initialize_from ?
    renderPath({path: trainerConfig.initialize_from, steps}) :
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

  const wandbLink = <a href={getWandbUrl({step})} title="Go to the WandB page for this training run" target="_blank">[WandB {wandbIcon}]</a>;
  const description = (
    <table className="train-table">
      <tbody>
        <tr><td>Dataset:</td><td>{datasetSummary}</td></tr>
        <tr><td>Architecture:</td><td>{architectureSummary}</td></tr>
        <tr><td>Optimizer:</td><td>{optimizerSummary}</td></tr>
        <tr><td>Initialization:</td><td>{initializationSummary}</td></tr>
        <tr><td>Training:</td><td>{trainerSummary}</td></tr>
        <tr><td>Hardware:</td><td>{resourcesSummary}</td></tr>
        <tr><td>{wandbLink}</td></tr>
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

function renderEvaluateStepDescription({step, steps, auxiliaryData}) {
  const results = auxiliaryData[apiResultsUrl(step)];
  const resultsSummary = step.config.evals.map(({task_alias}) => {
    const score = results ? round(results.data.groups[task_alias]["acc,none"], 3) : loadingIcon;
    return <div key={task_alias}>{task_alias}: {score}</div>;
  });
  const description = <table>
    <tbody>
      <tr><td>Model:</td><td>{renderPath({path: step.config.model_path, steps})}</td></tr>
      <tr><td>Results:</td><td>{resultsSummary}</td></tr>
    </tbody>
  </table>;
  return {name: "evaluate", description};
}

function renderExperimentStatus({step, auxiliaryData}) {
  const data = auxiliaryData[apiStatusUrl(step)];
  if (!data) {
    return loadingIcon;
  }
  const events = data.items;

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

function pathJoin(path, file) {
  return path + (path.endsWith("/") ? "" : "/") + file;
}

function apiStatusUrl(step) {
  const statusPath = pathJoin(step.output_path, ".executor_status");
  return apiViewUrl({path: statusPath, count: 100});
}

function apiResultsUrl(step) {
  const resultsPath = pathJoin(step.output_path, "results.json");
  return apiViewUrl({path: resultsPath, count: 100});
}

function viewStatusUrl(step) {
  const statusPath = pathJoin(step.output_path, ".executor_status");
  return viewSingleUrl(statusPath);
}

function viewInfoUrl(step) {
  const infoPath = pathJoin(step.output_path, ".executor_info");
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
  return "??";
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

function getBasename(path) {
  return path.split("/").pop();
}
