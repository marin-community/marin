import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Paper from '@mui/material/Paper';
import { useLocation } from 'react-router-dom';
import { apiViewUrl, renderError, renderDuration, renderDate, viewSingleUrl, round, joinSpans, renderSciNotation, checkJsonResponse } from './utils';

const wandbIcon = "📉";
const huggingfaceIcon = "🤗";
const infoIcon = "ℹ️";
const statusIcons = {
  "SUCCESS": "✅",
  "FAILED": "❌",
  "WAITING": "🧍",
  "RUNNING": "🏃",
};
const loadingIcon = "⏳";
const errorIcon = "❌";

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
        const payload = checkJsonResponse(response, setError);
        if (!payload) {
          return;
        }
        const experiment = payload.data || payload;
        setExperiment(experiment);

        // Prefetch all the urls (e.g., for status, results)
        const urls = getAllPrefetchUrls(experiment);

        // Fetch the status (events) files for each step
        const promises = urls.map(async (url) => {
          try {
            const response = await axios.get(url);
            const payload = checkJsonResponse(response, setError);
            if (!payload) {
              return;
            }
            setAuxiliaryData(auxiliaryData => Object.assign({}, auxiliaryData, {[url]: payload}));
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

function getAllPrefetchUrls(experiment) {
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

/**
 * Renders the header (link to GitHub, etc.)
 */
function renderExperimentHeader({experiment, path}) {
  const relativePath = extractRayRelativePath(experiment.caller_path);

  const links = [];

  // Link to code on GitHub
  const githubUrl = experiment.git_commit ?
    `https://github.com/stanford-crfm/marin/tree/${experiment.git_commit}/${relativePath}` :
    `https://github.com/stanford-crfm/marin/blob/main/${relativePath}`;
  links.push(<Button href={githubUrl} color="primary" target="_blank" rel="noreferrer">Code</Button>);

  // Link to plain data browser
  links.push(<Button href={viewSingleUrl(path)} target="_blank" rel="noreferrer">JSON</Button>);

  return (<div>
    <h3>Experiment: {relativePath}</h3>
    <div className="experiment-step-line">Created: {renderDate(experiment.created_date)}</div>
    <div className="description">{experiment.description}</div>
    <ButtonGroup>
      {links.map((link, i) => <span key={i}>{link}</span>)}
    </ButtonGroup>
  </div>);
}

/**
 * Key function for rendering all the experiments.
 * Right now, we just treat the experiments as a long listof steps that get rendered as rows of a table.
 * Each row has:
 * - Links (status, info)
 * - Name of the step (note: not unique)
 * - Function (which is custom)
 * - Description (which includes both arguments and outputs)
 *
 * The description will (based on the function) refer to different paths, and we
 * automatically detect these as the output path for the step.
 *
 * Note: we are currently not looking at the explicit dependencies of each step!
 * So that means a hardcoded path will show up as a (implicit) dependency and
 * dependencies that are not part of the selected fields will not show up.
 */
function renderExperimentSteps({experiment, auxiliaryData}) {
  const rows = [];
  experiment.steps.forEach((step, index) => {
    const row = [];

    row.push(<td className="experiment-step-table-cell" key="status">{renderExperimentStatus({step, auxiliaryData})}</td>);

    const info = <a href={viewInfoUrl(step)} target="_blank" rel="noreferrer" title="View raw JSON specification of this step">{infoIcon}</a>;
    row.push(<td className="experiment-step-table-cell" key="info">{info}</td>);

    const stepName = <a href={viewOutputPathUrl(step)} target="_blank" rel="noreferrer" title={`View raw output path produced by this step:\n${step.output_path}`}>[{step.name}]</a>;
    row.push(<td className="experiment-step-table-cell" key="step-name">{stepName}</td>);

    row.push(<td className="experiment-step-table-cell" key="equals">:=</td>);

    try {
      const {name, description} = renderStepDescription({step, steps: experiment.steps, auxiliaryData});
      row.push(<td className="experiment-step-table-cell" key="name" title={step.fn_name}>{name}</td>);
      row.push(<td className="experiment-step-table-cell" key="description">{description}</td>);
    } catch (error) {
      console.error(error);
      row.push(<td className="experiment-step-table-cell" key="name" title={step.fn_name}>{step.fn_name}</td>);
      row.push(<td className="experiment-step-table-cell" key="description"><span className="error">{error.message}</span></td>);
    }

    rows.push(<tr key={index} id={step.output_path}>{row}</tr>);
  });
  return (<table className="experiment-steps-table"><tbody>{rows}</tbody></table>);
}

function renderDownloadStep({step}) {
  const hfDatasetId = step.config.hf_dataset_id;
  const revision = step.config.revision;
  const hfUrl = `https://huggingface.co/datasets/${hfDatasetId}/tree/${revision}`;
  return {name: "download", description: <a href={hfUrl} target="_blank" rel="noreferrer">{huggingfaceIcon}{hfDatasetId}</a>};
}

function renderDownloadNemotronCCStep({step}) {
  const url = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html";
  const description = "Nemotron-CC from Common Crawl";
  return {name: "download", description: <a href={url} target="_blank" rel="noreferrer">{description}</a>};
}

function renderTransferStep({step, steps}) {
  const description = renderPath({path: step.config.input_path, steps});
  return {name: "copy", description};
}

function renderRaw2JsonStep({step, steps}) {
  const description = renderPath({path: step.config.input_path, steps});
  return {name: "raw2json", description};
}

function renderFastTextTransformStep({step, steps}) {
  const description = renderPath({path: step.config.input_path, steps});
  return {name: "fasttext2json", description};
}

function renderConvertEvalToDolmaStep({step, steps}) {
  const description = <table>
    <tbody>
        <tr><td>Input:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
    </tbody>
  </table>;
  return {name: "eval2dolma", description};
}

function renderRunClassificationInferenceStep({step, steps}) {
  const description = <table>
    <tbody>
      <tr><td>Model:</td><td>{step.config.model_type}</td></tr>
      <tr><td>Input data:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
      <tr><td>Output attribute:</td><td>{step.config.attribute_name}</td></tr>
    </tbody>
  </table>;
  return {name: "run_inference", description};
}

function renderRunGenerationInferenceStep({step, steps}) {
  const description = <table>
    <tbody>
      <tr><td>Model:</td><td>{getBasename(step.config.model_name)}</td></tr>
      <tr><td>Prompt:</td><td>{step.config.template.substring(0, 80)}... <span title={step.config.template}>{infoIcon}</span></td></tr>
      <tr><td>Input data:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
    </tbody>
  </table>;
  return {name: "run_inference", description};
}

function renderMeduSampleStep({step, steps}) {
  const description = <table>
    <tbody>
      <tr><td>Input data:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
      <tr><td>Processor:</td><td>{step.config.processor_type}</td></tr>
    </tbody>
  </table>;
  return {name: "sample", description};
}

function renderTrainClassifierStep({step, steps}) {
  const rows = step.config.datasets.map((dataset, index) => {
    return <tr key={index}>
      <td>{renderPath({path: dataset.input_doc_path, steps})}</td>
      <td>{dataset.label}</td>
    </tr>;
  });
  const description = <table>
    <tbody>
      {rows}
    </tbody>
  </table>;
  return {name: "train_classifier", description};
}

function renderHfLaunchTrainingStep({step, steps}) {
  const resources = step.config.resource_config;
  const hardwareSummary = summarizeResources(resources);
  const trainingConfig = step.config.training_config;
  const modelName = trainingConfig.model_name;
  const modelLink = <a href={`https://huggingface.co/${modelName}`} target="_blank" rel="noreferrer">{huggingfaceIcon}{modelName}</a>;
  const description = <table>
    <tbody>
      <tr><td>Base model:</td><td>{modelLink}</td></tr>
      <tr><td>Input data:</td><td>{renderPath({path: trainingConfig.train_dataset, steps})}</td></tr>
      <tr><td>Hardware:</td><td>{hardwareSummary}</td></tr>
    </tbody>
  </table>;
  return {name: "train_lm", description};
}

function renderConsolidateStep({step, steps}) {
  const filters = step.config.filters.map((filter) => {
    return <div key={filter.name}>{renderPath({path: filter.attribute_path, steps})}.{filter.label} (keep {filter.keep_fraction})</div>;
  });
  const description = <table>
    <tbody>
      <tr><td>Input data:</td><td>{renderPath({path: step.config.input_path, steps})}</td></tr>
      <tr><td>Filters:</td><td>{filters}</td></tr>
    </tbody>
  </table>;
  return {name: "consolidate", description};
}

function renderTokenizeStepDescription({step, steps}) {
  const tokenizer = step.config.tokenizer;
  const tokenizerUrl = tokenizer ? `https://huggingface.co/${tokenizer}/raw/main/tokenizer.json` : null;
  const trainPaths = Array.isArray(step.config.train_paths) ? step.config.train_paths : [];
  const validationPaths = Array.isArray(step.config.validation_paths) ? step.config.validation_paths : [];
  const localPaths = trainPaths.concat(validationPaths);

  let rawSourceDescription;
  if (localPaths.length > 0) {
    rawSourceDescription = localPaths.map((path, index) => <div key={index}>{renderPath({path, steps})}</div>);
  } else if (step.config.id) {
    const datasetId = step.config.id;
    const hfDatasetUrl = `https://huggingface.co/datasets/${datasetId}`;
    const revision = step.config.revision ? `@${step.config.revision}` : "";
    rawSourceDescription = <a href={hfDatasetUrl} target="_blank" rel="noreferrer">{huggingfaceIcon}{datasetId}{revision}</a>;
  } else if (step.config.dataset) {
    rawSourceDescription = step.config.dataset;
  } else {
    rawSourceDescription = "See step config";
  }

  const rows = [
    <tr key="tokenizer">
      <td>Tokenizer:</td>
      <td>{tokenizerUrl ? <a href={tokenizerUrl} target="_blank" rel="noreferrer">{tokenizer}</a> : tokenizer}</td>
    </tr>,
    <tr key="raw">
      <td>Raw text:</td>
      <td>{rawSourceDescription}</td>
    </tr>,
  ];

  if (step.config.sample_count) {
    rows.push(<tr key="sample-count"><td>Sample count:</td><td>{step.config.sample_count}</td></tr>);
  }

  return {
    name: "tokenize",
    description: <table><tbody>{rows}</tbody></table>,
  };
}

function renderTrainLmStep({step, steps}) {
  // Old runs don't have a train_config, it's just config
  const trainConfig = step.config.train_config || step.config.config || step.config;
  const dataConfig = trainConfig.data;
  const datasetSummary = renderDatasetSummary({dataConfig, steps});

  // Model
  const modelConfig = trainConfig.model;
  const architectureSummary = [
    modelConfig.n_routed_experts && `${modelConfig.n_routed_experts} experts (${modelConfig.num_experts_per_tok} active) + ${modelConfig.n_shared_experts} shared experts`,
    `d_model = ${modelConfig.hidden_dim}`,
    `d_ff = ${modelConfig.intermediate_dim}`,
    `n_heads = ${modelConfig.num_heads} (${modelConfig.num_kv_heads} kv)`,
    `n_layers = ${modelConfig.num_layers}`,
    `seq_len = ${modelConfig.seq_len}`,
    `rope(${modelConfig.rope.factor},${modelConfig.rope.theta})`,
    modelConfig.activation_function,
  ].filter((x) => x != null).join(", ");

  // Optimizer
  const optimizerConfig = trainConfig.optimizer;
  const finalLearningRate = optimizerConfig.learning_rate * optimizerConfig.min_lr_ratio;
  const optimizerSummary = [
    `lr = ${renderSciNotation(optimizerConfig.learning_rate)} → ${renderSciNotation(finalLearningRate)} (${optimizerConfig.lr_schedule})`,
    `warmup = ${optimizerConfig.warmup}`,
    `betas = ${optimizerConfig.beta1},${optimizerConfig.beta2}`,
    `weight_decay = ${optimizerConfig.weight_decay}`,
  ].join(", ");

  // Initialization + training
  const trainerConfig = trainConfig.trainer;
  const initializeFrom = trainConfig.initialize_from_checkpoint || trainConfig.initialize_from_hf || trainerConfig.initialize_from;
  const initializationSummary = initializeFrom ?
    renderPath({path: initializeFrom, steps}) :
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
  const resourcesSummary = summarizeResources(resources);

  const wandbLink = <a href={getWandbUrl({step})} title="Go to the WandB page for this training run" target="_blank" rel="noreferrer">[WandB {wandbIcon}]</a>;
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

  function getUrl(location) {
    return location.cache_dir || location.train_urls[0];
  }

  if (nonzeroSources.length === 1) {
    return renderPath({path: getUrl(nonzeroSources[0][1]), steps});
  }

  nonzeroSources.forEach(([source, location]) => {
    const row = [<td key="source">{renderPath({path: getUrl(location), steps})}</td>].concat(
      allWeights.map(([step, weights], stage) => <td key={step}>{round(weights[source] / sumWeights[stage], 3)}</td>)
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

function renderEvaluateStep({step, steps, auxiliaryData}) {
  const results = auxiliaryData[apiResultsUrl(step)];
  function getScore(key) {
    const score = results.data.results?.[key]?.["acc,none"];
    return score ? round(score, 3) : "n/a";
  }
  const rows = step.config.evals.map(({name, task_alias}, index) => {
    const key = task_alias || name;  // task_alias = hellaswag_0shot, name = hellaswag
    return <tr key={index}>
      <td>{key}</td>
      <td>{results?.data ? getScore(key) : loadingIcon}</td>
    </tr>;
  })
  const resultsSummary = <table>
    <tbody>
      {rows}
    </tbody>
  </table>;
  const description = <table>
    <tbody>
      <tr><td>Model:</td><td>{renderPath({path: step.config.model_path, steps})}</td></tr>
      <tr><td><a href={viewResultsUrl(step)} target="_blank" rel="noreferrer">Results</a>:</td><td>{resultsSummary}</td></tr>
    </tbody>
  </table>;
  return {name: "evaluate", description};
}

function renderExperimentStatus({step, auxiliaryData}) {
  const data = auxiliaryData[apiStatusUrl(step)];
  if (!data) {
    return loadingIcon;
  }
  if (data.error) {
    return <span className="error" title={data.error}>{errorIcon}</span>;
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
    <a href={viewStatusUrl(step)} target="_blank" rel="noreferrer" title={`View raw JSON status of this step ${temporalInfo}`}>
      {lastStatus}
    </a>
  </span>;
}

const stepRenderers = {
  // Download
  "marin.download.huggingface.download.download": renderDownloadStep,
  "marin.download.huggingface.download_hf.download_hf": renderDownloadStep,
  "marin.download.huggingface.download_gated_manual.download_and_upload_to_store": renderDownloadStep,
  "marin.download.nemotron_cc.download_nemotron_cc.download_nemotron_cc": renderDownloadNemotronCCStep,
  "marin.download.filesystem.transfer.transfer_files": renderTransferStep,
  "marin.raw2json.huggingface.qa.raw2json.raw2json": renderRaw2JsonStep,
  "marin.transform.conversation.transform_conversation.transform_hf_dataset": renderRaw2JsonStep,
  "marin.transform.fasttext.transform.main": renderFastTextTransformStep,
  "marin.transform.evaluation.eval_to_dolma.convert_eval_to_dolma": renderConvertEvalToDolmaStep,

  // Inference for data filtering
  "marin.processing.classification.inference.run_inference": renderRunClassificationInferenceStep,
  "marin.generation.inference.run_inference": renderRunGenerationInferenceStep,
  "marin.datashop.pipeline.run_medu_dataset_sampling_pipeline": renderMeduSampleStep,
  "marin.processing.classification.fasttext.train_fasttext.train": renderTrainClassifierStep,
  "marin.processing.classification.bert.train_bert.train": renderTrainClassifierStep,
  "marin.classifiers.hf.launch_ray_training.launch_training_with_ray": renderHfLaunchTrainingStep,

  "marin.processing.classification.consolidate.consolidate": renderConsolidateStep,

  // Tokenize, train, eval
  "marin.processing.tokenize.tokenize.tokenize": renderTokenizeStepDescription,
  "marin.training.training.run_levanter_train_lm": renderTrainLmStep,
  "marin.evaluation.run.evaluate": renderEvaluateStep,
};

function renderStepDescription({step, steps, auxiliaryData}) {
  const renderer = stepRenderers[step.fn_name];
  if (renderer) {
    return renderer({step, steps, auxiliaryData});
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

  // What path to show
  const {step, replacedPath} = replacePath({path, steps});

  function onMouseEnter(step) {
    document.getElementById(step.output_path).classList.add("highlight");
  }

  function onMouseLeave(step) {
    document.getElementById(step.output_path).classList.remove("highlight");
  }

  const link = <a href={viewSingleUrl(linkedPath)} target="_blank" rel="noreferrer"
      className={step && "path-link"}
      title={`View raw path:\n${linkedPath}`}
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

function viewResultsUrl(step) {
  const resultsPath = pathJoin(step.output_path, "results.json");
  return viewSingleUrl(resultsPath);
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

function summarizeResources(resources) {
  if (!resources) {
    return "n/a";
  }

  if (resources.tpu_type) {
    const slices = resources.slice_count ?? resources.num_slices ?? resources.num_tpu ?? 1;
    return `${slices} x ${resources.tpu_type}`;
  }

  if (resources.gpu_type || resources.num_gpus || resources.gpu_count) {
    const count = resources.num_gpus ?? resources.gpu_count ?? resources.gpu_per_node ?? 1;
    const gpuType = resources.gpu_type || resources.accelerator_type || "GPU";
    return `${count} x ${gpuType}`;
  }

  if (resources.num_cpus) {
    const count = resources.num_cpus;
    const suffix = count === 1 ? "" : "s";
    return `${count} CPU${suffix}`;
  }

  if (resources.instance_type) {
    return resources.instance_type;
  }

  return "n/a";
}
