# Client configs

Client configs are json yaml that map model names to a list of configurations for instantiating a client such as the OpenAI one. 
We use a list to allow switching client when you hit a rate limit (e.g. using a different organization ID or using Azure). 

## Configuring OpenAI

To use the new OpenAI configuration, create `openai_configs.yaml` and set `export OPENAI_CLIENT_CONFIG_PATH=<path to openai_configs.yaml>`, by default the path is `<alpaca_eval_path>/client_configs/openai_configs.yaml`. The configuration should be a dictionary whose values are a list of configurations for OpenAI's client. We use a list to allow switching OpenAI client when you hit a rate limit (e.g. using a different organization ID or using Azure).

Here's the simplest configuration if you don't need to switch between OpenAI clients:

```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your organization ID>"
```


Here's if you want to switch between organization IDs when you hit a rate limit:

```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your 1st organization ID>"

    - api_key: "<your OpenAI API key here>"
      organization: "<your 2nd organization ID>"
```

Note that the order does NOT matter: we will select randomly the client. This allows running multiple jobs in parallel while decreasing the chance of using the same client.  

Sometimes you may need configurations that are model specific and use a different client class, for example when using Azure's client. In this case, you can do the following:


```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your 1st organization ID>"

    - api_key: "<your OpenAI API key here>"
      organization: "<your 2nd organization ID>"

gpt-4-1106-preview: # only when using `model_name: gpt-4-1106-preview` which is AlpacaEval2's model name
    - "default" # this will append all the `default` configs
    
    - client_class: "openai.AzureOpenAI" # doesn't use the `openai.OpenAI` client class
      # the following are passed to the `AzureOpenAI` client class
      azure_deployment: "gpt-4-1106" # name of the latest GPT-4 turbo on azure change as needed
      api_key: "<your Azure OpenAI API key here>"
      azure_endpoint: "<your Azure OpenAI API base here>"
      api_version: "2024-03-01-preview"
```


Here the configurations will be appended to `default` when using the model_name `gpt-4-1106-preview` in the `evaluators_configs` such as [here](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/configs.yaml#L6). When hitting a rate limit we will be then switching between two OpenAI clients and one Azure, each using the same underlying model. Note that when using Azure, some parameters might be slightly different and thus cause issues, as Azure typically lags a few months behind OpenAI's API. 

## Fully backward compatible

Prior to `alpaca_eval==0.3.7` the recommended way of setting the client was to use the environment variables `OPENAI_API_KEYS` / `OPENAI_ORGANIZATION_IDS`, which are lists of comma separated constants. **Using those will still work but will raise a warning**. Under the hood, if:
1. `openai_configs.yaml` does not exist, and
2. the environment variables are set

then the result will essentially correspond to the following config (where keys should be expanded):

```yaml
default:
- api_key: "<OPENAI_API_KEYS[0]>"
  organization: "<OPENAI_ORGANIZATION_IDS[0]>"

- api_key: "<OPENAI_API_KEYS[1]>"
  organization: "<OPENAI_ORGANIZATION_IDS[1]>"

...
```
