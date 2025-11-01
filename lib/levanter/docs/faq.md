# Frequently Asked Questions

## Project

### Why is it called Levanter?

Levanter is a wind that blows from the east in the Mediterranean. Stanford CRFM's first training project was
called [Mistral](https://github.com/stanford-crfm/mistral), which is another Mediterranean wind. (That Mistral
has no relation to the now more famous [Mistral AI](https://www.mistral.ai/). They took our name!)


## Installation Issues

### CUDA: `XLA requires ptxas version 11.8 or higher`

`jaxlib.xla_extension.XlaRuntimeError: INTERNAL: XLA requires ptxas version 11.8 or higher`

This error occurs when your local CUDA installation is too old. When you follow the
[GPU installation instructions](Getting-Started-GPU.md), you install a version of CUDA in a pip environment.
If you have another version of CUDA installed on your machine, it may be interfering with the pip environment.
The usual solution for this is to either upgrade your local CUDA installation or hide it from your PATH. Usually this works:

```bash
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')
```

You should add that to your `.bashrc` or `.zshrc` or whatever shell you use and restart your shell.


## Nuisance Warnings

### Transformers: `None of PyTorch, TensorFlow >= 2.0, or Flax have been found.`

`None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.`

Hugging Face's Transformers library is a dependency of Levanter, but we only use it for the tokenizers and a few utilities.
If you don't plan on using the Hugging Face models, you can safely ignore this warning. If the warning bothers you,
you can set the environment variable `export TRANSFORMERS_VERBOSITY=error` to silence it.

## Cloud Issues

### Permission 'storage.objects.get' denied on resource

```
gcsfs.retry.HttpError: Anonymous caller does not have storage.objects.get access to the Google Cloud Storage object.
Permission 'storage.objects.get' denied on resource (or it may not exist)., 401
```

If you're using Google Cloud Storage on a non-TPU machine, you might get errors like this. The solution is to log
into your Google Cloud account on the machine:

```bash
gcloud auth login
gcloud auth application-default login
```

## Ray Issues

### RuntimeError: Failed to start ray head with exit code 256

Probably ray is still running and Levanter didn't clean up the ray cluster (or another user is using the same port).
If the former, you can kill the ray cluster with `ray stop`. If the latter, there's not much you can do about it.
[Ray doesn't work super well when multiple users are running Ray on the same machine.](https://github.com/ray-project/ray/issues/20634)
Try docker?

Another reason could be the ports are not open in your VM. If using GCP, check the firewall settings of your VPC and expose port `61964` (used by ray).
