# Getting Started with Marin

This tutorial will guide you through the basic setup and usage of Marin.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marin-community/marin.git
   cd marin
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -e .
   ```


## Next Steps

- If you're on GPU, see [local-gpu.md](local-gpu.md) for a GPU-specific walkthrough for getting started.

- Read about our [language modeling efforts](..//lm/overview.md)
- Train a [tiny language model](..//how-to-guides/train-an-lm.md) using Marin.
- Read about Marin's key concepts and principles in [Concepts](../explanation/concepts.md)
- Learn about the [Executor framework](../explanation/executor.md): how to manage Python libraries, run big parallel jobs using Ray, how versioning works, etc.
- Read about [Experiments](../explanation/experiments.md): how we use the executor framework to run machine learning experiments.
