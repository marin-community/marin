import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "alpaca_eval", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find `__version__`.")

PACKAGES_DEV = [
    "pre-commit>=3.2.0",
    "black>=23.1.0",
    "isort",
    "pytest",
    "pytest-mock",
    "pytest-skip-slow",
    "pytest-env",
    "python-dotenv",
]
PACKAGES_ANALYSIS = ["seaborn", "matplotlib", "jupyterlab"]
PACKAGES_LOCAL = [
    "accelerate",
    "transformers",
    "bitsandbytes",
    "torch",
    "xformers",
    "peft",
    "optimum",
    "einops",
    "vllm",
]
PACKAGES_ALL_API = [
    "anthropic>=0.18",
    "cohere<5.0.0a0",
    "replicate",
    "boto3>=1.28.58",
    "google-generativeai",
]
PACKAGES_ALL = PACKAGES_LOCAL + PACKAGES_ALL_API + PACKAGES_ANALYSIS + PACKAGES_DEV

setuptools.setup(
    name="alpaca_eval",
    version=version,
    description="AlpacaEval : An Automatic Evaluator of Instruction-following Models",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    author="The Alpaca Team",
    install_requires=[
        "python-dotenv",
        "datasets",
        "openai>=1.5.0",
        "pandas",
        "tiktoken>=0.3.2",
        "fire",
        "scipy",
        "huggingface_hub",
        "patsy",
        "scikit-learn",
    ],
    extras_require={
        "analysis": PACKAGES_ANALYSIS,
        "dev": PACKAGES_DEV,
        "local": PACKAGES_LOCAL,
        "api": PACKAGES_ALL_API,
        "all": PACKAGES_ALL,
    },
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "alpaca_eval=alpaca_eval.main:main",
        ],
    },
    include_package_data=True,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
