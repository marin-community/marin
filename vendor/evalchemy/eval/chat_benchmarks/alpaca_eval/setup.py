import setuptools

setuptools.setup(
    name="alpaca_eval",
    version="0.6.5",
    description="AlpacaEval prompt processing and grading for Evalchemy",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=[
        "datasets",
        "numpy",
        "pandas",
        "pyyaml",
        "scipy",
        "huggingface_hub",
        "patsy",
        "scikit-learn",
    ],
    python_requires=">=3.12",
    include_package_data=True,
)
