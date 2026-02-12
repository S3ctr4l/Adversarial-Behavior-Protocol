from setuptools import setup, find_packages

setup(
    name="adversarial-benevolence",
    version="0.1.0",
    description="Structural AI safety through entropy dependency and hierarchical verification",
    author="Joshua Roger Joseph Just",
    author_email="mytab5141@protonmail.com",
    url="https://github.com/S3ctr4l/Adversarial-Benevolence-Protocol",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.12.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",  # CC BY 4.0 not in PyPI list
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)