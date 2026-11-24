from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neurospike",
    version="1.0.0",
    author="Nikshep & NeuroSpike Contributors",
    description="Neuromorphic Event-Driven Intelligence with Spiking Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
    ],
)
