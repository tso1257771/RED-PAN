#!/usr/bin/env python3
"""
Setup script for REDPAN
========================

High-Performance SeisBench-Inspired RED-PAN Implementation
for continuous seismic phase picking with 10-50x performance improvements.
"""

from setuptools import setup, find_packages
import os
import re

# Get version from __init__.py
def get_version():
    with open("redpan/__init__.py", "r", encoding="utf-8") as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.0"

# Read the README file
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "REDPAN: High-Performance SeisBench-Inspired RED-PAN Implementation"

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#") and not line.startswith("-")
            ]
    return requirements

setup(
    name="redpan",
    version=get_version(),
    author="Wu-Yu Tso",
    author_email="tso1257771@gmail.com",
    maintainer="Rick and Contributors",
    description="High-Performance Deep Learning Seismic Phase Picker with SeisBench-Style Optimization",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/tso1257771/RED-PAN",
    project_urls={
        "Bug Tracker": "https://github.com/tso1257771/RED-PAN/issues",
        "Documentation": "https://github.com/tso1257771/RED-PAN/blob/main/docs/",
        "Source Code": "https://github.com/tso1257771/RED-PAN",
        "Changelog": "https://github.com/tso1257771/RED-PAN/blob/main/CHANGELOG.md",
        "Docker": "https://hub.docker.com/r/redpan/redpan",
    },
    packages=find_packages(exclude=["tests*", "scripts*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0,<2.0.0",
        "tensorflow>=2.4.0,<3.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "seismic": [
            "obspy>=1.2.0",
            "h5py>=3.0.0",
        ],
        "visualization": [
            "matplotlib>=3.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "performance": [
            "psutil>=5.0.0",
            "multiprocess>=0.70.0",
            "numba>=0.50.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "ipywidgets>=7.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "pytest-xdist>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "docker": [
            "docker>=5.0.0",
        ],
        "all": [
            "obspy>=1.2.0",
            "h5py>=3.0.0",
            "matplotlib>=3.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "psutil>=5.0.0",
            "multiprocess>=0.70.0",
            "numba>=0.50.0",
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    keywords=[
        "seismology", 
        "earthquake", 
        "phase-picking", 
        "machine-learning", 
        "tensorflow",
        "red-pan",
        "seisbench",
        "continuous-processing",
        "real-time"
    ],
    entry_points={
        "console_scripts": [
            "redpan-benchmark=redpan.cli:benchmark_command",
            "redpan-test=redpan.cli:test_command",
            "redpan-demo=redpan.cli:demo_command",
            "redpan-parallel=redpan.cli:parallel_command",
        ],
    },
    include_package_data=True,
    package_data={
        "redpan": ["*.md", "docs/*.md", "examples/*.py", "tests/*.py"],
    },
    zip_safe=False,
)
