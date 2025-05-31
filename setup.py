#!/usr/bin/env python3
"""
Setup script for torch-circuit package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="torch-circuit",
    version="0.9.1",
    author="ntippens",
    author_email="ndtippens@gmail.com",
    description="A PyTorch extension for building neural networks with skip connections and repeatable blocks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/torch-circuit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
    },
    keywords="pytorch neural-networks deep-learning skip-connections resnet",
    project_urls={
        "Bug Reports": "https://github.com/ndtippens/torch-circuit/issues",
        "Source": "https://github.com/ndtippens/torch-circuit",
        "Documentation": "https://github.com/ndtippens/torch-circuit/",
    },
    include_package_data=True,
    zip_safe=False,
)
