"""Install script for MiniTorch."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minitorch",
    version="0.1.0",
    author="York",
    description="A minimal deep learning framework for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/york810327-rgb/minitorch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
    },
)