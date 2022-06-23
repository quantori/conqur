#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()
with open("HISTORY.rst") as history_file:
    history = history_file.read()
requirements = [
    "pip==19.2.3",
    "bump2version==0.5.11",
    "wheel==0.33.6",
    "watchdog==0.9.0",
    "flake8==3.7.8",
    "tox==3.14.0",
    "numpy",
    "pandas",
    "scikit-learn",
    "setuptools",
]

test_requirements = []

setup(
    author="Aleksandr Grefenstein",
    author_email="alexander.grefenshtejn@quantori.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Batch effects removal for microbiome data via conditional quantile",
    entry_points={
        "console_scripts": [
            "conqur=conqur.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="conqur",
    name="conqur",
    packages=find_packages(include=["conqur", "conqur.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Karkarmath/conqur",
    version="0.1.0",
    zip_safe=False,
)
