#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()
with open("HISTORY.rst") as history_file:
    history = history_file.read()
requirements = [
    "numpy>=1.21.6",
    "scikit-learn>=1.0.2",
    "setuptools>=60.2.0"
]

extras = {
    "test": [
        "pandas>=1.3.0",
    ]
}


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
    description="Batch effects removal for microbiome data via conditional quantile regression",
    entry_points={
        "console_scripts": [
            "conqur=conqur.cli:main",
        ],
    },
    install_requires=requirements,
    extras_require=extras,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="conqur",
    name="conqur",
    packages=find_packages(include=["conqur", "conqur.*"]),
    url="https://github.com/Karkarmath/conqur",
    version="0.1.0",
    zip_safe=False,
)
