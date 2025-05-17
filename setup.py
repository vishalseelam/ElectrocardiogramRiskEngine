from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="ecg-risk-engine",
    version="1.0.0",
    author="Vishal Seelam",
    author_email="vishal.seelam@tcu.edu",
    description="A machine learning-based ECG analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishalseelam/ElectrocardiogramRiskEngine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ecg-server=main:main",
        ],
    },
) 