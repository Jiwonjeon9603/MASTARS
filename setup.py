from setuptools import find_packages, setup

setup(
    name="mastars",
    version="0.1.0",
    description="MASTARS: multi-agent trajectory augmentation with diffusion models",
    packages=find_packages(include=["mastars", "mastars.*"]),
    python_requires=">=3.8",
)
