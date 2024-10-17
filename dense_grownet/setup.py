from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="dense_grownet",
    version="0.1.7",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scikit-learn",
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
