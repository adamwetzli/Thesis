from setuptools import setup, find_packages

setup(
    name="masters_thesis",           # Package name (what you'd pip install)
    version="0.1",               # Version number
    packages=find_packages(where="src"),  # Finds all packages in src/
    package_dir={"": "src"},     # Maps package root to src/ folder
)