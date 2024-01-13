#python setup.py sdist bdist_wheel
import os
from setuptools import setup, find_packages


setup(
    name="bikesharing",
    version="0.1",
    description="AIMLOPS Bike Sharing Project",
    long_description="AIMLOPS Module 3 Mini Project 2",
    install_requires=[
        'numpy',
        'pandas',
        'pydantic',
        'scikit-learn',
        'strictyaml',
        'ruamel.yaml',
        'joblib',
        'pytest'
    ],
    include = ["config/*.yaml", "bikeshare_model/VERSION"]
    
)
