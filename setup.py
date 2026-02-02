from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="heart-disease-prediction-api",
    version="1.0.0",
    description="API for predicting heart disease risk based on patient data",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
    include_package_data=True,
)