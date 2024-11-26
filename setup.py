from setuptools import setup, find_packages

setup(
    name="demand_forecasting",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'pyspark',
        'tensorflow',
        'flask',
        'scikit-learn'
    ]
) 