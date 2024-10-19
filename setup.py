from setuptools import setup, find_packages

setup(
    name='NumericalCodes',
    version='1.0.0',
    license='MIT',
    description='Numerical codes for simple CFD',
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    packages=find_packages(),
)
