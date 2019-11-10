# -*- coding: utf8 -*-

import os
from setuptools import setup, find_packages

# Meta information
dirname = os.path.dirname(__file__)

setup(
    name='gammanet',
    version='0.1',
    author='Lane',
    author_email='thomas.joseph.lane@gmail.com',
    url='https://github.com/tjlane/gammanet',

    # Packages and depencies
    packages=['gammanet'],
    package_dir={'gammanet': 'gammanet'},
    install_requires=[
        'numpy',
        'torch'
    ],

    # Data files
    #package_data={
    #    'gammanet': [
    #        'test/data/*.*',
    #    ]
    #},

    # Scripts
    #entry_points={
    #    'console_scripts': [
    #        'python-boilerplate = python_boilerplate.__main__:main'],
    #},

    # Other configurations
    zip_safe=False,
    platforms='any',
)

