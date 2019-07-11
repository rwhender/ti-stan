#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:31:37 2019

@author: R. Wesley Henderson
"""

from setuptools import setup, find_packages


def readme():
    with open('README.md', "r") as f:
        return f.read()


setup(name='TIStan',
      version='0.1.3',
      description='Adaptively annealed thermodynamic integration with Stan',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='https://github.com/rwhender/ti-stan',
      author='Wesley Henderson',
      author_email='wesley.henderson11@gmail.com',
      license='LGPL-3.0-or-later',
      packages=find_packages(),
      package_data={'TIStan.tests': ['*.dill', '*.stan']},
      install_requires=['pystan', 'numpy', 'matplotlib', 'dill'])
