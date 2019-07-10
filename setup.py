#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:31:37 2019

@author: R. Wesley Henderson
"""

from setuptools import setup

setup(name='TIStan',
      version='0.1',
      description='Adaptively annealed thermodynamic integration with Stan',
      url='https://github.com/rwhender/ti-stan',
      author='Wesley Henderson',
      author_email='wesley.henderson11@gmail.com',
      license='LGPLv3',
      packages=['TIStan'],
      install_requires=['pystan', 'numpy', 'matplotlib', 'dill'],
      zip_safe=False)
