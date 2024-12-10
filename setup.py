#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    setup.py
#
#    AFToolKit setup.
#


# Standard library
from setuptools import setup, find_packages

# Setup options
name = "AFToolKit"

author = "Shashkova Tatiana, Sindeeva Maria, Ivanisenko Nikita, Telepov Alexander"

version = "1.0.0"

description = \
    "Python library for routine protein engineering tasks using AlphaFold2."

package_dir = {"AFToolKit": "AFToolKit"}

packages = find_packages(exclude=['tests'])

package_data = \
    {"AFToolKit": ["processing/*",
                   "processing/openfold/*",
                   "models/*"
                   ]}

entry_points = \
    {"console_scripts": \
         ["run_protein_task=AFToolKit.processing.run_protein_task:main",
          "run_protein_complex_task=AFToolKit.processing.run_protein_complex_task:main"],
     }

install_requires = ["biopython",
                    "numpy",
                    "pandas",
                    "Biopandas"]

# Run the setup
setup(name=name,
      # url=url,
      author=author,
      version=version,
      description=description,
      include_package_data=True,
      package_data=package_data,
      package_dir=package_dir,
      packages=packages,
      entry_points=entry_points,
      install_requires=install_requires)
