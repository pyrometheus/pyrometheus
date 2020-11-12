#! /usr/bin/env python3

from setuptools import setup, find_packages

ver_fname = "pyrometheus/version.py"

with open(ver_fname) as inf:
    version_file_contents = inf.read()

ver_dic = {}
exec(compile(version_file_contents, ver_fname, "exec"), ver_dic)

setup(name="pyrometheus",
      version=ver_dic["VERSION_TEXT"],
      description="Code generation for Cantera mechanisms",
      long_description=open("README.rst", "r").read(),
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Other Audience",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          "Topic :: Software Development :: Libraries",
          "Topic :: Utilities",
          ],

      python_requires="~=3.6",

      install_requires=[
          "cantera",
          "pymbolic",
          "mako",
          ],

      author="Andreas Kloeckner",
      url="https://github.com/inducer/pyrometheus",
      author_email="inform@tiker.net",
      license="MIT",
      packages=find_packages())
