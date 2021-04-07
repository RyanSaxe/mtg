import re

from pkg_resources import parse_requirements
import pathlib
from setuptools import find_packages, setup

README_FILE = 'README.md'
REQUIREMENTS_FILE = 'requirements.txt'
VERSION_FILE = 'mtg/_version.py'
VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')

version = r.group(1)
long_description = open(README_FILE, encoding='utf-8').read()
install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]

setup(
    name='mtg',
    version=version,
    description='mtg is a collection of data science and ml projects for Magic:the Gathering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ryan Saxe',
    author_email='ryancsaxe@gmail.com',
    url='https://github.com/RyanSaxe/mtg',
    packages=find_packages(),
    install_requires=install_requires,
)

import os
import pathlib
pathlib.Path('data/').mkdir(parents=True, exist_ok=True)
