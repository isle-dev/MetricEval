import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '0.01'
DESCRIPTION = 'a python package for evaluating evaluation metrics '
LONG_DESCRIPTION = 'metric-eval contains a set of statistical tools to evalaute an evaluation metric in term of reliability and validity.'

# Setting up
setup(
    name="metric-eval",
    version=VERSION,
    author='Ziang Xiao, Susu Zhang', 
    author_email='ziang.xiao@jhu.edu, szhan105@illinois.edu',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    keywords=['python','metrics','evaluation','measurement','natural language processing','natural language generation'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)