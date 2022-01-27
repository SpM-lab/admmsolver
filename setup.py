# Copyright (C) 2021-2022 H. Shinaoka and others
# SPDX-License-Identifier: MIT
import io, os.path, re
from setuptools import setup, find_packages

def readfile(*parts):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()


def extract_version(*parts):
    """Extract value of __version__ variable by parsing python script"""
    initfile = readfile(*parts)
    version_re = re.compile(r"(?m)^__version__\s*=\s*['\"]([^'\"]*)['\"]")
    match = version_re.search(initfile)
    assert match is not None
    return match.group(1)


VERSION = extract_version('src', 'admmsolver', '__init__.py')
REPO_URL = "https://github.com/SpM-lab/admmsolver"
LONG_DESCRIPTION = readfile('README.md')

setup(
    name='admmsolver',
    version=VERSION,

    description=
        'Fast and general ADMM solver',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=' '.join([
        'ADMM'
        ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        ],

    url=REPO_URL,
    author="Hiroshi Shinaoka",
    author_email='h.shinaoka@gmail.com',

    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={
        'test': ['pytest', 'mypy'],
        'dev': ['pytest', 'mypy']
        },

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={
        "admmsolver": ["py.typed"],
    },
    )
