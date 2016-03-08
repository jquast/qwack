#!/usr/bin/env python
"""Distutils setup script."""
import os
import setuptools

HERE = os.path.dirname(__file__)

_install_requires = ['pyyaml', 'blessed']
try:
    import enum
except ImportError:
    _install_requires.append('enum34')

setuptools.setup(
    name='qwack',
    version='0.0.2',
    install_requires=_install_requires,
    long_description=open(os.path.join(HERE, 'README.rst')).read(),
    description='a quickly written hack (1985) variant for Python.',
    author='Jeff Quast',
    author_email='contact@jeffquast.com',
    license='MIT',
    packages=['qwack',],
    url='https://github.com/jquast/qwack',
    # TODO: ensure dat/world.yaml is distributed as resource
    include_package_data=True,
    zip_safe=True,
)
