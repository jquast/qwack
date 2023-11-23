#!/usr/bin/env python
"""Distutils setup script."""
import os
import setuptools

HERE = os.path.dirname(__file__)

setuptools.setup(
    name='qwack',
    version='0.0.3',
    install_requires=['pyyaml', 'blessed'],
    long_description=open(os.path.join(HERE, 'README.rst')).read(),
    description='a rogue-like game of mysterious origins!',
    author='Jeff Quast',
    author_email='contact@jeffquast.com',
    license='MIT',
    packages=['qwack'],
    # just add the tilesets and world.yaml so far ..
    package_data={"dat": ["*.zip", "*.yaml"]},
    url='https://github.com/jquast/qwack',
    include_package_data=True,
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'qwack = qwack:main.main',
        ] }
)
