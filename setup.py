import setuptools
from setuptools import find_namespace_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name = 'specfem',
    version = '0.1.0',
    description = 'Code used for the MSc thesis',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Pedro Martínez',
    author_email = 'martinezsaiz.pedro@gmail.com',
    url = 'https://github.com/pedro-mrtnz/msc_thesis.git',
    packages = find_namespace_packages(),
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires = '>=3.6.0',
    install_requires = [
        'numpy', 'scipy', 'pandas', 'obspy', 'gmsh', 'pygmsh', 'meshio'
    ]
)