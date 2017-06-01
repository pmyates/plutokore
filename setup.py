from setuptools import setup, find_packages

setup(
    name='plutokore',
    packages=find_packages(), 
    version='0.10',
    description='Python tool for analysing PLUTO simulation data',
    author='Patrick Yates',
    author_email='patrick.yates@utas.edu.au',
    url='https://github.com/pmyates/plutokore',
    keywords=['pluto', 'astrophsyics'],
    license='GPL-3.0',
    install_requires=['numpy', 'matplotlib', 'tabulate', 'astropy', 'h5py', 'pyyaml', 'scipy', 'contextlib2', 'future'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-datafiles'],
)
