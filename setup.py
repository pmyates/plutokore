from setuptools import setup, find_packages

setup(
    name='plutokore',
    packages=find_packages(), 
    version='0.2',
    description='Python tool for analysing PLUTO simulation data',
    author='Patrick Yates',
    author_email='patrick.yates@utas.edu.au',
    url='https://github.com/opcon/plutokore',
    keywords=['pluto', 'astrophsyics'],
    license='MIT',
    requires=['numpy', 'matplotlib', 'tabulate'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
    ]
)
