from setuptools import setup

setup(
    name='plutokore',
    packages=['plutokore'],
    version='0.1',
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
