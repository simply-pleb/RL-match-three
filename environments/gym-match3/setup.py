from setuptools import setup

setup(
    name='gym_match3',
    version='0.0.1',
    install_requires=[
        'gym==0.26.2',
        'numpy==1.26.4',
        'matplotlib==3.8.2',
    ],
    test_suite='tests'
)
