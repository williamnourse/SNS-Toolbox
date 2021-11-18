from setuptools import setup

setup(
    name='sns_toolbox',
    version='0.9',
    packages=['sns_toolbox'],
    url='https://github.com/wnourse05/SNS-Toolbox',
    license='Apache v2.0',
    author='William Nourse',
    author_email='nourse@case.edu',
    description='Tools for designing Synthetic Nervous Systems, and simulating them on various software/hardware backends',
    install_requires=['graphviz',
                      'torch',
                      'numpy>=1.17'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
