from setuptools import setup, find_packages


def readme_file_contents():
    with open('README.rst') as f:
        data = f.read()
    return data

setup(
    name='sns_toolbox',
    version='1.2.0',
    packages=find_packages(),
    url='https://github.com/wnourse05/SNS-Toolbox',
    license='Apache v2.0',
    author='William Nourse',
    author_email='nourse@case.edu',
    description='Tools for designing Synthetic Nervous Systems, and simulating them on various software/hardware backends',
    long_description='Documentation is available on ReadTheDocs  at https://sns-toolbox.readthedocs.io/en/latest/index.html',
    setup_requires=['wheel'],
    python_requires='>=3.5',
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
