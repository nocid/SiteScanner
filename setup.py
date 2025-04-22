from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()
   
    required = [line for line in required if line and not line.startswith('#')]

setup(
    name='sitescanner',
    version='0.1.0', 
    author='Romain Pastre, Daniel Guinon Fort, Leyao Jin',
    author_email='romain.pastre01@estudiant.upf.edu, daniel.guinon01@estudiant.upf.edu, leyao.jin01@estudiant.upf.edu',
    description='Predict protein binding sites from PDB structures.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nocid/SiteScanner', 
    packages=find_packages(),
    install_requires=required,
    # Include package data (like weights)
    package_data={
        'sitescanner': ['weights/*.pth'],
    },
    include_package_data=True,
    
    entry_points={
        'console_scripts': [
            'sitescanner=sitescanner.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8', 
)