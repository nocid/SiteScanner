from setuptools import setup, find_packages

# It's good practice to read dependencies from requirements.txt
# Or list them directly here.
# Note: Handling the complex PyG install might require extra steps or documentation.
INSTALL_REQUIRES = [
    'torch',
    'torch_geometric',
    'biopython',  # Covers Bio.PDB etc.
    'e3nn>=0.5.0', # Specify versions if needed
    'numpy',
    'scipy',
    'scikit-learn', # For sklearn metrics/splits
    'joblib', # For parallel processing if used outside training script
    'transformers', # For ESM
    'fair-esm', # ESM dependency
    # PyTorch Scatter/Sparse/Cluster/SplineConv are often dependencies of PyG
    # Pip might handle them via PyG's dependencies, but check PyG install instructions
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    # Add other direct dependencies if any
]

setup(
    name='sitescanner',
    version='1.0.0', # Start with an initial version
    packages=find_packages(), # Automatically find your 'sitescanner' package
    install_requires=INSTALL_REQUIRES,
    entry_points={
        'console_scripts': [
            'sitescanner = sitescanner.cli:main', # This creates the command-line tool
        ],
    },
    # Add other metadata like author, description, url, etc.
    author='Romain Pastre, Leyao Jin, Daniel Guiñon Fort',
    author_email='romain.pastre01@estudiant.upf.edu',
    description='Predict protein binding sites using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nocid/SiteScanner', # Link to your repo
    classifiers=[ # PyPI classifiers
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # Specify compatible Python versions
)