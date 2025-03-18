from setuptools import setup, find_packages

setup(
    name="hdc_rna",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "torch>=1.9.0",
        "tqdm>=4.50.0",
    ],
    author="HDC-RNA Team",
    author_email="example@example.com",
    description="Hyperdimensional Computing for RNA 3D Structure Prediction",
    keywords="RNA, bioinformatics, hyperdimensional computing, machine learning",
    url="https://github.com/example/hdc_rna",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
) 