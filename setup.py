from setuptools import setup, find_packages

# Dependencias opcionales para desarrollo y documentación
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.800",
        "pre-commit>=2.10.0",
    ],
    "docs": [
        "mkdocs-material>=9.0.0",
        "mkdocstrings[python]>=0.20.0",
        "pymdown-extensions>=9.0",
    ],
    "optuna": [
        "optuna>=3.0.0",
        "optuna-dashboard>=0.7.0",
    ]
}

# Combinar todas las dependencias para instalación completa
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="ibioml",
    version="0.1.5",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0"
    ],
    extras_require=extras_require,
    author="Juan Ignacio Ponce",
    author_email="jiponce@ibioba-mpsp-conicet.gov.ar",
    description="ML toolkit for neuro decoding experiments at IBIoBA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/IBioML",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    keywords="neuroscience machine-learning decoding brain-computer-interface",
)
