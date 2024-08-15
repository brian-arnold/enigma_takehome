# Enigma takehome exercise

This code contains functions to perform a single Factor Analysis as specified in Ghahramani & Hinton (1996).

The goal of factor analysis is to find values for $\Lambda$ and $\Psi$ that best model the covariance structure of the data. Because these parameters depend on unobserved latent factors (and vice versa), we estimate them via EM. 

## Installation

To run this code, I first create a conda environment using the following code:

```
mamba create --name enigma \
anaconda::ipykernel \
anaconda::numpy \
conda-forge::matplotlib \
anaconda::seaborn \
anaconda::pandas \
anaconda::scipy \
anaconda::pytest \
-y
```

I then activate this environment:

```
mamba activate enigma
```

Then move into the base directory of `my_package` and pip install the package in development mode:

```
cd my_package
pip install -e .
```

## Usage

Once this environment is created, I can specify it as the kernel in VSCode. Then, as the top of `notebooks/factor_analysis.ipynb` that implements the functions in `my_package`, I use:

```
from my_package import functions
```

To run tests, activate the conda environment `enigma`, navigate to the base directory of `my_package`, and type `pytest` on the terminal.

Please see `notebooks/factor_analysis.ipynb` to see how to use the functions in `my_package` and confirm that they work as expected.

## Improvement
- Currently, convergence is monitored using changes in $\Lambda$ and $\Psi$ matrices, but computing the likelihood instead may be simpler and more interpretable. But, the likelihood equation was long, and done > perfect.
- The algorithm could be simplified, shortened, and enhanced for speed. This would invovle refactoring the code and running tests to ensure the behavior of the code hasn't changed.
- The current functions in `functions.py` are a little long. Splitting them up could help with developing tests for individual components.