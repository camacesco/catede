# _catede_

The python software _catede_ is a collection of different estimation methods for the entropy and divergence in the context of categorical distribution with finite number of categories.
Provided a data sample in the shape of a list or other comprehensive supported formats, the software provides methods to estimate the Shannon entropy, the Simpson index, the Kullback-Leibler divergence and the Hellinger distance.
In particular it provides the code for the Dirichlet prior mixture estimators for the divergence, published in Camaglia et al. (2024), [Phys. Rev. E](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.024305).

![](docs/source/kullback-leibler-dirichlet-scheme.png)

## Installation

The package _catede_ can be installed from github by running from terminal the following command line :

 ```pip install git+https://github.com/statbiophys/catede.git```.

## References

Cite this as:

Francesco Camaglia, Ilya Nemenman, Thierry Mora, Aleksandra Walczak (2024). Bayesian estimation of the Kullback-Leibler divergence for categorical sytems using mixtures of Dirichlet priors. [Phys. Rev. E](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.024305)

## Tutorial

A brief tutorial can be found [here](tutorial.ipynb).

## Contact

Any issues or questions should be addressed to [us](mailto:francesco.camaglia@phys.ens.fr).

## License

Free use of catede is granted under the terms of the GNU General Public License version 3 (GPLv3).