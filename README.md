# Conformal time-series forecasting

Implementation for [Stankevičiūtė et al. 
"Conformal time-series forecasting", NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html).

This codebase builds on the implementation for
"Frequentist Uncertainty in Recurrent Neural Networks
via Blockwise Influence Functions" (ICML 2020), available at
https://github.com/ahmedmalaa/rnn-blockwise-jackknife
under the BSD 3-clause license. 



## Installation
Python 3.6+ is recommended. Install the dependencies from [`requirements.txt`](./requirements.txt).



## Replicating Results
To replicate experiment results, run the notebooks:
* [`synthetic.ipynb`](./synthetic.ipynb)
* [`medical.ipynb`](./medical.ipynb)



## Citing

If you use our code in your research, please cite:

```
@inproceedings{stankeviciute2021conformal,
  author = {Stankevičiūtė, Kamilė and Alaa, Ahmed M. and {van der Schaar}, Mihaela},
  title = {Conformal time-series forecasting},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021}
}
```
