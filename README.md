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
* [`synthetic_bjrnn.ipynb`](./synthetic.ipynb) (**Note:** this notebook should be executed with requirements as per [`requirements_bjrnn.txt`](./requirements_bjrnn.txt).)
* [`medical.ipynb`](./medical.ipynb)

You can download the publicly available data for this work [here](https://drive.google.com/drive/folders/1fD66DKTMjQNxLrfVZo803ScXawkyth7P?usp=sharing). As the MIMIC-III dataset [requires PhysioNet credentialing](https://mimic.mit.edu/docs/gettingstarted/) to access, you must become a credentialed user on PhysioNet before accessing the data. To get access to the dataset as used in this work, please contact the authors and provide proof of your PhysioNet credentialing.



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
