# Conformal Time-Series Forecasting

Codebase for "Conformal Time-Series Forecasting", submitted for review at NeurIPS
2021 (Paper6977), to be available online upon acceptance.

This codebase builds on the codebase for
"Frequentist Uncertainty in Recurrent Neural Networks
via Blockwise Influence Functions" (ICML 2020), available at
https://github.com/ahmedmalaa/rnn-blockwise-jackknife
under the BSD 3-clause license. 

---

## Directory contents

This repository contains the Appendix to the paper at
`NeurIPS2021_conformal_rnn_Appendix.pdf`.

While we do not upload raw or processed data for the real-world medical datasets,
(though the processing procedures are available at `utils/data_processing_*.py`),
we save the results of the models, available at the `saved_results` directory. 

The examples of how the models were trained and results computed for both synthetic
and medical datasets are available at `synthetic.ipynb` and `medical.ipynb`
respectively.

The model implementations are available at the `models` directory. 
The main contribution of this
paper is the CoRNN (here CPRNN) model, available at `models/cprnn.py`.
