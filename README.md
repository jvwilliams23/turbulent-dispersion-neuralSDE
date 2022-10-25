# Turbulent dispersion neuralSDE

Train a neural network to learn drift and diffusion components of stochastic differential equations (SDEs), using OpenFOAM data. We use this for modelling turbulent dispersion.

## Installation
### Installation - OpenFOAM
A make file is provided in openfoam/AllMake. This compiles the modified "lagrangian" libraries (namely, "intermediate" and "turbulence" libraries).


### Installation - Python
To install the required packages for the python script, use:
```bash
conda create --name sdeenv tensorflow=2.4.1 Keras=2.4.3 numpy=1.20 scipy=1.6.0 setuptools=51.0 joblib=1.0.1 python=3.8 
conda activate sdeenv
pip install hjson
```
Also, the python script assumes that the dataset from [our Kaggle repository](https://www.kaggle.com/datasets/jvwilliams23/filtered-direct-numerical-simulation-dataset) has been downloaded to "dataset-filteredDNS/" (feel free to modify the path).

## BibTeX Citation

If you use our model in a scientific publication, we would appreciate using the following citation for our preprint (paper citation will be added later), and dataset:

```
@article{williams2022neuralSDEarxiv,
  title = {Neural stochastic differential equations for particle dispersion in large-eddy simulations of homogeneous isotropic turbulence},
  author = {Williams, Josh and Wolfram, Uwe and Ozel, Ali},
  doi = {10.48550/ARXIV.2208.08156},
  url = {https://arxiv.org/abs/2208.08156},
  keywords = {Fluid Dynamics (physics.flu-dyn), FOS: Physical sciences, FOS: Physical sciences},
  publisher = {arXiv},
  journal = {arXiv preprint arXiv:2208.08156},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}

@misc{williams2022filteredDNSkaggle,
        title={Filtered direct numerical simulation dataset},
        url={https://www.kaggle.com/dsv/3998403},
        DOI={10.34740/KAGGLE/DSV/3998403},
        publisher={Kaggle},
        author={Josh Williams and Uwe Wolfram and Ali Ozel},
        year={2022}
}
```


