# Turbulent dispersion neuralSDE

Train a neural network to learn drift and diffusion components of stochastic differential equations (SDEs), using OpenFOAM data. We use this for modelling turbulent dispersion.

### Installation
To install the required packages, use:
```bash
conda create --name sdeenv tensorflow=2.4.1 Keras=2.4.3 numpy=1.20 scipy=1.6.0 setuptools=51.0 joblib=1.0.1 python=3.8 
```

**Full code for training and testing model will be made available after peer-review.**

## BibTeX Citation

If you use our model in a scientific publication, we would appreciate using the following citation for our dataset (paper citation will be added later):

```
@misc{filteredDNSWilliams2022,
        title={Filtered direct numerical simulation dataset},
        url={https://www.kaggle.com/dsv/3998403},
        DOI={10.34740/KAGGLE/DSV/3998403},
        publisher={Kaggle},
        author={Josh Williams and Uwe Wolfram and Ali Ozel},
        year={2022}
}
```


