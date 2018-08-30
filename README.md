[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# MHCnptep
MHCNPTEP is a tool developed based on open source neural network models for peptide-MHC binding affinity prediction by [Hammerbacher lab](). It combines the predictions for naturally processed and T-cell reactive neoepitopes.
The predictors are implemented in Python using [keras](https://keras.io).

## Getting started

The package is installed on CTC as a Python module within conda tensorfow environment. To activate the environment:

```shell
module purge
module load miniconda3/4.3.3
source activate tensorflow
```

Then setup $LD_LIBRARY_PATH and run Python:

```shell
LD_LIBRARY_PATH=/home/zhaoweil/bin/lib64:/home/zhaoweil/bin/usr/lib64/ /home/zhaoweil/bin/lib64/ld-2.18.so `which python`
```

Now you should see the correct version of Python loaded:

```
Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55)
```

## Making predictions using pre-trained model in Python

```python
import tensorflow as tf
import mhcflurry
from mhcflurry import Class1BindingPredictor
from mhcflurry.class1_allele_specific.load import Class1AlleleSpecificPredictorLoader
modelsDir = "/work/genomics/tools/mhcnptep/mhcnptep/database/"

# Load NP prediction model
loader_np = Class1AlleleSpecificPredictorLoader(modelsDir+"NP-models/allele_specific_trained_models/")

# Load Tepi prediction model
loader_t = Class1AlleleSpecificPredictorLoader(modelsDir+"Tepi-modelsallele_specific_trained_models_tepi/")

# Predict a ninemer using loaded models
model = loader_np.from_allele_name("A0201")
prediction = model.predict([''])
```

The predictions returned by `predict` are (0,1) probability score of naturally presented or TCR reactive.

## Making predictions for a batch using command-line 

After ```source activate tensorflow```, instead of invoking Python environment, using a written pipeline script to call predictions

```shell
LD_LIBRARY_PATH=/home/zhaoweil/bin/lib64:/home/zhaoweil/bin/usr/lib64/ /home/zhaoweil/bin/lib64/ld-2.18.so `which python` -i predict.py
```

```
  Allele   Peptide  Prediction
0  A0201  SIINFEKL  10672.347656
```



## Training your own models

#### ADD Jupyter notebook here

See the [class1_allele_specific_models.ipynb](https://github.com/hammerlab/mhcflurry/blob/master/examples/class1_allele_specific_models.ipynb) notebook for an overview of the Python API, including predicting, fitting, and scoring models.

There is also a script called `mhcflurry-class1-allele-specific-cv-and-train` that will perform cross validation and model selection given a CSV file of training data. Try `mhcflurry-class1-allele-specific-cv-and-train --help` for details.

## Details on the downloaded class I allele-specific models

Besides the actual model weights, the data downloaded with `mhcflurry-downloads fetch` also includes a CSV file giving the hyperparameters used for each predictor. Another CSV gives the cross validation results used to select these hyperparameters.

To see the hyperparameters for the production models, run:

```
open "$(mhcflurry-downloads path models_class1_allele_specific_single)/production.csv"
```

To see the cross validation results:

```
open "$(mhcflurry-downloads path models_class1_allele_specific_single)/cv.csv"
```

## Environment variables

## Integration with Master neoantigen pipeline

## Versioning

## Author
Weilong Zhao, weilong.zhao@merck.com
