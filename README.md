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
LD_LIBRARY_PATH=/work/genomics/tools/tensorflowLib/lib64:/work/genomics/tools/tensorflowLib/usr/lib64/ /work/genomics/tools/tensorflowLib/lib64/ld-2.18.so `which python`
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
loader_np = Class1AlleleSpecificPredictorLoader(modelsDir+"NP-models/")

# Load Tepi prediction model
loader_t = Class1AlleleSpecificPredictorLoader(modelsDir+"Tepi-models/")

# Predict a ninemer using loaded models
model = loader_np.from_allele_name("$YOUR_ALLELE")
prediction = model.predict(["$YOUR_PEPTIDE"])
prediction
array([$PREDICTED_VALUE], dtype=float32)
```

The predictions returned by `predict` are (0,1) probability score of naturally presented or TCR reactive.

Exit tensorflow/mhcflurry conda environment after job done:

```shell
source deactivate
```

##### Alternatively, run with a command-line script for single peptide prediction:

```shell
LD_LIBRARY_PATH=/work/genomics/tools/tensorflowLib/lib64:/work/genomics/tools/tensorflowLib/usr/lib64/ /work/genomics/tools/tensorflowLib/lib64/ld-2.18.so `which python` /work/genomics/tools/mhcnptep/run-scripts/predict_singlePep.py \

-a "A0201" -p "FLGGTTVCL"

source deactivate
```

## Making predictions for a batch using command-line script

After ```source activate tensorflow```, instead of invoking Python environment, using a written pipeline script to call predictions

```shell
LD_LIBRARY_PATH=/work/genomics/tools/tensorflowLib/lib64:/work/genomics/tools/tensorflowLib/usr/lib64/ /work/genomics/tools/tensorflowLib/lib64/ld-2.18.so `which python` /work/genomics/tools/mhcnptep/run-scripts/predict_batch.py \

-s /work/clinical/MK1308.PN001.PID21597.AID1303/analysis/batch.6.20180726/match0726 \

-op /work/clinical/MK1308.PN001.PID21597.AID1303/analysis/batch.6.20180726/out0726/optitype/ \

-n /work/clinical/MK1308.PN001.PID21597.AID1303/analysis/batch.6.20180726/out0726/neoI_II/ \

-o /work/clinical/MK1308.PN001.PID21597.AID1303/analysis/batch.6.20180726/out0726/neoIGS/
```

The ```predict_batch.py``` script takes input of annotated 9-mer neopeptide file `/$BATCH/$OUT/neoI_II/$SAMPLE.9.pep`:

```shell
$ python predict_batch.py
usage: predict_batch.py [-h] --sample_file SAMPLE_FILE --optitype OPTITYPE
                        --neo NEO --output OUTPUT [--logs LOGS]
```

## Training your own models (TODO)

#### ADD Jupyter notebook here

/*See the [class1_allele_specific_models.ipynb](https://github.com/hammerlab/mhcflurry/blob/master/examples/class1_allele_specific_models.ipynb) notebook for an overview of the Python API, including predicting, fitting, and scoring models.

/*There is also a script called `mhcflurry-class1-allele-specific-cv-and-train` that will perform cross validation and model selection given a CSV file of training data. Try `mhcflurry-class1-allele-specific-cv-and-train --help` for details.

## Details on the downloaded class I allele-specific models (TODO)

/*Besides the actual model weights, the data downloaded with `mhcflurry-downloads fetch` also includes a CSV file giving the hyperparameters used for each predictor. Another CSV gives the cross validation results used to select these hyperparameters.

/*To see the hyperparameters for the production models, run:

/*```
/*open "$(mhcflurry-downloads path models_class1_allele_specific_single)/production.csv"
/*```

To see the cross validation results:

/*```
/*open "$(mhcflurry-downloads path models_class1_allele_specific_single)/cv.csv"
/*```


## Integration with Master neoantigen pipeline (TODO)

## Versioning

## Author
Weilong Zhao, weilong.zhao@merck.com
