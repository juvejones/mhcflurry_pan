import mhcflurry
import pandas as pd
import numpy as np
import sklearn
import timeit
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(0)

from mhcflurry.dataset import Dataset
from mhcflurry import Class1BindingPredictor

from nose.tools import eq_
from numpy import testing

from mhcflurry.downloads import get_path


dataset = Dataset.from_csv("HLA-A0201")
dataset_9mer = Dataset(dataset._df.ix[dataset._df.peptide.str.len() == 9])

folds = mhcflurry.class1_allele_specific.cross_validation.cross_validation_folds(dataset, n_folds = 5)
print folds[0].train.affinities
print folds[0].test.affinities

predictor = Class1BindingPredictor(
    name="A0201",
    embedding_output_dim=32,
    activation="sigmoid",
    output_activation="sigmoid",
    layer_sizes=[64],
    optimizer="rmsprop",
    loss="binary_crossentropy",
    dropout_probability=0.2)
start_time = timeit.default_timer()
predictor.fit_dataset(folds[0].train, n_training_epochs=200)
print timeit.default_timer() - start_time
peptides = folds[0].test.peptides
elute_pred = predictor.predict(peptides)
elute_true = folds[0].test.affinities
eq_(len(elute_pred), len(elute_true))
df = pd.DataFrame(columns=["true","pred"])
df["true"] = elute_true
df["pred"] = elute_pred

result = mhcflurry.class1_allele_specific.scoring.make_scores(elute_true, elute_pred, threshold_nm=0.6)
print result
plt.plot(elute_true, elute_pred, 'ro')
plt.xlim(xmin=-0.1, xmax=1.1)
plt.show()

# layer_sizes_list = [[8], [16], [32]]
# benchmark_df = pd.DataFrame({'layer_size': layer_sizes_list}, columns=['layer_size','tau','auc','f1'])
# result = {}
# for item in layer_sizes_list:
#     predictor = Class1BindingPredictor(
#         name="A0201",
#         embedding_output_dim=16,
#         activation="relu",
#         output_activation="sigmoid",
#         layer_sizes=item,
#         optimizer="rmsprop",
#         loss="binary_crossentropy",
#         dropout_probability=0.1)
#     predictor.fit_dataset(folds[0].train, n_training_epochs=200)
#     peptides = folds[0].test.peptides
#     elute_pred = predictor.predict(peptides)
#     elute_true = folds[0].test.affinities
#     result1 = mhcflurry.class1_allele_specific.scoring.make_scores(elute_true, elute_pred)

# print benchmark_df

 
