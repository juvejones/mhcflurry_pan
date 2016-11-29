import mhcflurry
import pandas as pd
import numpy as np
import sklearn
import timeit, os
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(0)

from mhcflurry.dataset import Dataset
from mhcflurry import Class1BindingPredictor

from nose.tools import eq_
from numpy import testing

from mhcflurry.downloads import get_path

def hyperparameter_selection(model_value):
    predictor = Class1BindingPredictor(
        embedding_output_dim=32,
        activation="relu",
        output_activation="sigmoid",
        layer_sizes=[32],
        optimizer="rmsprop",
        loss="binary_crossentropy",
        dropout_probability=model_value)
    return predictor

def make_scores_five_datasets(filename, dropout_rate):
    dataset = Dataset.from_csv(filename)
    dataset_9mer = Dataset(dataset._df.ix[dataset._df.peptide.str.len() == 9])
    sizes = [2, 5, 10]
    test, train = dataset.random_split(n = int(len(dataset_9mer)/5))
    df = pd.DataFrame({'training size': sizes}, columns = ['training size', 'f1'])
    f1_list = []
    auc_list = []

    predictor = hyperparameter_selection(dropout_rate)

    for size in sizes:
        train_left, train_use = train.random_split(n = int(len(train)/size))
        start_time = timeit.default_timer()
        predictor.fit_dataset(train_use, n_training_epochs=250)
        print timeit.default_timer() - start_time
        peptides = test.peptides
        elute_pred = predictor.predict(peptides)
        elute_true = test.affinities
        result1 = mhcflurry.class1_allele_specific.scoring.make_scores(elute_true, elute_pred, threshold_nm=0.5)
        f1_list.append(result1['f1'])
        auc_list.append(result1['auc'])
    df['training size'] = len(train) - len(train)/df['training size']
    df['training size'] = df['training size'].astype(int)
    df['f1'] = f1_list
    df['auc'] = auc_list

    mhcflurry.class1_allele_specific.scoring.classification_report(elute_true, elute_pred, threshold = 0.5)
    for idx, value in enumerate(elute_pred):
        elute_pred[idx] = value + np.random.rand()/10
    print elute_pred 
    plt.scatter(elute_true, elute_pred, c='r', s=50, alpha=0.1)
    plt.xlim(xmin=-0.1, xmax=1.1)
    plt.ylim(ymin=-0.1, ymax=1.1)
    plt.show()
    
    return df	

#datasets = ["A0201", "B0702", "B3501", "B4403", "B5301", "B5701"]
datasets = ["B0702"]
symbols = ['r^-', 'g^-', 'b^-', 'ro-', 'go-', 'bo-']
dropout_rates = np.linspace(0.1, 0.5, num=5)
print dropout_rates

#plt.figure(1)
filename = os.path.join("./HLA-" + str(datasets[0]))
df = make_scores_five_datasets(filename, [0.2])
print df
#for idx, value in enumerate(dropout_rates):
#	filename = os.path.join("./HLA-" + str(datasets[0]))
#	df = make_scores_five_datasets(filename, value)
#	print df
	#plt.subplot(211)
	#plt.plot(df['training size'], df['auc'], symbols[idx], label=value)
	#plt.ylim(ymin = 0.7, ymax = 1.1)

	#plt.subplot(212)
	#plt.plot(df['training size'], df['f1'], symbols[idx], label=value)
	#plt.ylim(ymin = 0.0, ymax = 0.8)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.show()

# folds = mhcflurry.class1_allele_specific.cross_validation.cross_validation_folds(dataset_9mer, n_folds = 5)
# print len(folds[0].train)
# print len(folds[0].test)

# eq_(len(elute_pred), len(elute_true))
# df = pd.DataFrame(columns=["true","pred"])
# df["true"] = elute_true
# df["pred"] = elute_pred





 
