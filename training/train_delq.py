from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
import joblib
import pickle
import json
#from train import split_data, train_model, get_model_metrics

#Lib from aws

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from io import StringIO, BytesIO
#import time
import json
#import os

#import warnings
#import pytz
from datetime import datetime
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import tempfile
#from tqdm import tqdm
# from sagemaker.amazon.amazon_estimator import get_image_uri
#from time import gmtime, strftime

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression





def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="insurance_model.pkl",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,a new version of the dataset will be registered"),
        default="insurance",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="insurance_dataset",
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    print("Getting training parameters")

    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        #run.parent.log(k, v)
      

    # Get the dataset
    if (dataset_name):
        if (data_file_path == 'none'):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name)  # NOQA: E402, E501
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       "workspaceblobstore",
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    #run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
    df = dataset.to_pandas_dataframe()
	
	# from aws
	
	df = data.iloc[:9900, :]
	test_data = data.iloc[9900:, :]
	
	
	TARGET_COL = "DaysDel"
	y = df[TARGET_COL]
	X = df.drop(TARGET_COL, axis=1)
	#X.to_csv("preprocessed.csv", index=None, header=None )
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=22)
	train = X_train.copy()
	train['DaysDel'] = y_train
	val = X_test.copy()
	val['DaysDel'] = y_test
	
	
	rr = RandomForestClassifier()
	lg = LGBMClassifier()
	xg = XGBClassifier(n_estimators=10, eval_metric='merror')
	dt = DecisionTreeClassifier()
	ad = AdaBoostClassifier()
	nb = GaussianNB()
	kn = KNeighborsClassifier()
	logr = LogisticRegression()
	
	
	from sklearn.metrics import accuracy_score
	
	model_list = [logr, rr, lg, xg, dt, ad, nb, kn]\
	scores = []
	model_objects = {}
	for x in tqdm(model_list):
		x.fit(X_train, y_train)
		y_pred = x.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		scores.append(acc)
		
	names = ['Log', 'RF', 'LightGBM', 'XGB', 'DT', 'Adaboost', 'NaiveBayes', 'KNN']
	md = pd.DataFrame(scores,  index=names).rename(columns={0: 'accuracy'})
	acc = md.accuracy.to_list()
	metrics = [] 
	allmodels = []
	for n, s in zip(names, scores):
		am = { 'Model Name': n, 'Score': s }
		allmodels.append(am)
	model_dict = {}
	bmodel = []
	for name, model, score in zip(names, model_list,acc):
		model_dict[model] = name
		bmodel.append(model_dict)
	best_model = max(zip(model_dict.values(), model_dict.keys()))[1]
	best_model_name = model_dict[best_model]
	for name, model, score in zip(names, model_list,acc):
		model_dict[model] = score
		bmodel.append(model_dict)
	best_model = max(zip(model_dict.values(), model_dict.keys()))[1]
	best_model_score = max(zip(model_dict.values(), model_dict.keys()))[0]
	
	metrics.append(best_model_name)
	metrics.append(best_model_score)
	
	hyperparameter_space = {'max_depth':[None,4,6,8],'min_child_weight':[0.5, 1, 2]}
	from sklearn.model_selection import GridSearchCV
	gs = GridSearchCV(best_model, param_grid=hyperparameter_space,scoring="accuracy",n_jobs=2, cv=5, return_train_score=False)

	gs.fit(X_train, y_train, verbose=3) 
	
	best_model = gs.best_estimator_
	gs.best_estimator_._Booster.save_model(best_model_file_name)
	output_filename = "bestmodel.tar.gz"
	file_to_archive = best_model_file_name
	
	y_pred2 = gs.predict(X_test)
	score = gs.best_score_
	metrics.append(score)
	
	
	
	
	
	
	
	
	
	
	
	
    #data = split_data(df)

    # Train the model
    #model = train_model(data, train_args)

    # Evaluate and log the metrics returned from the train function
    #metrics = get_model_metrics(model, data)


    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', best_model)
    joblib.dump(value=best_model, filename=output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    # upload the model file explicitly into artifacts
    print("Uploading the model into run artifacts...")
    run.upload_file(name="./outputs/best_model/" + best_model, path_or_stream=output_path)
    print("Uploaded the model {} to experiment {}".format(best_model, run.experiment.name))
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded ")
    print(run.get_file_names())

    run.complete()


if __name__ == '__main__':
    main()
