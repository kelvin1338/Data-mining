#!/usr/bin/env python
# coding: utf-8

# In[393]:


import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from numpy import linalg as LA
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[394]:


#1i)
train_data = np.loadtxt('Xtrain.csv', delimiter=' ')
train_label = np.loadtxt('Ytrain.csv', delimiter=' ')
test_data = np.loadtxt('Xtest.csv', delimiter=' ')
print("Total number of training data: " + str(len(train_label))) #3000 training examples
print("Total number of testing data: " + str(len(test_data))) #5000 testing examples


# In[395]:


plt.matshow(train_data[:20,:20])
plt.show()


# In[396]:


#1ii)
positive_count = np.sum(train_label==1)
negative_count = len(train_label) - positive_count

print("Number of positive examples: " + str(positive_count))
print("Number of negative examples: " + str(negative_count))


# 1iii) Use AUC-PR as main performance metric because data set is heavily imbalanced
# 
# 1iv) Accuracy = 1/k (k is number of classes) = 1/2, because classifier is randomly guessing between labels, hence after large number of guesses, hence accuracy converges to 1/2
# 
# 1v) AUC-ROC will be 0.5 because it ranks a random positive example higher than a random negative example 50% of the time.
# AUC-PR will be 1179/3000 because the area under the PR curve is directly proportional to class imbalance, because recall is plotted as a function of precision.

# In[397]:


#Q2i
def train_knn_with_data(train_data, train_label, k=1, do_print = True):        
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(train_data, train_label)

    accuracies = []
    auc_roc_scores = []
    auc_pr_scores = []
    nn_classifiers = []
    fold_num = 0
    for train_index, test_index in skf.split(train_data, train_label):
        fold_num += 1
        X_train, X_test = train_data[train_index], train_data[test_index] #recalling the ith data point
        y_train, y_test = train_label[train_index], train_label[test_index]
        sknn = KNeighborsClassifier(n_neighbors=k)
        sknn.fit(X_train,y_train)
        nn_classifiers.append(sknn)

        y_pred = sknn.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc) #stores accuracy of each fold
        auc_roc = roc_auc_score(y_test, y_pred)
        auc_roc_scores.append(auc_roc) #Appends AUC ROC score to list
        auc_pr = average_precision_score(y_test, y_pred)
        auc_pr_scores.append(auc_pr) #Appends AUC PR score to list
        if do_print:
            print(f"Fold {fold_num}: Accuracy: {acc}, AUC-ROC: {auc_roc}, AUC-PR: {auc_pr}")
    
    #Q2ii
    if do_print:
        print(f"Mean accuracy is {np.mean(accuracies)}")
        print(f"Std of accuracy is {np.std(accuracies)}")
        print(f"Mean AUC-ROC is {np.mean(auc_roc_scores)}")
        print(f"Std of AUC-ROC is {np.std(auc_roc_scores)}")
        print(f"Mean AUC-PR is {np.mean(auc_pr_scores)}")
        print(f"Std of AUC-PR is {np.std(auc_pr_scores)}")

    return np.mean(auc_pr_scores)


# Note that AUC-PR was used to assess performance because dataset is imbalanced

# In[398]:


train_knn_with_data(train_data, train_label, k=1) #Q2i + Q2ii


# In[399]:


#Q2iii)
#Fitting kNN where k=1 using normalized data
rstate = np.random.RandomState(42) 
pre_process_data_normalize = normalize(train_data)
train_knn_with_data(pre_process_data_normalize, train_label, k=1)


# In[400]:


#Fitting kNN where k=1 using standard scaler
rstate = np.random.RandomState(42) 
pre_process_data_StandardScaler = StandardScaler().fit_transform(train_data)
train_knn_with_data(pre_process_data_StandardScaler, train_label, k=1)


# In[401]:


#Fitting kNN where k=1 using MinMaxScaler
rstate = np.random.RandomState(42) 
pre_process_data_MinMaxScaler = MinMaxScaler().fit_transform(train_data)
train_knn_with_data(pre_process_data_MinMaxScaler, train_label, k=1)


# In[402]:


#Fitting kNN where k=1 using Power Transformer
rstate = np.random.RandomState(42) 
pre_process_data_PowerTransformer = PowerTransformer(method='yeo-johnson').fit_transform(train_data)
train_knn_with_data(pre_process_data_PowerTransformer, train_label, k=1)


# In[403]:


#Fitting kNN where k=1 using Quantile Transformer
rstate = np.random.RandomState(42) 
pre_process_data_QuantileTransformer = QuantileTransformer(output_distribution='uniform').fit_transform(train_data)
train_knn_with_data(pre_process_data_QuantileTransformer, train_label, k=1)


# After using It can be observed that preprocessing the data using normalizer, standard scaler, min max scaler, power transformer and quantile transformer generally achieves increased AUC-PR, boosting the number by approximately 0.01-0.02. 

# In[404]:


#Q2iv
from hyperopt import fmin, tpe, hp
rstate = np.random.RandomState(42) 
# FMIN minimises the output of train_knn_with_data, negating it turns minimums into maximums, hence we use 1 minus
best = fmin(fn=lambda k: 1.0-train_knn_with_data(train_data, train_label, k, False),
    space=hp.choice("k", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best)


# This returns the index, so {'k': 15} means k=16

# In[405]:


#Q3)
#Refined the code from Q2 such that it is generalised rather than specifically for kNN
def train_classifier(classifier, train_data, train_label, do_print=False):
    
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(train_data, train_label)

    accuracies = []
    auc_roc_scores = []
    auc_pr_scores = []
    nn_classifiers = []
    fold_num = 0
    
    
    for train_index, test_index in skf.split(train_data, train_label):
        fold_num += 1
        X_train, X_test = train_data[train_index], train_data[test_index] #recalling the ith data point
        y_train, y_test = train_label[train_index], train_label[test_index]
        classifier.fit(X_train,y_train)
        nn_classifiers.append(classifier) # originally was sknn

        y_pred = classifier.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc) #stores accuracy of each fold
        auc_roc = roc_auc_score(y_test, y_pred)
        auc_roc_scores.append(auc_roc)
        auc_pr = average_precision_score(y_test, y_pred)
        auc_pr_scores.append(auc_pr)
        if do_print:
            print(f"Fold {fold_num}: Accuracy: {acc}, AUC-ROC: {auc_roc}, AUC-PR: {auc_pr}")
    
    if do_print:
        print(f"Mean accuracy is {np.mean(accuracies)}")
        print(f"Std of accuracy is {np.std(accuracies)}")
        print(f"Mean AUC-ROC is {np.mean(auc_roc_scores)}")
        print(f"Std of AUC-ROC is {np.std(auc_roc_scores)}")
        print(f"Mean AUC-PR is {np.mean(auc_pr_scores)}")
        print(f"Std of AUC-PR is {np.std(auc_pr_scores)}")
    
    # Use PR to assess score because imbalanced data set
    return (auc_pr_scores, accuracies, auc_roc_scores)


# In[406]:


X = MinMaxScaler().fit_transform(train_data)  #Please see Q4 below for explanation
x_pca_reduced = PCA(n_components = 121).fit_transform(X)


# In[407]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from hyperopt import STATUS_OK

def get_preprocessed_data(pre_process_choice, data_to_change):
    if pre_process_choice=="none":
        process_data = data_to_change
    elif pre_process_choice == "normalise":
        process_data = normalize(data_to_change)
    elif pre_process_choice == "standardscaler":
        process_data = StandardScaler().fit_transform(data_to_change)
    elif pre_process_choice == "minmaxscaler":
        process_data = MinMaxScaler().fit_transform(data_to_change)
    elif pre_process_choice == "powertransformer":
        process_data = PowerTransformer(method='yeo-johnson').fit_transform(data_to_change)
    elif pre_process_choice == "quantiletransformer":
        process_data = QuantileTransformer(output_distribution='uniform').fit_transform(data_to_change)
    elif pre_process_choice == "x_pca_reduced": #This part is only relevant to Q4 and Q5
        X = MinMaxScaler().fit_transform(data_to_change)  #Please see Q4 below for explanation
        process_data = PCA(n_components = 121).fit_transform(X)
        
    return process_data


def tune_classifier(args_dict):
    
    args = args_dict["args"]
    pre_process_choice = args_dict["preprocess"]
    
    process_data = get_preprocessed_data(pre_process_choice, train_data)
        
    classifier_name = args["name"]
    
    if (classifier_name == "knn"):
        k = args["k"]
        cls = KNeighborsClassifier(n_neighbors=k)
        
    elif (classifier_name.startswith("svm")):
        c = args["C_svm"]
        svm_type = args["svm_type"]
        if svm_type == "linear":
            cls = LinearSVC(C=c)
        elif svm_type == "kernel":
            cls = SVC(C=c) #We use SVC for kernelized SVM because the default kernel of SVC(C=c) is rbf
            
    elif (classifier_name == "naivebayes"):
        bayes_type = args["bayes_type"]
        if bayes_type == "gaussian":
            cls = GaussianNB()
        elif bayes_type == "multinomial":
            cls = MultinomialNB()
        elif bayes_type == "bernoulli":
            cls = BernoulliNB()
        elif bayes_type == "complement":
            cls = ComplementNB()
            
    elif (classifier_name == "logisticregression"):
        c = args["C"]
        solver_method = args["solver_method"]
        cls= LogisticRegression(C=c, solver=solver_method)
        
    elif (classifier_name == "perceptron"):
        penalty_type = args["penalty_type"]
        a = args["a"]
        cls = Perceptron(penalty=penalty_type, alpha=a)
        
    elif (classifier_name == "xgboost"):
        
        learning_rate_xg = args["learning_rate"]
        max_depth_xg = args[ "max_depth"]
        min_child_weight_xg = args["min_child_weight"]
        gamma_xg = args["gamma"]
        colsample_bytree_xg = args["colsample_bytree"]
        cls = XGBClassifier(learning_rate = learning_rate_xg, max_depth = max_depth_xg, min_child_weight = min_child_weight_xg, gamma = gamma_xg, colsample_bytree = colsample_bytree_xg)
    
    pr_all, acc_all, roc_all = train_classifier(cls, process_data, train_label, do_print=False)
    pr_mean, acc_mean, roc_mean = np.mean(pr_all), np.mean(acc_all), np.mean(roc_all)
    return {
        "loss": 1.0-pr_mean,        
        'status': STATUS_OK,
        "other_stuff":
        {
            "mean_accuracy": acc_mean,
            "mean_auc_pr": pr_mean,
            "mean_auc_roc": roc_mean,
            "fold_accuracies": pr_all,
            "fold_auc_prs": pr_all,
            "fold_auc_rocs": roc_all,
            "classifier_pickle": pickle.dumps(cls),
            "args_dict": args_dict
        }
    }


# In[408]:


from tabulate import tabulate
import csv
def extract_hyperopt_best_result(trials):
    best_result = trials.results[trials.losses().index(min(trials.losses()))]
    preprocess_method = best_result["other_stuff"]["args_dict"]["preprocess"]
    classifier_type_name = best_result["other_stuff"]["args_dict"]["args"]["name"]
    mean_accuracy = best_result["other_stuff"]["mean_accuracy"]
    mean_auc_pr = best_result["other_stuff"]["mean_auc_pr"]
    mean_auc_roc = best_result["other_stuff"]["mean_auc_roc"]
    
    
    fold_accuracies = best_result["other_stuff"]["fold_accuracies"]
    fold_auc_prs = best_result["other_stuff"]["fold_auc_prs"]
    fold_auc_rocs = best_result["other_stuff"]["fold_auc_rocs"]
    
    return [classifier_type_name, preprocess_method, mean_accuracy, mean_auc_pr, mean_auc_roc] + fold_accuracies + fold_auc_prs +fold_auc_rocs

def print_csv(file_name, list_trials):
    
    best_results = [extract_hyperopt_best_result(trials) for trials in list_trials]
    with open(file_name, "w") as cur_file:
        writer = csv.writer(cur_file)
        acc_headers = [f"Acc Fold {i+1}" for i in range(5)]
        auc_pr_headers = [f"AUC-PR Fold {i+1}" for i in range(5)]
        auc_roc_headers = [f"AUC-ROC Fold {i+1}" for i in range(5)]
        headers=["Classifier", "Preprocessing Type", "Mean Accuracy", "Mean AUC-PR", "Mean AUC-ROC"] + acc_headers + auc_pr_headers +auc_roc_headers
        writer.writerow(headers)
        for results in best_results:
            writer.writerow(results)

def pretty_print_all_results(list_trials):
    
    best_results = [extract_hyperopt_best_result(trials) for trials in list_trials]
    acc_headers = [f"Acc Fold {i+1}" for i in range(5)]
    auc_pr_headers = [f"AUC-PR Fold {i+1}" for i in range(5)]
    auc_roc_headers = [f"AUC-ROC Fold {i+1}" for i in range(5)]
    headers=["Classifier", "Preprocessing Type", "Mean Accuracy", "Mean AUC-PR", "Mean AUC-ROC"] + acc_headers + auc_pr_headers +auc_roc_headers
    print(tabulate(best_results, headers=headers ))

def print_best_trial_results(trials):
    best_result = trials.results[trials.losses().index(min(trials.losses()))]
    
    for key, val in best_result["other_stuff"]["args_dict"].items():
        print(key, ": ", val)
        
    fold_accuracies = best_result["other_stuff"]["fold_accuracies"]
    fold_auc_prs = best_result["other_stuff"]["fold_auc_prs"]
    fold_auc_rocs = best_result["other_stuff"]["fold_auc_rocs"]
    print("Fold Accuracies: ", fold_accuracies)
    print("Fold AUC-PRs: ", fold_auc_prs)
    print("Fold AUC-ROCs: ", fold_auc_rocs)
    
    pretty_print_all_results([trials])
    


# For the following question (Q3), all the results required by the question are exported to a csv file called all_results. This file should be generated when the code of Q3 finishes running, however I have attached the all_results.csv to the ZIP file just in case.

# In[409]:


all_classifier_best_trials = []

# Q3: K nearest neighbour
rstate = np.random.RandomState(42)
preprocess_options = ["none", "normalise", "standardscaler", "minmaxscaler", "powertransformer", "quantiletransformer"]
knn_trials = Trials()
best_k = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "knn",
            "k": hp.choice("k", list(range(1,31,1))),                        
        },
        "preprocess": hp.choice("preprocess", preprocess_options) } ,
    trials=knn_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)


print_best_trial_results(knn_trials)
all_classifier_best_trials.append(knn_trials)


# In[410]:


#Q3: Linear SVM
rstate = np.random.RandomState(42) 
linear_svm_trials = Trials()
best_svm = fmin(fn=tune_classifier,
     space={
        "args": {
            "name": "svm_linear",
            "C_svm": hp.uniform("C", 1.0, 10.0),
            "svm_type": "linear"
        },
        "preprocess": hp.choice("preprocess", preprocess_options) },
    trials=linear_svm_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_svm)
all_classifier_best_trials.append(linear_svm_trials)


# In[411]:


#Q3: Kernelized SVM
rstate = np.random.RandomState(42) 
kernel_svm_trials = Trials()
best_svm = fmin(fn=tune_classifier,
     space={
        "args": {
            "name": "svm_kernel",
            "C_svm": hp.uniform("C", 1.0, 10.0),
            "svm_type": "kernel"
        },
        "preprocess": hp.choice("preprocess", preprocess_options) },
    trials=kernel_svm_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_svm)
all_classifier_best_trials.append(kernel_svm_trials)


# In[412]:


#Naive Bayes
rstate = np.random.RandomState(42) 
nb_trials = Trials()
best_NB = fmin(fn=tune_classifier,
    space={
        "args": 
        {
            "name":  "naivebayes",
            "bayes_type": hp.choice("bayes_type", ["gaussian", "multinomial", "bernoulli", "complement"])
        },
        "preprocess" : "minmaxscaler" 
    },
    trials=nb_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_NB)
all_classifier_best_trials.append(nb_trials)


# Note that Naive Bayes only takes non negative values as input, the only appropriate preprocessing method is MinMaxScaler which scales it to between 0 and 1

# In[413]:


#Q3: Logistic Regression
rstate = np.random.RandomState(42) 
lr_trials = Trials()
best_LR = fmin(fn=tune_classifier,
    space={
        "args":
        {
            "name": "logisticregression",
            "C":  hp.choice("C", [0.001, 0.01, 0.1, 1, 10, 100, 1000]),  #the C parameter controls the sparsity of the model
             "solver_method": hp.choice("solver_method", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        },
        "preprocess": hp.choice("preprocess", preprocess_options) },
    trials=lr_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_LR)
all_classifier_best_trials.append(lr_trials)


# In[414]:


#Perceptron
rstate = np.random.RandomState(42) 
percept_trials = Trials()
best_perceptron = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "perceptron",
            "penalty_type":  hp.choice("penalty_type", ["l2", "l1", "elasticnet", "None"]),
            "a": hp.choice("a", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        },
        "preprocess": hp.choice("preprocess_perceptron", preprocess_options)
    },
    trials=percept_trials,
    algo=tpe.suggest,
    max_evals=35, rstate=rstate)
print(best_perceptron)
all_classifier_best_trials.append(percept_trials)


# In[415]:


print_csv("all_results.csv", all_classifier_best_trials)


# Based on the results generated (see all_results.csv), the optimal classifier is kernelised SVM which has a AUC-PR of 0.665709

# In[416]:


#Question 4
X = MinMaxScaler().fit_transform(train_data) #PCA typically scales from 0 to 1, hence use Min Max scaler

x_new = PCA().fit_transform(X)

plt.scatter(x_new[:,0], x_new[:,1], c = train_label, s=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# The purple points are more distributed towards the left side of the graph where the value of PC1 is low and randomly scattered across the axis of PC2.
# The yellow points are concentrated near the middle of the graph where the value of PC1 is slightly above zero, and randomly scattered across the axis of PC2.
# There is separation observable by eye, mainly caused by PC1.

# In[417]:


#Scree plot
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.title('Scree plot')

plt.show()


# In[418]:


#Cumulative explained variance plot
cumulative_var = np.cumsum(pca.explained_variance_ratio_[:250])
plt.plot(cumulative_var)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative explained variance')

plt.show()


first_index = list(cumulative_var >= 0.95).index(True) #Finds the index where it starts to go from F to T
print(first_index)


# Hence we choose the first 121 Principal Components because they explain 95% of the variance.

# In[419]:


x_new = PCA().fit_transform(X)
x_pca_reduced = PCA(n_components = 121).fit_transform(X) #Transforming the data using PCA

#Repeating a similar procedure compared to question 3, except we now use the PCA dimension reduced data
rstate = np.random.RandomState(42) 
pca_trials = Trials()
best_pca_svm = fmin(fn=tune_classifier,
    space={"args":
           {
           "name": "svm_pca",
           "C_svm": hp.uniform("C", 1.0, 10.0),
           "svm_type": "kernel"
           },
           "preprocess": "x_pca_reduced"},
    trials=pca_trials,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_pca_svm)
print(pretty_print_all_results([pca_trials]))


# In[420]:


rstate = np.random.RandomState(42) 
best_pca_xgboost = fmin(fn=tune_classifier,
    space={
        "args":
        {
            "name": "xgboost",
            "learning_rate": hp.choice("learning_rate", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
            "max_depth": hp.choice("max_depth", [3,4,5,6,7,8,10,12,15]),
            "min_child_weight": hp.choice("min_child_weight", [1,3,5,7,9]),
            "gamma": hp.choice("gamma", [0.0, 0.1, 0.2, 0.3, 0.4]),
            "colsample_bytree":  hp.choice("colsample_bytree", [0.1, 0.3, 0.5, 0.7, 0.9])
    },
           "preprocess": "x_pca_reduced"} ,
    algo=tpe.suggest,
    max_evals=30, rstate=rstate)
print(best_pca_xgboost)


# In[424]:


# Q5
# FMIN minimises the output of train_knn_with_data, negating it turns minimums into maximums
rstate = np.random.RandomState(42) 
#Refined preprocess options to include PCA dimension reduced data
preprocess_options_new = ["none", "normalise", "standardscaler", "minmaxscaler", "powertransformer", "quantiletransformer", "x_pca_reduced"]
from hyperopt import Trials
space={
        "args": hp.choice("args", 
                    [   
                        #K means
                         {
                             "name": "knn",
                             "k": hp.choice("k",  list(range(1,31, 1)))
                         },
                        #SVM linear and Kernel
                         {
                             "name": "svm",
                             "C_svm": hp.uniform("C_svm", 1.0, 10.0),
                             "svm_type" : hp.choice("svm_type", ["linear", "kernel"])
                         },
#                         {
#                            "name":  "naivebayes",
#                            "bayes_type": hp.choice("bayes_type", ["gaussian", "multinomial", "bernoulli", "complement"])
#                         },
                         #Logistic Regression
                        {
                            "name": "logisticregression",
                            "C":  hp.choice("C", [0.001, 0.01, 0.1, 1, 10, 100, 1000]),  #the C parameter controls the sparsity of the model
                             "solver_method": hp.choice("solver_method", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        },
                        
                        
                        #Perceptron
                         {
            "name": "perceptron",
            "penalty_type":  hp.choice("penalty_type", ["l2", "l1", "elasticnet", "None"]),
            "a": hp.choice("a", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        },
                        
                       #XGboost
                 {
            "name": "xgboost",
            "learning_rate": hp.choice("learning_rate", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
            "max_depth": hp.choice("max_depth", [3,4,5,6,7,8,10,12,15]),
            "min_child_weight": hp.choice("min_child_weight", [1,3,5,7,9]),
            "gamma": hp.choice("gamma", [0.0, 0.1, 0.2, 0.3, 0.4]),
            "colsample_bytree":  hp.choice("colsample_bytree", [0.1, 0.3, 0.5, 0.7, 0.9])
    }
                    ]),
        "preprocess": hp.choice("preprocess", preprocess_options_new)}

trials = Trials()
best_all = fmin(fn=tune_classifier,
     space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=250, rstate = rstate)
print(best_all)

best_result = trials.results[trials.losses().index(min(trials.losses()))]


# In[425]:


for key, val in best_result.items():
    if key != "other_stuff":
        print(key, ": ", val)
    else:
        for stuff_key, stuff_val in best_result["other_stuff"].items():
            if stuff_key != "classifier_pickle":
                print(stuff_key, ": ", stuff_val)
                
best_classifier = pickle.loads(best_result["other_stuff"]["classifier_pickle"])
preprocess_type = best_result["other_stuff"]["args_dict"]["preprocess"]
pre_processed_data = get_preprocessed_data(preprocess_type, test_data)
predictions = best_classifier.predict(pre_processed_data)
with open('u1619893.csv', 'w') as writer:
    for i in predictions:
        writer.write(str(i) + "\n")


# The pipeline produced here is a collection of all classifiers and all preprocessing methods used in question 1, 2, 3 and 4 . Naive Bayes was disabled because it performed poorly based on question 3 and hence to save running time. The results of the optimal pipeline is shown in the output above, being kernelized SVM with C = 2.59, preprocess type = powertransformer, achieving an accuracy of 0.8110, AUC-PR of 0.6711 and AUC-ROC of 0.8051.
