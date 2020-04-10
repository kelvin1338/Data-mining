#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


D = h5py.File('breast.h5', 'r')
X,Y,P = D['images'],np.array(D['counts']),np.array(D['id'])

np.savez_compressed('breast', X=X, Y=Y, P=P)
compressed = np.load('breast.npz')
X = compressed['X']
Y = compressed['Y']
P = compressed['P']


# In[2]:


#1i)
len(P) #1-13 training, 14-18 test, 7404 total
train_total = list(P >= 14).index(True)
print("Number of total test examples: " + str(train_total))
test_total = len(P) - list(P >= 14).index(True)
print("Number of total test examples: " + str(test_total))


# In[6]:


X_train = X[:5841]
X_test = X[5841:]

Y_train = Y[:5841]
Y_test = Y[5841:]

P_train = P[:5841]
P_test = P[5841:]


# Splitting the train and test data

# In[7]:


#1ii)
fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(3,3, figsize=(10,10))
ax1.imshow(X_train[0])
ax2.imshow(X_train[1])
ax3.imshow(X_train[2])
ax4.imshow(X_train[3])
ax5.imshow(X_train[4])
ax6.imshow(X_train[5])
ax7.imshow(X_train[6])
ax8.imshow(X_train[7])
ax9.imshow(X_train[8])


plt.show()


# From the images, we can the brown dots represent the lymphocytes, whereas the other dots generally represent noise

# In[8]:


#1iii)

bins_list = [0, 1, 5, 10, 20, 50, 200, 10000]
#n, bins, patches = plt.hist(Y_train, bins = bins_list, facecolor='blue')
#plt.show()

fig = plt.figure

fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.hist(Y, bins=bins_list, edgecolor='k')
ax = fig.add_subplot(211)
h,e = np.histogram(Y, bins=bins_list)
histogram = ax.bar(range(len(bins_list)-1),h, width=1, edgecolor='k')
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = '0'
labels[2] = '[1,5]'
labels[3] = '[6,10]'
labels[4] = '[11,20]'
labels[5] = '[21,50]'
labels[6] = '[51,200]'
labels[7] = '201+'
ax.set_xticklabels(labels)

for rectangle in histogram:
    height = rectangle.get_height()
    ax.text(rectangle.get_x() + rectangle.get_width()/2, 1.05*height, '%d' %int(height), ha='center', va='bottom')
#autolabel(histogram)


plt.show()


# In[9]:


from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
from skimage.exposure import rescale_intensity

cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                             'saddlebrown'])

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(3,3, figsize=(14,14))
ax1.imshow(X_train[0])
ax1.set_title("Image 1 (RGB)")
ax2.imshow(X_train[1])
ax2.set_title("Image 2 (RGB)")
ax3.imshow(X_train[2])
ax3.set_title("Image 3 (RGB)")

ax4.imshow(rgb2hed(X_train[0])[:,:,2], cmap=cmap_dab)
ax4.set_title("Image 1 (HED)")
ax5.imshow(rgb2hed(X_train[1])[:,:,2], cmap=cmap_dab)
ax5.set_title("Image 2 (HED)")
ax6.imshow(rgb2hed(X_train[2])[:,:,2], cmap=cmap_dab)
ax6.set_title("Image 3 (HED)")

def image_intensity_rescaler(i):
    d = rescale_intensity(rgb2hed(X_train[i])[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(d), d, d))
    return zdh

ax7.imshow(image_intensity_rescaler(0))
ax7.set_title("Image 1 (Stain separated, rescaled)")
ax8.imshow(image_intensity_rescaler(1))
ax8.set_title("Image 2 (Stain separated, rescaled)")
ax9.imshow(image_intensity_rescaler(2))
ax9.set_title("Image 3 (Stain separated, rescaled)")

plt.show()


# In[10]:


#1v)
from scipy.stats import entropy

X_brown = []
for i in X:
    X_brown.append(rgb2hed(i))

brownchannel_average = []
brownchannel_variance = []
brownchannel_entropy = []
for i in (X_brown):
    d = i[:,:,2]
    brownchannel_average.append(d.mean())
    brownchannel_variance.append(d.var())
    brownchannel_entropy.append(entropy(d).mean())


# In[11]:


plt.scatter(brownchannel_average, Y)


# There is fairly strong positive correlation between the brown channel average and the lymphocyte, therefore it is a useful feature.

# In[12]:


#1vi)

for i in range(1,19):
    print("Patient", i, "has", np.count_nonzero(P==i), "images")


# Some patients have a lot of images associated (i.e. 3rd patient has 958 images), whereas some patients have very few images associated (i.e. 5th patient has 44 images). The patients with a lot of images associated may skew the data and if unweighted, it will skew the regression model and reflect more of the patients with many images. For example, if all images from each patient contributed 1/18 towards the regression model then the regression model will be less biased.

# 1vii) The following metrics can be used to assess performance of regression models of our type:
# 
# - Sum of squared errors <br>
# - Mean squared error <br>
# - Mean absolute error <br>
# - Root mean square error <br>
# - R^2 score <br>
# 
# I would use the RMSE because it squares the error before averaging the, which is better when errors are disproportionally worse. R^2 is scaled between 0 and 1, which is easier to interpret but does not explicitly indicate how much errors deviate unlike RMSE. 
# 

# In[13]:


#Q2i a,b,c)

print("Average of brown channel in 1st image in sample: " + str(rgb2hed(X_train[0])[:,:,2].mean()))
print("Average of red channel in 1st image in sample: " + str(X_train[0][:,:,0].mean()))
print("Average of green channel in 1st image in sample: " + str(X_train[0][:,:,1].mean()))
print("Average of blue channel in 1st image in sample: " + str(X_train[0][:,:,2].mean()))



print("Variance of brown channel in 1st image in sample: " + str(rgb2hed(X_train[0])[:,:,2].var()))
print("Variance of red channel in 1st image in sample: " + str(X_train[0][:,:,0].var()))
print("Variance of green channel in 1st image in sample: " + str(X_train[0][:,:,1].var()))
print("Variance of blue channel in 1st image in sample: " + str(X_train[0][:,:,2].var()))



print("Entropy of brown channel in 1st image in sample: " + str((entropy(rgb2hed(X_train[0])[:,:,2]).mean())))
print("Entropy of red channel in 1st image in sample: " + str((entropy(X_train[0][:,:,0]).mean())))
print("Entropy of green channel in 1st image in sample: " + str((entropy(X_train[0][:,:,1]).mean())))
print("Entropy of blue channel in 1st image in sample: " + str((entropy(X_train[0][:,:,2]).mean())))


# In[14]:


#Q2id)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,6))

ax1.hist((rgb2hed(X_train[0]))[:,:,0].flat, color='brown')
ax1.set_title("Histogram of brown channel")

ax2.hist((X_train[0])[:,:,0].flat, color='red')
ax2.set_title("Histogram of red channel")
              
ax3.hist((X_train[0])[:,:,1].flat, color='green')
ax3.set_title("Histogram of green channel")
              
ax4.hist((X_train[0])[:,:,2].flat, color='blue')
ax4.set_title("Histogram of blue channel")


# The histograms correspond to the first image in the given sample

# In[15]:


#Q2i e)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA

X_PCA = MinMaxScaler().fit_transform(X_train[0].reshape(-1,1)) #PCA typically scales from 0 to 1, hence use Min Max scaler
x_new = IncrementalPCA(n_components=1).fit_transform(X_PCA)
print(x_new)


# In[16]:


#Last part of Q2i
redchannel_average = []
redchannel_variance = []
redchannel_entropy = []
for i in X:
    d = i[:,:,0]
    redchannel_average.append(d.mean())
    redchannel_variance.append(d.var())
    redchannel_entropy.append(entropy(d).mean())
    
    
greenchannel_average = []
greenchannel_variance = []
greenchannel_entropy = []
for i in X:
    d = i[:,:,1]
    greenchannel_average.append(d.mean())
    greenchannel_variance.append(d.var())
    greenchannel_entropy.append(entropy(d).mean())

    
bluechannel_average = []
bluechannel_variance = []
bluechannel_entropy = []
for i in X:
    d = i[:,:,2]
    bluechannel_average.append(d.mean())
    bluechannel_variance.append(d.var())
    bluechannel_entropy.append(entropy(d).mean())
    
    
# from scipy.stats import entropy

# X_brown = []
# for i in X:
#     X_brown.append(rgb2hed(i))

# brownchannel_average = []
# brownchannel_variance = []
# brownchannel_entropy = []
# for i in (X_brown):
#     d = i[:,:,2]
#     brownchannel_average.append(d.mean())
#     brownchannel_variance.append(d.var())
#     brownchannel_entropy.append(entropy(d).mean())


# In[17]:


#Question 2
fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]) = plt.subplots(4,3, figsize=(15,15))
ax1.scatter(brownchannel_average, Y, c='brown')
ax1.set_title("Average of brown count")
ax2.scatter(brownchannel_variance, Y, c='brown')
ax2.set_title("Variance of brown count")
ax3.scatter(brownchannel_entropy, Y, c='brown')
ax3.set_title("Entropy of brown count")

ax4.scatter(redchannel_average, Y, c='red')
ax4.set_title("Average of red count")
ax5.scatter(redchannel_variance, Y, c='red')
ax5.set_title("Variance of red count")
ax6.scatter(redchannel_entropy, Y, c='red')
ax6.set_title("Entropy of red count")

ax7.scatter(greenchannel_average, Y, c='green')
ax7.set_title("Average of green count")
ax8.scatter(greenchannel_variance, Y, c='green')
ax8.set_title("Variance of green count")
ax9.scatter(greenchannel_entropy, Y, c='green')
ax9.set_title("Entropy of green count")

ax10.scatter(bluechannel_average, Y, c='blue')
ax10.set_title("Average of blue count")
ax11.scatter(bluechannel_variance, Y, c='blue')
ax11.set_title("Variance of blue count")
ax12.scatter(bluechannel_entropy, Y, c='blue')
ax12.set_title("Entropy of blue count")


plt.show()


# In[18]:


print("Correlation coefficient of brown average vs count: " +  str(pearsonr(brownchannel_average, Y)[0]))
print("Correlation coefficient of brown variance vs count: " +  str(pearsonr(brownchannel_variance, Y)[0]))
print("Correlation coefficient of brown entropy vs count: " +  str(pearsonr(brownchannel_entropy, Y)[0]))
print("")
print("Correlation coefficient of red average vs count: " +  str(pearsonr(redchannel_average, Y)[0]))
print("Correlation coefficient of red variance vs count: " +  str(pearsonr(redchannel_variance, Y)[0]))
print("Correlation coefficient of red entropy vs count: " +  str(pearsonr(redchannel_entropy, Y)[0]))
print("")
print("Correlation coefficient of green average vs count: " +  str(pearsonr(greenchannel_average, Y)[0]))
print("Correlation coefficient of green variance vs count: " +  str(pearsonr(greenchannel_variance, Y)[0]))
print("Correlation coefficient of green entropy vs count: " +  str(pearsonr(greenchannel_entropy, Y)[0]))
print("")
print("Correlation coefficient of blue average vs count: " +  str(pearsonr(bluechannel_average, Y)[0]))
print("Correlation coefficient of blue variance vs count: " +  str(pearsonr(bluechannel_variance, Y)[0]))
print("Correlation coefficient of blue entropy vs count: " +  str(pearsonr(bluechannel_entropy, Y)[0]))


# It appears that the two main important features is brown average and blue average. Blue average is relatively strong and positively correlated with count number, whereas blue average is relatively strong and negatively correlated with count number.

# In[24]:


#Q2 latter part)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

def train_classifier(classifier, train_data, train_label, do_print=False, batch_size = 32):
    
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(train_data, train_label)

    rmse = []
    corr = []
    r_squared = []
    nn_classifiers = []
    fold_num = 0
    
    train_data = np.array(train_data)
    
    for train_index, test_index in skf.split(train_data, train_label):
        fold_num += 1
        
        y_train = train_label[train_index]
        y_test =  train_label[test_index]
        X_train = train_data[train_index]
        X_test = train_data[test_index]
        
        classifier.fit(X_train,y_train)
        nn_classifiers.append(classifier)

        y_pred = classifier.predict(X_test)
        rmse_calc = sqrt(mean_squared_error(y_test, y_pred))
        rmse.append(rmse_calc) #stores accuracy of each fold
        corr_calc = pearsonr(y_test, y_pred.flatten())[0]
        corr.append(corr_calc)
        r_squared_calc = r2_score(y_test, y_pred)
        r_squared.append(r_squared_calc)
        if do_print:
            print(f"Fold {fold_num}: RMSE: {rmse_calc}, CORR: {corr_calc}, R^2: {r_squared_calc}")
    
    if do_print:
        print(f"Mean RMSE is {np.mean(rmse)}")
        print(f"Std of RMSE is {np.std(rmse)}")
        print(f"Mean of correlation coefficient is {np.mean(corr)}")
        print(f"Std of correlation coefficient is {np.std(corr)}")
        print(f"Mean R^2 is {np.mean(r_squared)}")
        print(f"Std of R^2 is {np.std(r_squared)}")
    
    
    return (rmse, corr, r_squared)


# In[25]:


num_of_samples = 7404
num_of_features = 12
custom_feature = np.zeros((num_of_samples, num_of_features))
for i in range(num_of_samples):
    custom_feature[i, 0] = brownchannel_average[i]
    custom_feature[i, 1] = redchannel_average[i]
    custom_feature[i, 2] = greenchannel_average[i]
    custom_feature[i, 3] = bluechannel_average[i]
    custom_feature[i, 4] = brownchannel_variance[i]
    custom_feature[i, 5] = redchannel_variance[i]
    custom_feature[i, 6] = greenchannel_variance[i]
    custom_feature[i, 7] = bluechannel_variance[i]
    custom_feature[i, 8] = brownchannel_entropy[i]
    custom_feature[i, 9] = redchannel_entropy[i]
    custom_feature[i, 10] = greenchannel_entropy[i]
    custom_feature[i, 11] = bluechannel_entropy[i]


# In[26]:


custom_feature_train = custom_feature[:5841, :]
custom_feature_test = custom_feature[5841:,:]


# In[45]:


from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, hp
from hyperopt import Trials


def tune_classifier(args_dict):
    
    args = args_dict["args"]
    
    classifier_name = args["name"]
    
    if (classifier_name == "OLS"):
        n_jobs_OLS = args["n_jobs"]
        cls = LinearRegression(n_jobs = n_jobs_OLS)
        
    elif (classifier_name.startswith("MLP")):

            activation_MLP = args["activation"]
            solver_MLP = args["solver"]
            alpha_MLP = args["alpha"]
            cls = MLPClassifier(activation = activation_MLP, solver = solver_MLP, alpha = alpha_MLP)
            
    elif (classifier_name == "Ridge"):
            
            alpha_Ridge = args["alpha"]
            solver_Ridge = args["solver"]
            cls = Ridge(alpha = alpha_Ridge, solver = solver_Ridge)
            
    elif (classifier_name == "SVR"):

            C_SVR = args["C_SVR"]
            cls= SVR(C=C_SVR, cache_size=7000)
        
    
    rmse_all, corr_all, r2_all = train_classifier(cls, custom_feature_train, Y_train, do_print=False)
    rmse_mean, corr_mean, r2_mean = np.mean(rmse_all), np.mean(corr_all), np.mean(r2_all)
    return {
        "loss": 1.0-rmse_mean,        
        'status': STATUS_OK,
        "other_stuff":
        {
            "mean_rmse": rmse_mean,
            "mean_corr": corr_mean,
            "mean_r2": r2_mean,
            "fold_rmse": rmse_all,
            "fold_corr": corr_all,
            "fold_r2": r2_all,
           # "classifier_pickle": pickle.dumps(cls),
            "args_dict": args_dict
        }
    }


# Adapted code from first assignment

# In[48]:


rstate = np.random.RandomState(42)
OLS_trials = Trials()
best_OLS = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "OLS",
            "n_jobs": hp.choice("n_jobs", list(range(1,10,1))),                        
        },
         } ,
    trials=OLS_trials,
    algo=tpe.suggest,
    max_evals=25, rstate=rstate)

print(best_OLS)


# The returned output is the index of the selected attribute. For example, 'n_jobs':2 means optimal n_jobs equals 3.

# In[54]:


#From above, optimal parameter for OLS is when n_jobs = 2

from sklearn.linear_model import LinearRegression
classifier = LinearRegression(n_jobs = 2)
train_classifier(classifier, custom_feature_train, Y_train, True)
classifier_feature_predict = classifier.predict(custom_feature_test)
plt.scatter(Y_test, classifier_feature_predict)


# In[49]:


rstate = np.random.RandomState(42)
MLP_trials = Trials()
best_MLP = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "MLP",
            "activation": hp.choice("activation", ["identity", "logistic", "tanh", "relu"]),
            "solver": hp.choice("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": hp.choice("alpha", [0.001, 0.01, 0.1, 1, 10, 100])
        },
         } ,
    trials=MLP_trials,
    algo=tpe.suggest,
    max_evals=25, rstate=rstate)

print(best_MLP)


# In[56]:


#From above, optimal parameters for MLP is when activation = identity, alpha = 10 and solver = lbfgs
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(activation = "identity", alpha = 10, solver = "lbfgs")
train_classifier(classifier, custom_feature_train, Y_train, 32, True)
classifier_feature_predict = classifier.predict(custom_feature_test)
plt.scatter(Y_test, classifier_feature_predict)


# In[50]:


rstate = np.random.RandomState(42)
Ridge_trials = Trials()
best_Ridge = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "Ridge",
            "alpha": hp.choice("alpha", [0.001, 0.01, 0.1, 1, 10, 100]),
            "solver": hp.choice("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        },
         } ,
    trials=Ridge_trials,
    algo=tpe.suggest,
    max_evals=25, rstate=rstate)

print(best_Ridge)


# In[57]:


from sklearn.linear_model import Ridge
classifier = Ridge(alpha = 0.001, solver = "saga")
train_classifier(classifier, custom_feature_train, Y_train, 32, True)
classifier_feature_predict = classifier.predict(custom_feature_test)
plt.scatter(Y_test, classifier_feature_predict)


# In[51]:


rstate = np.random.RandomState(42)
SVR_trials = Trials()
best_SVR = fmin(fn=tune_classifier,
    space={
        "args": {
            "name": "SVR",
            "C_SVR": hp.choice("C_SVR", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        },
         } ,
    trials=SVR_trials,
    algo=tpe.suggest,
    max_evals=25, rstate=rstate)

print(best_SVR)


# In[58]:


from sklearn.svm import SVR
classifier = SVR(C=0.00001)
train_classifier(classifier, custom_feature_train, Y_train, 32, True)
classifier_feature_predict = classifier.predict(custom_feature_test)
plt.scatter(Y_test, classifier_feature_predict)


# In[32]:


#from future import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras import backend as K

#batch_size = 1000
num_classes = 1
epochs = 1

# input image dimensions
#img_rows, img_cols = 299, 299
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(Y_train.reshape(-1,1), num_classes)
#y_test = keras.utils.to_categorical(Y_test.reshape(-1,1), num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(299,299,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='linear'))

model.compile(loss="mean_absolute_error",
              optimizer=keras.optimizers.Adadelta(),
              metrics=['mean_absolute_error'])


# In[33]:


# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test)
#predictions = model.predict(X_test)

train_classifier(model, X_train, Y_train, True)
#rms = sqrt(mean_squared_error(Y_test, predictions))
#corr = pearsonr(y_test, y_pred)[0]
#r2_score(y_test, y_pred)


# In[34]:


plt.scatter(Y_test, model.predict(X_test))

