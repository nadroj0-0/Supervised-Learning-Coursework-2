from random import gauss

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import requests
from io import StringIO

#def row_kern (x1, kernel_func, x2=None, *kwargs):
#     x1 = np.asarray(x1)
#     n1 = len(x1)
#     if x2 is None:
#         x2 = x1
#         n2 = n1
#     else:
#         x2 = np.asarray(x2)
#         n2 = len(x2)
#     K = np.zeros(n1)
#     for i in range(n1):
#         K[i] = kernel_func(x1[i], x2)
#     return K

# Training and Test Data Splitter
def train_test_split (df, i=0) :
    train = df.sample(frac=0.8, random_state=i)
    test = df.drop(train.index)
    xtrain = np.array(train.drop(columns=[0]))
    xtest = np.array(test.drop(columns=[0]))
    ytrain = np.array(train[0])
    ytest = np.array(test[0])
    return xtrain, xtest, ytrain, ytest

def norm_p(x, p, axis=None) :
    if p > 0 :
        x=np.array(x)
        return np.sum(np.abs(x)**p, axis=axis)**(1/p)
    else:
        return 'Norm not defined : p <= 0'
def guass_ker_matrix(x1, x2):
    x1sq = np.sum(x1 ** 2, axis=1)[:, None]  # (n1, 1)
    x2sq = np.sum(x2 ** 2, axis=1)[None, :]  # (1, n2)
    num = x1sq + x2sq - 2 * (x1 @ x2.T)
    return (num)
def guass_kern (x1, x2, c=1, **kwargs):
    #diff = np.array(x1) - np.array(x2)
    x1sq = np.sum(x1 ** 2, axis=1)[:, None]  # (n1, 1)
    x2sq = np.sum(x2 ** 2, axis=1)[None, :]  # (1, n2)
    num = x1sq + x2sq -2*(x1@x2.T)
    # num = guass_ker_matrix(x1, x2)
    # diff = x1 - x2
    # num = norm_p(diff, 2, axis=-1)**2
    K = np.exp(-c * (num))
    return K
def poly_kern_matrix (x1, x2):
    return (x1@x2.T)
def poly_kern (x1, x2, d=1, **kwargs) :
    # matrix = poly_kern_matrix(x1, x2)
    # return (matrix)**d
    return (x1@x2.T) ** d

def block_kern (x1, x2, n1, n2, kernel_func, bSize = 250, **kwargs):
    noBlocks1 = int(np.ceil(n1 / bSize))
    noBlocks2 = int(np.ceil(n2 / bSize))
    blocks1 = []
    blocks2 = []
    for b in range(noBlocks1):
        start = b * bSize
        end = min((b + 1) * bSize, n1)
        blocks1.append((start, end))

    for b in range(noBlocks2):
        start = b * bSize
        end = min((b + 1) * bSize, n2)
        blocks2.append((start, end))
    K = np.zeros((n1, n2))
    for iStart, iEnd in blocks1:
        for jStart, jEnd in blocks2:
            K[iStart:iEnd, jStart:jEnd] = kernel_func(x1[iStart:iEnd, :], x2[jStart:jEnd, :], **kwargs)
    return K


def full_kern (x1, kernel_func, x2=None, bSize=250, **kwargs) :
    x1 = np.array(x1)
    n1 =len(x1)
    if x2 is None :
        x2 = x1
        n2 = n1
    else:
        x2 = np.array(x2)
        n2 = len(x2)

    # x1 = x1[:, None, :] #Make 3D and stitch together features dimension
    # x2 = x2[None, :, :]
    if n1 > bSize:
        K = block_kern(x1, x2, n1, n2, kernel_func=kernel_func, **kwargs)
    else :
        K = kernel_func(x1, x2, **kwargs) #Return full Kernel
    return K

def random_epoch (x, y):
    reordered = np.random.permutation(len(x))  # random arrray of digits 0-len(xtrain)-1
    xtrain_epoch, ytrain_epoch = [], []
    for i in range(len(reordered)):
        xtrain_epoch.append(x[reordered[i]])
        ytrain_epoch.append(y[reordered[i]])
    return xtrain_epoch, ytrain_epoch, reordered




def two_class_ker_perc_pred (alpha, x, y, target_class, K_precal_test):
    y_pred = []
    total_errors = 0
    for i in range(len(x)):
        K = K_precal_test[i, :]
        # K = K_precal[:,i]
        K = np.array(K)
        if (alpha @ K) >= 0:
            y_hat = 1
        else:
            y_hat = -1
        if y_hat != y[i]:
            total_errors = total_errors + 1
        y_pred.append(y_hat)
    total_errors = total_errors / len(x)
    return y_pred, total_errors

def confusion_matrix(y, yhat, Classes):
    confMatrix = np.zeros((len(Classes), len(Classes)))
    digitCounts = np.zeros(len(Classes))
    for i in range((len(y))):
        digitCounts[y[i]] = digitCounts[y[i]] + 1
        if yhat[i] != y[i]:
            confMatrix[y[i], yhat[i]] = confMatrix[y[i], yhat[i]] + 1
    for i in range((len(Classes))):
        if digitCounts[i] > 0:
            confMatrix[i, :] = confMatrix[i,:]/digitCounts[i]
    return confMatrix
def k_class_ker_perc_pred (alpha, x, y, k_classes, K_precal_test, confusion=False):
    y_pred = []
    total_errors = 0
    for i in range(len(x)) :
        y_hat = []
        K = K_precal_test[i, :]
        # K = K_precal[:,i]
        K = np.array(K)
        for j, k in enumerate (k_classes) :
            y_hat.append(alpha[j] @ K)
        y_pred.append(k_classes[y_hat.index(max(y_hat))])
        if y_pred[i] != y[i]:
            total_errors = total_errors + 1
    if confusion == True:
        confMatrix = confusion_matrix(y, y_pred, k_classes)
    else:
        confMatrix = None
    total_errors = total_errors/len(x)
    return y_pred, total_errors, confMatrix


def two_class_ker_perc_full (df, kernel_func, target_class, epoch = 1, K_precal_train = None, K_precal_test=None, split = None, **kwargs):
    if split is None : #If no prespecified train test splits given make from scratch
        xtrain, xtest, ytrain, ytest = train_test_split(df)
    else : #Otherwise if prespecified train test splits given unpack
        xtrain, xtest, ytrain, ytest = split
    ytrain = np.where(ytrain == target_class, 1, -1)  # Change ytrain 1 = target_class, -1 otherwise
    ytest = np.where(ytest == target_class, 1, -1)  # Same for ytest
    epoch_threshold = 3
    no_improvement_count = 0
    min_epoch_errors = np.inf
    alpha = np.zeros(len(xtrain))
    best_alpha = alpha.copy()
    if K_precal_train is None :
        # print('Pre computing Train Kernel')
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs)
        # print('Train Kernel Computed')
    total_errors_train = 0
    for e in range(epoch) :
        epoch_errors = 0
        #print(f"Epoch {e}, Class = {target_class}")
        for i in range(len(xtrain)) :
            alpha_i = alpha[:i]  # all previous alpha's
            K = K_precal_train[:i, i]
            if (alpha_i @ K) >= 0:
                y_hat = 1
            else :
                y_hat = -1
            if y_hat != ytrain[i] :
                alpha[i] = alpha[i] + ytrain[i]
                epoch_errors = epoch_errors + 1
                total_errors_train = total_errors_train + 1
        if epoch_errors < min_epoch_errors:
            min_epoch_errors = epoch_errors
            no_improvement_count = 0
            best_alpha = alpha.copy()
        else :
            no_improvement_count = no_improvement_count + 1
        if no_improvement_count >= epoch_threshold:
            #print(f"Stopped early at epoch {e} for class {target_class}")
            break
    total_errors_train = min_epoch_errors / len(xtrain)

    alpha = best_alpha.copy()
    if K_precal_test is None :
        # print('Pre Computing Test Kernel')
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
        # print('Test Kernel Computed')
        y_pred_test, total_errors_test = two_class_ker_perc_pred(alpha, xtest, ytest, target_class, K_precal_test)
        return alpha, total_errors_train, y_pred_test, total_errors_test
    # y_pred = []
    # total_errors = 0
    # alpha = np.array(alpha)
    # for i in range(len(xtest)) :
    #     K = K_precal_test[i, :]
    #     # K = K_precal[:,i]
    #     K = np.array(K)
    #     if (alpha @ K) >= 0:
    #         y_hat = 1
    #     else:
    #         y_hat = -1
    #     if y_hat != ytest[i]:
    #         total_errors = total_errors + 1
    #     y_pred.append(y_hat)
    # total_errors = total_errors/len(xtest)
    return alpha, total_errors_train, None, None


def k_class_ker_perc (df, kernel_func, k_classes, epoch=1, split=None, K_precal_train = None, K_precal_test = None, confusion=False, **kwargs):
    if split is None : xtrain, xtest, ytrain, ytest = train_test_split(df) #create test train splits
    else : xtrain, xtest, ytrain, ytest = split
    alpha, y_pred_test, train_errors, test_erros = [], [], [], []
    if K_precal_train is None:
        # print('pre computing  train kernel')
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs) #Precalculate the Kernel Matrix
        # print('train kernel computed')
    if K_precal_test is None:
        # print('pre computing test kernel')
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
        # print('test kernel computed')
        # print('kernels computed')
    for k in k_classes : #Loop through the k classes
        split = (xtrain, xtest, ytrain.copy(), ytest.copy()) #Store new dataset with y {1,-1} labels
        alpha_k, train_errors_k, _, _ = two_class_ker_perc_full(df, kernel_func, k, epoch=epoch, K_precal_train=K_precal_train, K_precal_test=K_precal_test, split = split) #Train 2 class classifier, (k vs rest)
        alpha.append(alpha_k) #Add k vs rest alpha values to alpha list
        train_errors.append(train_errors_k)
    y_pred_test, test_errors, confMatrix = k_class_ker_perc_pred(alpha, xtest, ytest, k_classes, K_precal_test, confusion)
    me_train = np.mean(train_errors)
    me_test = np.mean(test_errors)
    return alpha, y_pred_test, me_train, me_test, train_errors, test_errors, confMatrix

def cross_validation(xtrain, ytrain, folds, kernel_func, paramlist, k_classes, epoch_lim=1, **kwargs):
    positions = np.arange(len(xtrain))
    np.random.shuffle(positions)
    foldPositions = np.array_split(positions, folds)


    K_precal = {p: full_kern(xtrain, kernel_func=kernel_func, d=p, c=p) for p in paramlist}


    paraMatrix = paramlist

    cross_results = []
    results = np.zeros((folds, len(paramlist), 2))
    for j, p in enumerate(paramlist):
        K_full = K_precal[p]

        for i in range(folds):

            xtrain_fold = np.concatenate([foldPositions[f] for f in range(folds) if f != i])
            xtest_fold = foldPositions[i]


            K_train = K_full[np.ix_(xtrain_fold, xtrain_fold)]
            K_test = K_full[np.ix_(xtest_fold, xtrain_fold)]
            ytrain_fold = ytrain[xtrain_fold]
            ytest_fold = ytrain[xtest_fold]
            xtrain_fold = xtrain[xtrain_fold]
            xtest_fold = xtrain[xtest_fold]
            split_fold = (xtrain_fold, xtest_fold, ytrain_fold, ytest_fold)

            alpha, y_pred_test, me_train, me_test, _, _, _ = k_class_ker_perc(None, kernel_func, k_classes, epoch=epoch_lim, split=split_fold, K_precal_train = K_train, K_precal_test = K_test, d=p, c=p)
            results[i, j, 0] = me_train
            results[i, j, 1] = me_test

            # alpha = ker_ridg_sol(K_train, gam, ytrain_fold)
            # y_hat_train = K_train @ alpha
            # y_hat_test = K_test @ alpha
            # mse_train_fold = np.mean((ytrain_fold - y_hat_train) ** 2)
            # mse_test_fold = np.mean((ytest_fold - y_hat_test) ** 2)
            # mse = [mse_train_fold, mse_test_fold]
            # results.append(mse)
    avgme_train = np.mean(results[:, :, 0], axis=0)
    avgme_test = np.mean(results[:, :, 1], axis=0)
        # avgmse = [0, 0]
        #
        # for i in range(2):
        #     total = 0
        #     for j in range(folds):
        #         total += results[j][i]
        #     avgmse[i] = total / folds

    avgme_df = pd.DataFrame({'Param': paramlist, 'avg_me_train_fold': avgme_train, 'avg_me_test_fold': avgme_test})
    # cross_results = pd.concat(cross_results)
    avgme_df = avgme_df.sort_values(by ='avg_me_test_fold', ascending=True)
    min_me = avgme_df.iloc[0]
    return float(min_me['Param']), min_me['avg_me_train_fold'], min_me['avg_me_test_fold'], avgme_df


# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims_k_class_ker_perc(df, r, kernel_func, k_classes, param, epoch_lim = 30, splits=None,  crossVal=False, confusion=False, **kwargs) :
    if crossVal == True:
        results = np.zeros((r, 1, 2))
    else:
        results=np.zeros((r, len(param),2))
    if confusion == True:
        confMatrix = np.zeros((r, len(k_classes), len(k_classes)))
    else :
        confMatrix = None
    fig, ax = None, None
    best_params = []

    for i in range(r):
        if splits is None:
            xtrain, xtest, ytrain, ytest = train_test_split(df, i)
            split = (xtrain, xtest, ytrain.copy(), ytest.copy())
        else:
            split = splits[i]
            xtrain, xtest, ytrain, ytest = split
        if crossVal == True:
            minparam, _, _, crossValdf= cross_validation(xtrain, ytrain, 5, kernel_func, param, k_classes, epoch_lim=epoch_lim)

            best_params.append({
                "run": int(i),
                "best_param": float(minparam)
            })
            alpha, y_pred_test, me_train, me_test, _, _, confMatrix_r = k_class_ker_perc(None, kernel_func, k_classes, epoch=epoch_lim, split=split, confusion=confusion, d=minparam, c=minparam)
            results[i, 0, 0] = me_train
            results[i, 0, 1] = me_test
            if confusion == True:
                confMatrix[i] = confMatrix_r
        else :
            minparam = None
            for j, p in enumerate(param):
                # print(f'run {i}, parameter {p}')
        #print(f'For r = {i+1}, minsig = 2**{np.log2(minsig)}, mingam = 2**{np.log2(mingam)}')
                alpha, y_pred_test, me_train, me_test, _, _, _= k_class_ker_perc(None, kernel_func, k_classes, epoch = epoch_lim, split=split, d=p, c=p)
        #krm = ker_reg_sol(xtrain, ytrain, kernel_func, sig=minsig, gam=mingam)
                results[i,j,0] = me_train
                results[i,j,1] = me_test

    avgme_train= np.mean(results[:, :, 0], axis=0)
    avgsde_train = np.std(results[:, :, 0], axis=0)
    avgme_test = np.mean(results[:, :, 1], axis=0)
    avgsde_test = np.std(results[:, :, 1], axis=0)

    if crossVal == True:
        avgme_df = pd.DataFrame({'param': minparam, 'avg_me_train': avgme_train, 'avg_me_test': avgme_test, 'avg_me_train_sd': avgsde_train, 'avg_me_test_sd': avgsde_test})
    else:
        avgme_df = pd.DataFrame({'param': param, 'avg_me_train': avgme_train, 'avg_me_test': avgme_test, 'avg_me_train_sd': avgsde_train,'avg_me_test_sd': avgsde_test})
    # Save aggregated best params (Q5a) after all runs
    # if r > 1:  # only when doing repeated runs
    #     save_json({"runs": best_params}, "Q5/best_params.json")
    #
    # if plot == True and r>0:
    #     fig, ax = plot_crossVal_mse(crossValdf)
    #     return avgmse_df, alpha, fig, ax, minsig, mingam, crossValdf
    # else:
    #     return avgmse_df, alpha, minsig, mingam
    if confusion == True:
        avgconfMatrix = np.mean(confMatrix, axis=0)
        sdconfMatrix = np.std(confMatrix, axis=0)
        fullconfMatrix = np.stack((avgconfMatrix, sdconfMatrix), axis=2)
        return avgme_df, fullconfMatrix
    else :
        return avgme_df



url = 'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat'
data = np.loadtxt("zipcombo.dat")
df = pd.DataFrame(data)
dftest = df.sample(frac=0.50, random_state=0).reset_index(drop=True)
#df = pd.read_csv(url)
print(df.head())

print('Q1')

print('Q2')
start = time.time()
alpha, train_errors, y_pred, test_errors =two_class_ker_perc_full(df, guass_kern, 5, 100)
print(f'2 class classifier for target class 1 : {test_errors}')
Classes = [0,1,2,3,4,5,6,7,8,9]
alpha, y_pred_test, me_train, me_test, train_errors, test_errors, _ = k_class_ker_perc(df, guass_kern, Classes, 100)
print(f'k class classifier for target classes 0-9 : {test_errors}')
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")

print('Q3')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df = r_sims_k_class_ker_perc(df, runs, poly_kern, Classes, param=d)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print(f'Polynomial Kernel : {runs} run')
print(avgme_df)

print('Q4')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix = r_sims_k_class_ker_perc(df, runs, poly_kern, Classes, param=d, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print(f'Polynomial Kernel : {runs} run with cross validation')
print(avgme_df)

print('Q5')
print('Average Confusion Matrix')
print(avgconfMatrix)



print('Q6')

print('Q7')
print('(b)')
runs = 20
# c= np.linspace(0.045, 0.055, 50).tolist()
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df = r_sims_k_class_ker_perc(df, runs, guass_kern, Classes, param=c)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print(f'Polynomial Kernel : {runs} run')
print(avgme_df)

print('(c)')
runs = 20
# c= np.linspace(0.045, 0.055, 50).tolist()
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix = r_sims_k_class_ker_perc(df, runs, guass_kern, Classes, param=c, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print(f'Polynomial Kernel : {runs} run with cross validation')
print(avgme_df)

print('Average Confusion Matrix')
print(avgconfMatrix)
