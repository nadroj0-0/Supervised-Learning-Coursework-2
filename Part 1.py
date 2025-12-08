import numpy as np
import pandas as pd
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

def guass_kern (x1, x2, beta=1,  sig=np.sqrt(0.5) ):
    #diff = np.array(x1) - np.array(x2)
    diff = x1 - x2
    num = norm_p(diff, 2, axis=-1)**2
    #Kij = np.exp(-beta*(num/(2*(sig**2))))
    K = np.exp(-beta * (num / (2 * (sig ** 2))))
    #return Kij
    return K

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
            K[iStart:iEnd, jStart:jEnd] = kernel_func(x1[iStart:iEnd, :, :], x2[:, jStart:jEnd, :], **kwargs)
    return K


def full_kern (x1, kernel_func, x2=None, element=None, **kwargs) :
    x1 = np.array(x1)
    n1 =len(x1)
    if x2 is None :
        x2 = x1
        n2 = n1
    else:
        x2 = np.array(x2)
        n2 = len(x2)
    if element is not None : #Return specific element of Kernel
        i = element[0]
        j = element[1]
        K = kernel_func(x1[i], x2[j], **kwargs)
    else :
        x1 = x1[:, None, :] #Make 3D and stitch together features dimension
        x2 = x2[None, :, :]
        if n1 > 250:
            K = block_kern(x1, x2, n1, n2, kernel_func=kernel_func, **kwargs)
        else :
            K = kernel_func(x1, x2, **kwargs) #Return full Kernel
    #K = np.zeros((n1,n2))
    #for i in range(n1) :
        #print(f'Calculating K[{i},..]')
        #for j in range(n2) :
            #K[i,j] = kernel_func(x1[i], x2[j], sig)
    return K

def random_epoch (x, y):
    reordered = np.random.permutation(len(x))  # random arrray of digits 0-len(xtrain)-1
    xtrain_epoch, ytrain_epoch = [], []
    for i in range(len(reordered)):
        xtrain_epoch.append(x[reordered[i]])
        ytrain_epoch.append(y[reordered[i]])
    return xtrain_epoch, ytrain_epoch, reordered

# Solves least squares equation, y = θw, returns weights w
# def ker_ridg_sol (K, gamma, y) :
#     l = len(y)
#     # return np.linalg.inv(theta.T@theta)@theta.T@y
#     alpha = np.linalg.solve((K+(gamma*l*np.identity(l))), y)
#     return alpha
#
# def ker_reg_sol(x,y, kernel_func, sig, gam) :
#     # theta = poly_basis(x,k)
#     K = full_kern(x, kernel_func=kernel_func, sig=sig)
#     # K = kernel_func(x, **kwargs)
#     # print('Theta = ')
#     # print(theta)
#     alpha = ker_ridg_sol(K, gam, y)
#     y_hat = K @ alpha
#     mse = np.mean((y - y_hat)**2)
#     return [alpha, y_hat, mse]

# def two_class_ker_perc_full (df, kernel_func, rand_epoch = False):
#     xtrain, ytrain, xtest, ytest = train_test_split(df)
#     alpha = np.zeros(len(xtrain))
#     epoch = 2 #Just set for example sake, will do further research into optimal value later
#     xtrain_online = []
#     ytrain_online = []
#     K_cache = []
#     for e in range(epoch) :
#         if rand_epoch == True and e != 0 :
#             xtrain_epoch, ytrain_epoch, reorder = random_epoch(xtrain, ytrain)
#         else :
#             xtrain_epoch, ytrain_epoch = xtrain, ytrain
#         for i in range(len(xtrain_epoch)) :
#             t = (e*len(xtrain))+i
#             xtrain_online.append(xtrain_epoch[i])
#             ytrain_online.append(ytrain_epoch[i])
#             if e == 0 :
#                 #Computing K(x_i, xt) for all i> t
#                 K = row_kern(xtrain, kernel_func=kernel_func, x2=xtrain_epoch[i])
#                 K_cache.append(K)
#             elif rand_epoch == True :
#                 K = K_cache[reorder[i]]
#             else :
#                 K = K_cache[i]
#             if (alpha @ K) >= 0
#                 y_hat = 1
#             else :
#                 y_hat = -1
#             if y_hat * y < 0 :
#                 alpha = alpha + 1
#     return

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

def k_class_ker_perc_pred (alpha, x, y, k_classes, K_precal_test):
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
    total_errors = total_errors/len(x)
    return y_pred, total_errors


def two_class_ker_perc_full (df, kernel_func, target_class, epoch = 1, K_precal_train = None, K_precal_test=None, split = None, rand_epoch = False):
    if split is None : #If no prespecified train test splits given make from scratch
        xtrain, xtest, ytrain, ytest = train_test_split(df)
    else : #Otherwise if prespecified train test splits given unpack
        xtrain = split[0]
        xtest = split[1]
        ytrain = split[2]
        ytest = split[3]
    epoch_threshold = 5
    no_improvement_count = 0
    min_epoch_errors = float('inf')
    ytrain = np.where(ytrain == target_class, 1, -1)  # Change ytrain 1 = target_class, -1 otherwise
    ytest = np.where(ytest == target_class, 1, -1)  # Same for ytest
    alpha = np.zeros(len(xtrain))
    best_alpha = alpha.copy()
    xtrain_online = []
    ytrain_online = []
    if K_precal_train is None :
        print('Pre computing Train Kernel')
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func)
        print('Train Kernel Computed')
    for e in range(epoch) :
        epoch_errors = 0
        #print(f"Epoch {e}, Class = {target_class}")
        for i in range(len(xtrain)) :
            #xtrain_online.append(xtrain[i])
            #ytrain_online.append(ytrain[i])
            alpha_i, K = [], []
            #for j in range(i) :
            #    K.append(K_precal_train[j,i])
            #    alpha_i.append(alpha[j])
            ##K = K_precal[:,i]
            #K = np.array(K)
            #alpha_i = np.array(alpha_i)
            alpha_i = alpha[:i]  # all previous alpha's
            K = K_precal_train[:i, i]
            if (alpha_i @ K) >= 0:
                y_hat = 1
            else :
                y_hat = -1
            if y_hat != ytrain[i] :
                alpha[i] = alpha[i] + ytrain[i]
                epoch_errors = epoch_errors + 1
        if epoch_errors < min_epoch_errors:
            min_epoch_errors = epoch_errors
            no_improvement_count = 0
            best_alpha = alpha.copy()
        else :
            no_improvement_count = no_improvement_count + 1
        if no_improvement_count >= epoch_threshold:
            print(f"Stopped early at epoch {e} for class {target_class}")
            break
    alpha = best_alpha.copy()
    if K_precal_test is None :
        print('Pre Computing Test Kernel')
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain)
        print('Test Kernel Computed')
        y_pred, total_errors = two_class_ker_perc_pred(alpha, xtest, ytest, target_class, K_precal_test)
        return alpha, y_pred, total_errors
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
    return alpha, None, None


def k_class_ker_perc (df, kernel_func, k_classes, epoch=1, rand_epoch = False):
    xtrain, xtest, ytrain, ytest = train_test_split(df) #create test train splits
    alpha, y_pred, errors = [], [], []
    print('pre computing  train kernel')
    K_precal_train = full_kern(xtrain, kernel_func=kernel_func) #Precalculate the Kernel Matrix
    print('train kernel computed')
    print('pre computing test kernel')
    K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain)
    print('test kernel computed')
    print('kernels computed')
    for k in k_classes : #Loop through the k classes
        split = (xtrain, xtest, ytrain.copy(), ytest.copy()) #Store new dataset with y {1,-1} labels
        alpha_k, _, _ = two_class_ker_perc_full(df, kernel_func, k, epoch=epoch, K_precal_train=K_precal_train, K_precal_test=K_precal_test, split = split, rand_epoch=rand_epoch) #Train 2 class classifier, (k vs rest)
        alpha.append(alpha_k) #Add k vs rest alpha values to alpha list
    y_pred, errors = k_class_ker_perc_pred(alpha, xtest, ytest, k_classes, K_precal_test)
    return alpha, y_pred, errors

url = 'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat'
data = np.loadtxt("zipcombo.dat")
df = pd.DataFrame(data)
#df = pd.read_csv(url)
print(df.head())

print('Q1')

print('Q2')

print('Q3')
runs = 20
d=[1,2,3,4,5,6,7]
alpha, y_pred, total_erros =two_class_ker_perc_full(df, guass_kern, 1, 100)
print(f'2 class classifier for target class 1 : {total_erros}')
Classes = [0,1,2,3,4,5,6,7,8,9]
alpha, y_pred, total_erros = k_class_ker_perc(df, guass_kern, Classes, 100)
print(f'k class classifier for target classes 0-9 : {total_erros}')