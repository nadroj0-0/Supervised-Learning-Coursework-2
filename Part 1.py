from random import gauss

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import requests
from io import StringIO

import json
import os
# Saving results and figures functions
#Saves a figure in Figures folder
def save_figure(fig, filename: str):
    path = os.path.join("outputs", "figures", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")

#Saves a json file in results folder
def save_json(data: dict, filename: str):
    path = os.path.join("outputs", "results", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def latex_escape(s: str) -> str:
    # If user already provided LaTeX math, don't touch it.
    if "$" in s:
        return s

    repl = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),   # <-- fine for normal text, but math handled above
        ("±", r"$\pm$"),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    return s


def wrap_latex_table_small(path):
    with open(path, "r") as f:
        content = f.read()

    wrapped = (
        "\\begingroup\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        + content +
        "\n\\endgroup\n"
    )

    with open(path, "w") as f:
        f.write(wrapped)

def postprocess_q3(*, avgme_df: pd.DataFrame, model_id: str, param_symbol: str = "d"):
    """
    Post-process Q3-style results with automatic filenames.

    Parameters
    ----------
    avgme_df : pd.DataFrame
        Output from r_sims_k_class_ker_perc
    model_id : str
        e.g. "OvR_poly", "OvR_gauss", "OvO_poly"
    param_symbol : str
        "d" (polynomial, default) or "c" (Gaussian)
    """

    def c_to_pow2_tex(c_val: float) -> str:
        # assumes c_val = 2^(-b/2) for integer b
        b = int(round(-2 * np.log2(c_val)))
        return f"$2^{{-{b}/2}}$"

    question_id = f"{model_id}_Q3"
    if param_symbol == "c":
        display_param = avgme_df["param"].map(c_to_pow2_tex)
        json_param = avgme_df["param"].astype(float).tolist()
        param_col = "c"
    else:
        display_param = avgme_df["param"]
        json_param = avgme_df["param"].tolist()
        param_col = "d"

    q3_table = pd.DataFrame({
        param_col: display_param,
        "mean train error rates (%)": (
                                              100 * avgme_df["avg_me_train"]
                                      ).round(3).astype(str) + " ± " + (100 * avgme_df["avg_me_train_sd"]).round(
            3).astype(str),
        "mean test error rates (%)": (
                                             100 * avgme_df["avg_me_test"]
                                     ).round(3).astype(str) + " ± " + (100 * avgme_df["avg_me_test_sd"]).round(
            3).astype(str),
    })

    save_outputs(
        question_id=question_id,
        tables={"summary": q3_table},
        jsons={
            "train_test_stats": {
                param_col: json_param,  # <- decimals for plotting etc.
                "train_mean": avgme_df["avg_me_train"].tolist(),
                "train_sd": avgme_df["avg_me_train_sd"].tolist(),
                "test_mean": avgme_df["avg_me_test"].tolist(),
                "test_sd": avgme_df["avg_me_test_sd"].tolist(),
            }
        }
    )



def postprocess_experiment(
    *,
    question_prefix: str,
    avgme_df=None,
    best_params=None,
    avgconfMatrix=None,
    hard=None,
    hard_pred=None,
    classes=None,
    param_symbol: str = "d"
):
    """
    Post-processes experiment outputs:
    - Q4-style CV table (if best_params provided)
    - Q5 confusion matrix (LaTeX + JSON, if avgconfMatrix provided)
    - Q6 hard examples figure (if hard provided)
    """

    # -------------------------
    # Q4: CV runs table
    # -------------------------
    def df_to_tex_with_index(df, filename: str):
        path = os.path.join("outputs", "tex", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_latex(
            path,
            index=True,          # <-- key difference
            header=True,
            escape=False
        )

    if best_params is not None:
        q4_table = pd.DataFrame({
            "run": [(bp["run"] + 1) for bp in best_params],
            f"{param_symbol}*": [bp["best_param"] for bp in best_params],
            "test error rate (%)": [(100 * bp["test_error"]).round(4) for bp in best_params]
        })

        save_outputs(
            question_id=f"{question_prefix}_Q4",
            tables={"cv_runs": q4_table}
        )

    # -------------------------
    # Q5: Confusion matrix
    # -------------------------
    if avgconfMatrix is not None and classes is not None:
        avgconfMatrix_raw = avgconfMatrix.copy()

        # Build formatted % mean ± std table
        conf_pct = np.empty((len(classes), len(classes)), dtype=object)
        for i in range(len(classes)):
            for j in range(len(classes)):
                mean = avgconfMatrix_raw[i, j, 0] * 100
                sd   = avgconfMatrix_raw[i, j, 1] * 100
                conf_pct[i, j] = f"{mean:.3f} ± {sd:.3f}"

        conf_df = pd.DataFrame(
            conf_pct,
            index=[str(c) for c in classes],
            columns=[f"\\textbf{{{c}}}" for c in classes]
        )
        conf_df.index.name = r" "

        # Save LaTeX (index ON)
        df_to_tex_with_index(conf_df, f"{question_prefix}_Q5_confusion.tex")
        wrap_latex_table_small(f"outputs/tex/{question_prefix}_Q5_confusion.tex")

        # Save JSON
        conf_json = {
            "classes": classes,
            "mean_percent": (avgconfMatrix_raw[:, :, 0] * 100).tolist(),
            "std_percent":  (avgconfMatrix_raw[:, :, 1] * 100).tolist()
        }

        save_outputs(
            question_id=f"{question_prefix}_Q5",
            jsons={"avg_confusion_matrix_pct": conf_json}
        )

    # -------------------------
    # Q6: Hard examples
    # -------------------------
    if hard is not None and hard_pred is not None:
        fig, ax = plot_pixel_images(hard, hard_pred)

        save_outputs(
            question_id=f"{question_prefix}_Q6",
            figures={"hard_examples": fig}
        )


#Saves a latex file in the tex folder for writing up in latex
def df_to_tex(df, filename: str):
    path = os.path.join("outputs", "tex", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    df = df.copy()
    df.columns = df.columns.astype(str)
    df.columns = [latex_escape(c) for c in df.columns]

    # Escape every string cell (only object columns)
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].map(lambda x: latex_escape(str(x)) if pd.notna(x) else x)
    df.to_latex(path, index=False, header=True, float_format="%.4f", escape=False)

def json_to_tex(data: dict, filename: str):
    rows = []
    for k, v in data.items():
        rows.append([k, v])

    df = pd.DataFrame(rows)
    path = os.path.join("outputs", "tex", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_latex(path, index=False, header=False, float_format="%.4f", escape=False)

def save_outputs(
    *,
    question_id: str,
    tables: dict = None,
    figures: dict = None,
    jsons: dict = None
):
    """
    Saves experiment outputs (tables, figures, jsons) in a structured way.

    Parameters
    ----------
    question_id : str
        e.g. "Q3", "Q4", "Q8_ovo_poly"
    tables : dict[str, pd.DataFrame]
        {name: dataframe}
    figures : dict[str, matplotlib.figure.Figure]
        {name: figure}
    jsons : dict[str, dict]
        {name: json_serialisable_object}
    """

    if tables:
        for name, df in tables.items():
            df_to_tex(df, f"{question_id}_{name}.tex")

    if figures:
        for name, fig in figures.items():
            save_figure(fig, f"{question_id}_{name}.png")

    if jsons:
        for name, data in jsons.items():
            save_json(to_json_safe(data), f"{question_id}_{name}.json")


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
    trainpos = train.index.to_numpy()
    testpos = test.index.to_numpy()
    return xtrain, xtest, ytrain, ytest, trainpos, testpos

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


def confusion_matrix(y, yhat, Classes):
    confMatrix = np.zeros((len(Classes), len(Classes)))
    digitCounts = np.zeros(len(Classes))
    for i in range((len(y))):
        digitCounts[int(y[i])] = digitCounts[int(y[i])] + 1
        if yhat[i] != y[i]:
            confMatrix[int(y[i]), int(yhat[i])] = confMatrix[int(y[i]), int(yhat[i])] + 1
    for i in range((len(Classes))):
        if digitCounts[i] > 0:
            confMatrix[i, :] = confMatrix[i,:]/digitCounts[i]
    return confMatrix

def confusion_matrix_df(confMatrix, Classes):
    row, col,_ = confMatrix.shape
    df = np.empty((row, col), dtype=object)
    for i in range(row):
        for j in range(col):
            me = confMatrix[i, j,0]
            sd = confMatrix[i, j,1]
            if me != 0:
                me = float(f'{me:.{4}g}')
                me = np.format_float_scientific(me, precision=4, exp_digits=2, trim='k')
                me = me.replace('e', 'x10')
            if sd != 0:
                sd = float(f'{sd:.{4}g}')
                sd = np.format_float_scientific(sd, precision=4, exp_digits=2, trim='k')
                sd = sd.replace('e', 'x10')
            df[i,j] = f'{me}±{sd}'
    return pd.DataFrame(df, index=Classes, columns=Classes)

def two_class_ker_perc_pred (alpha, x, y, xtrain_online, target_class, K_precal_test):
    y_pred = []
    total_errors = 0
    A = np.array(alpha)
    for i in range(len(x)):
        if len(alpha) == 0:
            score = 0
        else :
            K = K_precal_test[i, xtrain_online]
            score = (A @ K)
        # K = K_precal_test[i, :]
        # K = K_precal[:,i]
        # K = np.array(K)
        if score >= 0:
            y_hat = 1
        else:
            y_hat = -1
        if y_hat != y[i]:
            total_errors = total_errors + 1
        y_pred.append(y_hat)
    total_errors = total_errors / len(x)
    return y_pred, total_errors

def two_class_ker_perc_full (df, kernel_func, target_class, epoch = 1, K_precal_train = None, split = None, trainhelper=False, **kwargs):
    if split is None : #If no prespecified train test splits given make from scratch
        xtrain, xtest, ytrain, ytest, _, _ = train_test_split(df)
    else : #Otherwise if prespecified train test splits given unpack
        xtrain, xtest, ytrain, ytest = split
    ytrain = np.where(ytrain == target_class, 1, -1)  # Change ytrain 1 = target_class, -1 otherwise
    ytest = np.where(ytest == target_class, 1, -1)  # Same for ytest
    epoch_threshold = 3
    no_improvement_count = 0
    min_epoch_errors = np.inf
    # alpha = np.zeros(len(xtrain))
    alpha = []
    xtrain_online = []
    best_alpha = alpha.copy()
    if K_precal_train is None : #Checks to see if a precalculated train Kernel has been passed in, if not calculate
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs)
    total_errors_train = 0
    for e in range(epoch) :
        epoch_errors = 0
        #print(f"Epoch {e}, Class = {target_class}")
        for i in range(len(xtrain)) :
            if len(xtrain_online) == 0:
                score = 0
            else :
                K = K_precal_train[xtrain_online, i]
                A = np.array(alpha)
                score = A @ K
            # alpha_i = alpha[:i]  # all previous alpha's
            # K = K_precal_train[:i, i]
            # if (alpha_i @ K) >= 0:
            # K = K_precal_train[:, i]
            # K = K_precal_train[xtrain_online, i]
            if (score) >= 0:
                y_hat = 1
            else :
                y_hat = -1
            if y_hat != ytrain[i] :
                # alpha[i] = alpha[i] + ytrain[i]
                alpha.append(ytrain[i])
                xtrain_online.append(i)
                epoch_errors = epoch_errors + 1
                total_errors_train = total_errors_train + 1
        if epoch_errors < min_epoch_errors:
            min_epoch_errors = epoch_errors
            no_improvement_count = 0
            best_alpha = alpha.copy()
            best_xtrain_online = xtrain_online.copy()
        else :
            no_improvement_count = no_improvement_count + 1
        if no_improvement_count >= epoch_threshold:
            #print(f"Stopped early at epoch {e} for class {target_class}")
            break
    total_errors_train = min_epoch_errors / len(xtrain)

    alpha = best_alpha.copy()
    xtrain_online = best_xtrain_online.copy()
    if trainhelper == False: #If function is being called as a two class classifier complete the predictoin step, otherwise skip predictions
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
        y_pred_test, total_errors_test = two_class_ker_perc_pred(alpha, xtest, ytest, xtrain_online, target_class, K_precal_test)
        print("Alpha length:", len(alpha))
        print("Online history length:", len(xtrain_online))
        print("Max : Epochs * train size:", epoch * len(xtrain))
        return alpha, total_errors_train, xtrain_online, y_pred_test, total_errors_test
    return alpha, total_errors_train, xtrain_online, None, None #If function being called to do training step in another algorithm only return training outputs

def k_class_one_vs_rest_ker_perc_pred (alpha, x, y, xtrain_online, k_classes, K_precal_test, confusion=False):
    y_pred = []
    total_errors = 0
    for i in range(len(x)) :
        # y_hat = []
        scores = []
        # K = K_precal_test[i, :]
        # K = K_precal[:,i)
        # K = np.array(K)
        for j, k in enumerate (k_classes) :
            A = np.array(alpha[j])
            xtrain_online_j = xtrain_online[j]
            if len(A) == 0:
                score=0
            else :
                K = K_precal_test[i, xtrain_online_j]
                score = (A @ K)
            scores.append(score)
            # y_hat.append(alpha[j] @ K)
        y_pred.append(k_classes[np.argmax(scores)])
        # y_pred.append(k_classes[scores.index(max(scores))])
        if y_pred[i] != y[i]:
            total_errors = total_errors + 1
    if confusion == True:
        confMatrix = confusion_matrix(y, y_pred, k_classes)
    else:
        confMatrix = None
    total_errors = total_errors/len(x)
    return y_pred, total_errors, confMatrix

def k_class_one_vs_rest_ker_perc (df, kernel_func, k_classes, epoch=1, split=None, K_precal_train = None, K_precal_test = None, confusion=False, **kwargs):
    if split is None :
        xtrain, xtest, ytrain, ytest, _, _ = train_test_split(df) #create test train splits
    else :
        xtrain, xtest, ytrain, ytest = split
    alpha, y_pred_test, train_errors, test_erros, xtrain_online = [], [], [], [], []
    if K_precal_train is None:
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs) #Precalculate the Kernel Matrix
    if K_precal_test is None:
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
    for k in k_classes : #Loop through the k classes
        split = (xtrain, xtest, ytrain.copy(), ytest.copy()) #Store new dataset with y {1,-1} labels
        alpha_k, train_errors_k, xtrain_online_k, _, _ = two_class_ker_perc_full(None, kernel_func, k, epoch=epoch, K_precal_train=K_precal_train, split = split, trainhelper=True) #Train 2 class classifier, (k vs rest)
        alpha.append(alpha_k) #Add k vs rest alpha values to alpha list
        xtrain_online.append(xtrain_online_k)
        train_errors.append(train_errors_k)
    y_pred_test, test_errors, confMatrix = k_class_one_vs_rest_ker_perc_pred(alpha, xtest, ytest, xtrain_online, k_classes, K_precal_test, confusion)
    me_train = np.mean(train_errors)
    me_test = np.mean(test_errors)
    return alpha, y_pred_test, me_train, me_test, train_errors, test_errors, confMatrix

def k_class_one_vs_one_ker_perc_pred (alpha, x, y, xtrain_online, k_classes, K_precal_test, confusion=False):
    y_pred = []
    total_errors = 0
    for j in range(len(x)):
        y_hat = np.zeros(len(k_classes))
        for i, a in enumerate(k_classes):
            for k, b in enumerate (k_classes):
                if k<i :
                    K_test_pair = K_precal_test[(i,k)]["K"]
                    testPos = K_precal_test[(i,k)]["testPos"]
                    if j not in testPos:
                        continue
                    K_test_pair_jrow = np.where(testPos == j)[0][0]
                    # K = K_test_pair[K_test_pair_jrow, :]
                    # K = np.array(K)
                    A = np.array(alpha[(i,k)])
                    xtrain_online_ik = xtrain_online[(i,k)]
                    if len(A) == 0:
                        score=0
                    else :
                        K = K_test_pair[K_test_pair_jrow, xtrain_online_ik]
                        score = (A @ K)
                    # if (alpha[(i,k)] @ K) >= 0:
                    if score >= 0:
                        y_hat[i] = y_hat[i]+1
                    else:
                        y_hat[k] = y_hat[k]+1
        y_pred.append(k_classes[np.argmax(y_hat)])
        if y_pred[j] != y[j]:
            total_errors = total_errors + 1
    if confusion == True:
        confMatrix = confusion_matrix(y, y_pred, k_classes)
    else:
        confMatrix = None
    total_errors = total_errors/len(x)
    return y_pred, total_errors, confMatrix

def k_class_one_vs_one_ker_perc(df, kernel_func, k_classes, epoch=1, split=None, K_precal_train = None, K_precal_test = None, confusion=False, **kwargs):
    if split is None:
        xtrain, xtest, ytrain, ytest, _, _ = train_test_split(df)  # create test train splits
    else:
        xtrain, xtest, ytrain, ytest = split
    alpha, K_train_pairs, K_test_pairs, xtrain_online = {}, {}, {}, {}
    y_pred_test, train_errors, test_errors = [], [], []
    if K_precal_train is None:
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs) #Precalculate the Kernel Matrix
    if K_precal_test is None:
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
    for i in k_classes :
        for k  in k_classes:
            if k < i :
                trainPos = np.where((ytrain == i)|((ytrain == k)))[0]
                ytrain_pair = ytrain[trainPos]
                xtrain_pair = xtrain[trainPos]
                testPos = np.where((ytest == i) | ((ytest == k)))[0]
                ytest_pair = ytest[testPos]
                xtest_pair = xtest[testPos]
                split = (xtrain_pair, None, ytrain_pair, None)
                K_train_pair = K_precal_train[np.ix_(trainPos, trainPos)]  # Precalculate the Kernel Matrix
                # K_test_pair  = K_precal_test[:,trainPos]
                K_test_pair = K_precal_test[np.ix_(testPos, trainPos)]
                alpha_ik, train_errors_ik, xtrain_online_ik, _, _ = two_class_ker_perc_full(None, kernel_func, i, epoch=epoch,K_precal_train=K_train_pair,split=split, trainhelper=True)  # Train 2 class classifier, (k vs rest)
                alpha[(i,k)] = alpha_ik  # Add k vs rest alpha values to alpha list
                xtrain_online[(i,k)] = xtrain_online_ik
                K_train_pairs[(i, k)] = K_train_pair
                K_test_pairs[(i,k)] = {"K" : K_test_pair, "testPos": testPos}
                train_errors.append(train_errors_ik)
    y_pred_test, test_errors, confMatrix = k_class_one_vs_one_ker_perc_pred(alpha, xtest, ytest, xtrain_online, k_classes, K_test_pairs, confusion)
    me_train = np.mean(train_errors)
    me_test = np.mean(test_errors)
    return alpha, y_pred_test, me_train, me_test, train_errors, test_errors, confMatrix

def cross_validation(xtrain, ytrain, folds, algorithm_func, kernel_func, paramlist, k_classes, epoch_lim=1, **kwargs):
    positions = np.arange(len(xtrain))
    foldPositions = np.array_split(positions, folds)

    K_precal = {p: full_kern(xtrain, kernel_func=kernel_func, d=p, c=p) for p in paramlist}

    results = np.zeros((folds, len(paramlist), 1))
    for j, p in enumerate(paramlist):
        K_train_full = K_precal[p]

        for i in range(folds):

            xtrain_foldPos = np.concatenate([foldPositions[f] for f in range(folds) if f != i])
            xtest_foldPos = foldPositions[i]

            K_train = K_train_full[np.ix_(xtrain_foldPos, xtrain_foldPos)]
            K_test = K_train_full[np.ix_(xtest_foldPos, xtrain_foldPos)]
            ytrain_fold = ytrain[xtrain_foldPos]
            ytest_fold = ytrain[xtest_foldPos]
            xtrain_fold = xtrain[xtrain_foldPos]
            xtest_fold = xtrain[xtest_foldPos]
            split_fold = (xtrain_fold, xtest_fold, ytrain_fold, ytest_fold)

            _, _, _, me_test, *_ = algorithm_func(None, kernel_func, k_classes, epoch=epoch_lim, split=split_fold,
                                                  K_precal_train=K_train, K_precal_test=K_test, d=p, c=p)

            results[i, j, 0] = me_test

    avgme_test = np.mean(results[:, :, 0], axis=0)

    avgme_df = pd.DataFrame({'Param': paramlist, 'avg_me_test_fold': avgme_test})
    avgme_df = avgme_df.sort_values(by ='avg_me_test_fold', ascending=True)
    min_me = avgme_df.iloc[0]
    return float(min_me['Param']), min_me['avg_me_test_fold'], avgme_df


# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims_k_class_ker_perc(df, r, k_classes, param, algorithm_func=k_class_one_vs_rest_ker_perc, kernel_func=poly_kern, epoch_lim = 1, splits=None,  crossVal=False, confusion=False, **kwargs) :
    if crossVal == True:
        results = np.zeros((r, 1, 2))
    else:
        results=np.zeros((r, len(param),2))
    if confusion == True:
        confMatrix = np.zeros((r, len(k_classes), len(k_classes)))
    else :
        confMatrix = None
    fig, ax = None, None
    tested_counts = np.zeros(len(df))
    error_counts = np.zeros(len(df))
    y_pred_record = np.zeros(len(df))

    best_params = []

    for i in range(r):
        if splits is None:
            xtrain, xtest, ytrain, ytest, trainpos, testpos = train_test_split(df, i)
            split = (xtrain, xtest, ytrain.copy(), ytest.copy())
        else:
            xtrain, xtest, ytrain, ytest, trainpos, testpos = splits[i]
            split = (xtrain, xtest, ytrain.copy(), ytest.copy())
        if crossVal == True:
            minparam, minparam_testerror, crossValdf= cross_validation(xtrain, ytrain, 5, algorithm_func, kernel_func, param, k_classes, epoch_lim=epoch_lim)

            best_params.append({
                "run": int(i),
                "best_param": float(minparam),
                "test_error": minparam_testerror
            })
            alpha, y_pred_test, me_train, me_test, _, _, confMatrix_r = algorithm_func(None, kernel_func, k_classes, epoch=epoch_lim, split=split, confusion=confusion, d=minparam, c=minparam)
            results[i, 0, 0] = me_train
            results[i, 0, 1] = me_test
            if confusion == True:
                confMatrix[i] = confMatrix_r
        else :
            minparam = None
            for j, p in enumerate(param):
                alpha, y_pred_test, me_train, me_test, *_ = algorithm_func(None, kernel_func, k_classes, epoch = epoch_lim, split=split, d=p, c=p)
                results[i,j,0] = me_train
                results[i,j,1] = me_test
        tested_counts[testpos] = tested_counts[testpos] + 1
        errors = (np.array(y_pred_test) != ytest).astype(int)
        error_counts[testpos] = error_counts[testpos] + errors
        y_pred_record[testpos] = y_pred_test

    hard = error_counts / np.maximum(tested_counts, 1)
    hard[tested_counts == 0] = -1
    h = np.argsort(hard)[::-1][:5]
    hard = df.iloc[h, :]
    hard_pred = y_pred_record[h]

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
        return avgme_df, fullconfMatrix, hard, hard_pred, best_params
    else :
        return avgme_df, hard, hard_pred, best_params

def plot_pixel_images(hard, hard_pred):
    n = len(hard)
    fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
    for j, (i, row) in enumerate(hard.iterrows()):
        num = row.iloc[1:].values.reshape(16, 16)
        label = int(row.iloc[0])
        ax[j].imshow(num, cmap="gray")
        ax[j].set_title(f"Label {label}\nIndex {i}\nModel Pred {int(hard_pred[j])}")
        ax[j].axis("off")
    return fig, ax


url = 'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat'
data = np.loadtxt("zipcombo.dat")
df = pd.DataFrame(data)
# df = pd.read_csv(url, sep=r"\s+", header=None)
dftest = df.sample(frac=0.05, random_state=0).reset_index(drop=True)
print(df.head())

print('Q1')

print('Q2')
epoch = 10
start = time.time()
alpha, train_errors, xtrain_online, y_pred, test_errors =two_class_ker_perc_full(dftest, poly_kern, 5, 100)
print('Two class kernel perceptron classifier, one vs rest')
print('Polynomial Kernel : 1 run')
print('Target class = 5')
print(f'Train error = {train_errors}, Test error = {test_errors}')


Classes = [0,1,2,3,4,5,6,7,8,9]
alpha, y_pred_test, me_train, me_test, train_errors, test_errors, _ = k_class_one_vs_rest_ker_perc(dftest, poly_kern, Classes, 10)
print('K class kernel perceptron classifier, one vs rest')
print('Polynomial Kernel : 1 run')
print('Target classes 0-9')
print(f'Train error = {train_errors}, Test error = {test_errors}')
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")

print('Q3')
runs = 20
rsplits = []
for i in range(runs):
    rsplits.append(train_test_split(dftest, i))
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _, _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=d, epoch_lim = epoch, splits=rsplits)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Polynomial Kernel : {runs} runs')
print(avgme_df)

postprocess_q3(
    avgme_df=avgme_df,
    model_id="OvR_poly"
)



# q3_table = pd.DataFrame({
#     "d": avgme_df["param"],
#     "mean train error rates (%)": (
#         100 * avgme_df["avg_me_train"]
#     ).round(3).astype(str)
#     + " ± "
#     + (100 * avgme_df["avg_me_train_sd"]).round(3).astype(str),
#     "mean test error rates (%)": (
#         100 * avgme_df["avg_me_test"]
#     ).round(3).astype(str)
#     + " ± "
#     + (100 * avgme_df["avg_me_test_sd"]).round(3).astype(str),
# })
#
# save_outputs(
#     question_id="Q3",
#     tables={
#         "poly_summary": q3_table
#     },
#     jsons={
#         "alpha": alpha,
#         "train_test_stats": {
#             "d": q3_table["d"].tolist(),
#             "train_mean": avgme_df["avg_me_train"].tolist(),
#             "train_sd": avgme_df["avg_me_train_sd"].tolist(),
#             "test_mean": avgme_df["avg_me_test"].tolist(),
#             "test_sd": avgme_df["avg_me_test_sd"].tolist(),
#         }
#     }
# )



print('Q4')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(dftest, runs, Classes, param=d, epoch_lim=epoch, splits=rsplits, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Polynomial Kernel : {runs} runs with cross validation')
print(avgme_df)

# q4_table = pd.DataFrame({
#     "run": [(bp["run"]+1) for bp in best_params],
#     "d*": [int(bp["best_param"]) for bp in best_params],
#     "test error rate (%)": [(100*bp["test_error"]).round(4) for bp in best_params]
# })
#
# save_outputs(
#     question_id="Q4",
#     tables={
#         "cv_runs": q4_table
#     }
# )


print('Q5')
print('Average Confusion Matrix')
avgconfMatrix_raw = avgconfMatrix.copy()
avgconfMatrix = confusion_matrix_df(avgconfMatrix, Classes)
print(avgconfMatrix)

# avgconfMatrix_pct = avgconfMatrix.copy()
# conf_pct = np.empty((len(Classes), len(Classes)), dtype=object)
#
# for i in range(len(Classes)):
#     for j in range(len(Classes)):
#         mean = avgconfMatrix_raw[i, j, 0] * 100
#         sd = avgconfMatrix_raw[i, j, 1] * 100
#         conf_pct[i, j] = f"{mean:.3f} ± {sd:.3f}"
#
# conf_pct_df = pd.DataFrame(
#     conf_pct,
#     index=[str(c) for c in Classes],
#     columns=[str(c) for c in Classes]
# )
# conf_json = {
#     "classes": Classes,
#     "mean_percent": (avgconfMatrix_raw[:, :, 0] * 100).tolist(),
#     "std_percent":  (avgconfMatrix_raw[:, :, 1] * 100).tolist()
# }
# save_json(conf_json, "Q5_avg_confusion_matrix_pct.json")
# conf_q5_df = conf_pct_df.copy()
#
# # Give the index a name so it appears as the first column header
# conf_q5_df.index.name = r" "
#
# # Bold column headers
# conf_q5_df.columns = [f"\\textbf{{{c}}}" for c in conf_q5_df.columns]
# def df_to_tex_with_index(df, filename: str):
#     path = os.path.join("outputs", "tex", filename)
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     df.to_latex(
#         path,
#         index=True,          # <-- key difference
#         header=True,
#         escape=False
#     )
# df_to_tex_with_index(conf_q5_df, "Q5_avg_confusion_matrix_pct.tex")
# wrap_latex_table_small("outputs/tex/Q5_avg_confusion_matrix_pct.tex")


print('Q6')
print('hard')
fig, ax = plot_pixel_images(hard, hard_pred)

# save_outputs(
#     question_id="Q6",
#     figures={
#         "hard_examples": fig
#     }
# )

postprocess_experiment(
    question_prefix="OvR_poly",
    avgme_df=avgme_df,
    best_params=best_params,
    avgconfMatrix=avgconfMatrix_raw,
    hard=hard,
    hard_pred=hard_pred,
    classes=Classes
)

print('Q7')
print('(b)')
runs = 20
# c= np.linspace(0.045, 0.055, 50).tolist()
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch, splits=rsplits)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)

postprocess_q3(
    avgme_df=avgme_df,
    model_id="OvR_gauss",
    param_symbol="c"
)



print('(c)')
runs = 20
# c= np.linspace(0.045, 0.055, 50).tolist()
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim=epoch, splits=rsplits, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs with cross validation')
print(avgme_df)

print('Average Confusion Matrix')
avgconfMatrix_raw = avgconfMatrix.copy()
avgconfMatrix = confusion_matrix_df(avgconfMatrix, Classes)
print(avgconfMatrix)

fig, ax = plot_pixel_images(hard, hard_pred)

postprocess_experiment(
    question_prefix="OvR_guass",
    avgme_df=avgme_df,
    best_params=best_params,
    avgconfMatrix=avgconfMatrix_raw,
    hard=hard,
    hard_pred=hard_pred,
    classes=Classes,
    param_symbol="c"
)

print('Q8')
print('(a)')

print('(b)')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _, _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=d, algorithm_func=k_class_one_vs_one_ker_perc, epoch_lim = epoch, splits=rsplits)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs one')
print(f'Polynomial Kernel : {runs} runs')
print(avgme_df)

postprocess_q3(
    avgme_df=avgme_df,
    model_id="OvO_poly"
)



print('(c)')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(dftest, runs, Classes, param=d, algorithm_func=k_class_one_vs_one_ker_perc, epoch_lim = epoch, splits=rsplits, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs one')
print(f'Polynomial Kernel : {runs} runs with cross validation')
print(avgme_df)

print('Average Confusion Matrix')
avgconfMatrix_raw = avgconfMatrix.copy()
avgconfMatrix = confusion_matrix_df(avgconfMatrix, Classes)
print(avgconfMatrix)

fig, ax = plot_pixel_images(hard, hard_pred)

postprocess_experiment(
    question_prefix="OvO_poly",
    avgme_df=avgme_df,
    best_params=best_params,
    avgconfMatrix=avgconfMatrix_raw,
    hard=hard,
    hard_pred=hard_pred,
    classes=Classes
)

print('Extension : one vs one Gaussian kenrel')
runs = 20
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _, _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, algorithm_func=k_class_one_vs_one_ker_perc, kernel_func=guass_kern, epoch_lim=epoch, splits=rsplits)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs one')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)

postprocess_q3(
    avgme_df=avgme_df,
    model_id="OvO_guass",
    param_symbol="c"
)


runs = 20
c = [2**(-b/2) for b in range(6, 13)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, algorithm_func=k_class_one_vs_one_ker_perc, kernel_func=guass_kern, epoch_lim = epoch, splits=rsplits, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs one')
print(f'Guassian Kernel : {runs} runs with cross validation')
print(avgme_df)

print('Average Confusion Matrix')
avgconfMatrix_raw = avgconfMatrix.copy()
avgconfMatrix = confusion_matrix_df(avgconfMatrix, Classes)
print(avgconfMatrix)

fig, ax = plot_pixel_images(hard, hard_pred)

postprocess_experiment(
    question_prefix="OvO_guass",
    avgme_df=avgme_df,
    best_params=best_params,
    avgconfMatrix=avgconfMatrix_raw,
    hard=hard,
    hard_pred=hard_pred,
    classes=Classes,
    param_symbol="c"
)

print('(d)')


plt.show()
plt.close()

