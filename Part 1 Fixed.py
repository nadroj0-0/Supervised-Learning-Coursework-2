from random import gauss

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import requests
from io import StringIO
import copy

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

def plot_gaussian_param_curve(
    avgme_df: pd.DataFrame,
    *,
    title: str,
    show_as_powers: bool = False,
    b_values: list[int] | None = None
):
    """
    Plot test error vs Gaussian kernel parameter c.

    Parameters
    ----------
    avgme_df : pd.DataFrame
        Must contain columns: 'param', 'avg_me_test'
    title : str
        Plot title (LaTeX allowed)
    show_as_powers : bool
        If True, x-axis labels shown as 2^{-b/2}
    b_values : list[int] | None
        Required if show_as_powers=True.
        Must align with avgme_df rows.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    x = avgme_df["param"].values
    y = avgme_df["avg_me_test"].values

    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_ylabel("Test error rate")
    ax.set_xlabel("Gaussian kernel parameter $c$")
    ax.set_title(title)

    if show_as_powers:
        if b_values is None:
            raise ValueError("b_values must be provided when show_as_powers=True")

        labels = [rf"$2^{{-{b}/2}}$" for b in b_values]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax



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

def predict(alpha, xtrain_online, KRow) :
    if len(alpha) == 0:
        return 0
    else:
        K = KRow[xtrain_online]
        return np.array(alpha) @ K

def update_model (score, y, alpha, xtrain_online, t):
    if score >= 0:
        y_hat = 1
    else:
        y_hat = -1
    if y_hat != y:
        alpha.append(y)
        xtrain_online.append(t)

def k_class_one_vs_rest_online_ker_perc (df, kernel_func, k_classes, epoch=1, split=None, K_precal_train = None, K_precal_test = None, confusion=False, **kwargs):
    if split is None :
        xtrain, xtest, ytrain, ytest, _, _ = train_test_split(df) #create test train splits
    else :
        xtrain, xtest, ytrain, ytest = split
    y_pred_train, y_pred_test = [], []
    alpha = {k: [] for k in k_classes}
    xtrain_online = {k: [] for k in k_classes}
    if K_precal_train is None:
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs) #Precalculate the Kernel Matrix
    if K_precal_test is None:
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
    total_errors = 0
    for e in range(epoch) :
        epoch_errors = 0
        for i in range(len(xtrain)):
            t = (e*len(xtrain)) + i
            scores = []
            for k in k_classes :
                score = predict(alpha[k], xtrain_online[k], K_precal_train[i, :]) #predict A @ K
                scores.append(score)
            y_hat = k_classes[np.argmax(scores)]
            y_pred_train.append(y_hat)
            if y_hat != ytrain[i]:
                total_errors = total_errors + 1
            for j, k in enumerate(k_classes) :
                if ytrain[i] == k:
                    ylabel_class = 1 #Label ytrain[i] should have in model k
                else:
                    ylabel_class = -1
                update_model(scores[j], ylabel_class, alpha[k], xtrain_online[k], i)
    train_error = total_errors / (t+1)
    total_errors = 0

    for i in range(len(xtest)):
        scores = []
        for k in k_classes:
            score = predict(alpha[k], xtrain_online[k], K_precal_test[i, :])
            scores.append(score)
        y_hat = k_classes[np.argmax(scores)]
        y_pred_test.append(y_hat)
        if y_hat != ytest[i]:
            total_errors = total_errors + 1
    test_error = total_errors / len(xtest)
    if confusion == True:
        confMatrix = confusion_matrix(ytest, y_pred_test, k_classes)
    else:
        confMatrix = None
    return alpha, y_pred_test, train_error, test_error, confMatrix

def k_class_one_vs_one_online_ker_perc (df, kernel_func, k_classes, epoch=1, split=None, K_precal_train = None, K_precal_test = None, confusion=False, **kwargs):
    if split is None :
        xtrain, xtest, ytrain, ytest, _, _ = train_test_split(df) #create test train splits
    else :
        xtrain, xtest, ytrain, ytest = split
    y_pred_train, y_pred_test = [], []
    pairs = [(i, k) for i in k_classes for k in k_classes if k < i]
    alpha = {(i, k): [] for (i, k) in pairs}
    xtrain_online = {(i, k): [] for (i, k) in pairs}

    if K_precal_train is None:
        K_precal_train = full_kern(xtrain, kernel_func=kernel_func, **kwargs) #Precalculate the Kernel Matrix
    if K_precal_test is None:
        K_precal_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, **kwargs)
    total_errors = 0

    for e in range(epoch) :
        epoch_errors = 0
        for i in range(len(xtrain)):
            t = (e*len(xtrain)) + i
            votes = np.zeros(len(k_classes))
            times_voted = np.zeros(len(k_classes))
            pair_scores = {}
            for (a,b) in pairs :
                pair_scores[(a,b)] = predict(alpha[(a,b)], xtrain_online[(a,b)], K_precal_train[i, :]) #Predict A@K
                if pair_scores[(a,b)] >= 0 :
                    votes[a] = votes[a] + 1
                else :
                    votes[b] = votes[b] + 1
                times_voted[a] = times_voted[a] + 1
                times_voted[b] = times_voted[b] + 1
                remaining_votes = (len(k_classes)-1) - times_voted
                current_most_votes = np.argmax(votes)
                max_possible = votes+ remaining_votes
                max_possible[current_most_votes] = -np.inf
                if votes[current_most_votes] > np.max(max_possible):
                    break


            y_hat = k_classes[np.argmax(votes)]
            y_pred_train.append(y_hat)
            if y_hat != ytrain[i]:
                total_errors = total_errors + 1
            for (a,b) in pairs:
                if ytrain[i] in (a,b):
                    if ytrain[i] == a:
                        ylabel_class =1
                    else :
                        ylabel_class =-1
                    if (a, b) not in pair_scores:
                        pair_scores[(a, b)] = predict(alpha[(a, b)], xtrain_online[(a, b)], K_precal_train[i, :])
                    update_model(pair_scores[(a,b)], ylabel_class, alpha[(a,b)], xtrain_online[(a,b)], i)

    train_error = total_errors / (t+1)
    total_errors = 0

    for i in range(len(xtest)):
        votes = np.zeros(len(k_classes))
        for (a,b) in pairs:
            score = predict(alpha[(a, b)], xtrain_online[(a, b)], K_precal_test[i, :])
            if score >= 0:
                votes[a] = votes[a] + 1
            else:
                votes[b] = votes[b] + 1
        y_hat = k_classes[np.argmax(votes)]
        y_pred_test.append(y_hat)
        if y_hat != ytest[i]:
            total_errors = total_errors + 1
    test_error = total_errors / len(xtest)
    if confusion == True:
        confMatrix = confusion_matrix(ytest, y_pred_test, k_classes)
    else:
        confMatrix = None
    return alpha, y_pred_test, train_error, test_error, confMatrix

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

            _, _, _, me_test, _ = algorithm_func(None, kernel_func, k_classes, epoch=epoch_lim, split=split_fold,
                                                  K_precal_train=K_train, K_precal_test=K_test, d=p, c=p)

            results[i, j, 0] = me_test

    avgme_test = np.mean(results[:, :, 0], axis=0)

    avgme_df = pd.DataFrame({'Param': paramlist, 'avg_me_test_fold': avgme_test})
    avgme_df = avgme_df.sort_values(by ='avg_me_test_fold', ascending=True)
    min_me = avgme_df.iloc[0]
    return float(min_me['Param']), min_me['avg_me_test_fold'], avgme_df


# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims_k_class_ker_perc(df, r, k_classes, param, algorithm_func=k_class_one_vs_rest_online_ker_perc, kernel_func=poly_kern, epoch_lim = 1, splits=None,  crossVal=False, confusion=False, **kwargs) :
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
            alpha, y_pred_test, me_train, me_test, confMatrix_r = algorithm_func(None, kernel_func, k_classes, epoch=epoch_lim, split=split, confusion=confusion, d=minparam, c=minparam)
            results[i, 0, 0] = me_train
            results[i, 0, 1] = me_test
            if confusion == True:
                confMatrix[i] = confMatrix_r
        else :
            minparam = None
            for j, p in enumerate(param):
                alpha, y_pred_test, me_train, me_test, _ = algorithm_func(None, kernel_func, k_classes, epoch = epoch_lim, split=split, d=p, c=p)
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




startTotal = time.time()
url = 'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat'
# data = np.loadtxt("zipcombo.dat")
# df = pd.DataFrame(data)
df = pd.read_csv(url, sep=r"\s+", header=None)
dftest = df.sample(frac=0.15, random_state=0).reset_index(drop=True)
# df2=df
# df=dftest
print(df.head())

print('Q1')


print('Q2')
epoch = 12
start = time.time()
# # alpha, train_errors, xtrain_online, y_pred, test_errors =two_class_ker_perc_full(df, poly_kern, 5, 100)
# print('Two class kernel perceptron classifier, one vs rest')
# print('Polynomial Kernel : 1 run')
# print('Target class = 5')
# print(f'Train error = {train_errors}, Test error = {test_errors}')


Classes = [0,1,2,3,4,5,6,7,8,9]
alpha, y_pred_test, train_errors, test_errors, _ = k_class_one_vs_rest_online_ker_perc(df, poly_kern, Classes, 10)
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
    rsplits.append(train_test_split(df, i))
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _, _, _ = r_sims_k_class_ker_perc(df, runs, Classes, param=d, epoch_lim = epoch, splits=rsplits)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Polynomial Kernel : {runs} runs')
print(avgme_df)

postprocess_q3(
    avgme_df=avgme_df,
    model_id="OvR_poly"
)


print('Q4')
runs = 20
d=[1,2,3,4,5,6,7]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(df, runs, Classes, param=d, epoch_lim=epoch, splits=rsplits, crossVal=True, confusion=True)
end = time.time()
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Polynomial Kernel : {runs} runs with cross validation')
print(avgme_df)
#
#
print('Q5')
print('Average Confusion Matrix')
avgconfMatrix_raw = avgconfMatrix.copy()
avgconfMatrix = confusion_matrix_df(avgconfMatrix, Classes)
print(avgconfMatrix)
#
#
print('Q6')
print('hard')
fig, ax = plot_pixel_images(hard, hard_pred)


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
print('(a) Explaining how S was chosen')
runs=1
c = [0.005, 0.01, 0.05, 0.1, 0.5 ,1, 5, 10]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch)
end = time.time()
print(f'c = {c}')
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)
fig, ax = plot_gaussian_param_curve(
    avgme_df,
    title="OvR Gaussian kernel: test error vs parameter $c$ (15% data subset, 1 run)",
    show_as_powers=False
)

save_outputs(
    question_id="Q7a_ovr_gauss_wide_15pct",
    figures={"param_sweep": fig}
)





runs=1
c = [2**(-b/2) for b in range(10, 20)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch)
end = time.time()
print(f'c = {c}')
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)
b_vals = list(range(10, 20))

fig, ax = plot_gaussian_param_curve(
    avgme_df,
    title="OvR Gaussian kernel: test error vs parameter $c$ (15% data subset, 1 run)",
    show_as_powers=True,
    b_values=b_vals
)

save_outputs(
    question_id="Q7a_ovr_gauss_logfine_15pct",
    figures={"param_sweep": fig}
)



runs=1
c = [2**(-b/2) for b in range(11, 18)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(dftest, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch)
end = time.time()
print(f'c = {c}')
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)
b_vals = list(range(11, 18))

fig, ax = plot_gaussian_param_curve(
    avgme_df,
    title="OvR Gaussian kernel: test error vs parameter $c$ (15% data subset, 1 run)",
    show_as_powers=True,
    b_values=b_vals
)

save_outputs(
    question_id="Q7a_ovr_gauss_logfiner_15pct",
    figures={"param_sweep": fig}
)


runs=1
c = [2**(-b/2) for b in range(9, 16)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(df, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch)
end = time.time()
print(f'c = {c}')
print(f"\nTotal runtime = {end - start:.2f} seconds ({(end-start)/60:.2f} minutes)")
print('K class kernel perceptron classifier, one vs rest')
print(f'Guassian Kernel : {runs} runs')
print(avgme_df)
b_vals = list(range(9, 16))

fig, ax = plot_gaussian_param_curve(
    avgme_df,
    title="OvR Gaussian kernel: test error vs parameter $c$ (full dataset, 1 run)",
    show_as_powers=True,
    b_values=b_vals
)

save_outputs(
    question_id="Q7a_ovr_gauss_logfiner_full",
    figures={"param_sweep": fig}
)






print('(b)')
runs = 20
# c= np.linspace(0.045, 0.055, 50).tolist()
c = [2**(-b/2) for b in range(9, 16)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _ , _, _ = r_sims_k_class_ker_perc(df, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim= epoch, splits=rsplits)
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
c = [2**(-b/2) for b in range(9, 16)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(df, runs, Classes, param=c, kernel_func=guass_kern, epoch_lim=epoch, splits=rsplits, crossVal=True, confusion=True)
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
avgme_df, _, _, _ = r_sims_k_class_ker_perc(df, runs, Classes, param=d, algorithm_func=k_class_one_vs_one_online_ker_perc, epoch_lim = epoch, splits=rsplits)
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
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(df, runs, Classes, param=d, algorithm_func=k_class_one_vs_one_online_ker_perc, epoch_lim = epoch, splits=rsplits, crossVal=True, confusion=True)
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
c = [2**(-b/2) for b in range(9, 16)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, _, _, _ = r_sims_k_class_ker_perc(df, runs, Classes, param=c, algorithm_func=k_class_one_vs_one_online_ker_perc, kernel_func=guass_kern, epoch_lim=epoch, splits=rsplits)
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
c = [2**(-b/2) for b in range(9, 16)]
Classes = [0,1,2,3,4,5,6,7,8,9]
start = time.time()
avgme_df, avgconfMatrix, hard, hard_pred, best_params = r_sims_k_class_ker_perc(df, runs, Classes, param=c, algorithm_func=k_class_one_vs_one_online_ker_perc, kernel_func=guass_kern, epoch_lim = epoch, splits=rsplits, crossVal=True, confusion=True)
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

endTotal = time.time()
print(f"\nTotal script runtime = {endTotal - startTotal:.2f} seconds ({(endTotal-startTotal)/60:.2f} minutes)")
plt.show()
plt.close()

