# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% KNN on Iris Dataset

# %% part 0 - import necessary modules
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris

# %% part 1 - load dataset
dataset = load_iris(return_X_y=False)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Here are some info. about the Iris dataset:')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# print('DESCR:\n', dataset.DESCR)
# print('%%%%%%%%%%%%%%%%%%%%%')
print('feature_names:\n', dataset.feature_names)
print('%%%%%%%%%%%%%%%%%%%%%')
print('target_names:\n', dataset.target_names)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

X, y = load_iris(return_X_y=True)

# %% part 2 - train and test model
num_folds = 10
kfold_object = KFold(n_splits=num_folds,
                     shuffle=True,
                     random_state=110)
best_k = -1; best_score = -1
min_k = 1; step_k = 2
best_k_list = [None] * num_folds
best_acc_list = [None] * num_folds
best_f1_list = [None] * num_folds
best_cm_list = [None] * num_folds
total_tr_time = 0
total_te_time = 0

fold_counter = 0
for train_index, test_index in kfold_object.split(X):
    start_time_train_tmp = time.time()
    fold_counter += 1
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    #########################################################
    X_tr_tr, X_tr_val, y_tr_tr, y_tr_val = train_test_split(X_train_fold,
                                                            y_train_fold,
                                                            test_size=0.15,
                                                            random_state=19)
    scaler = StandardScaler()
    X_tr_tr = scaler.fit_transform(X_tr_tr)
    X_tr_val = scaler.transform(X_tr_val)
    #########################################################
    extreme_max_k = len(y_tr_tr) - 1
    max_k = round(len(y_tr_tr) / 4)
    range_of_k = list(range(min_k, max_k, step_k))

    best_in_fold_score = 0
    for k in range_of_k:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_tr_tr, y_tr_tr)
        in_fold_score = clf.score(X_tr_val, y_tr_val)
        if in_fold_score > best_in_fold_score:
            best_in_fold_score = in_fold_score
            best_k_list[fold_counter-1] = k
    #########################################################

    clf = KNeighborsClassifier(n_neighbors=best_k_list[fold_counter - 1])

    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)

    clf.fit(X_train_fold, y_train_fold)

    finish_time_train_tmp = time.time()
    total_tr_time += finish_time_train_tmp - start_time_train_tmp

    start_time_test_tmp = time.time()

    on_fold_score = clf.score(X_test_fold, y_test_fold)
    best_acc_list[fold_counter-1] = on_fold_score
    y_test_fold_pred = clf.predict(X_test_fold)
    best_f1_list[fold_counter-1] = f1_score(y_test_fold,
                                            y_test_fold_pred,
                                            average='weighted')
    best_cm_list[fold_counter-1] = confusion_matrix(y_test_fold,
                                                    y_test_fold_pred,
                                                    labels=[0, 1, 2])
    #########################################################

    finish_time_test_tmp = time.time()
    total_te_time += finish_time_test_tmp - start_time_test_tmp

print('--------------------------------------------')
print('Computation Time:')
print('--------------------------------------------')
print('Comp. Time --> Total Training:',
      round(total_tr_time, 6),
      'seconds')
print('Comp. Time --> Total Testing:',
      round(total_te_time, 6),
      'seconds')
print('%%%%%%%%%%%%%%%%%%%%%')

# %% part * - report result of CV
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Cross Val. Report:')
print('%%%%%%%%%%%%%%%%%%%%%')
print('for KNN, hyper-parameter is "k".', f'Best found "k" was: {best_k_list}.')
print(f'Number of Folds: {num_folds}; Range of searched K: ({min_k}, {max_k}, {step_k}).')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

# %% part 3 - produce CM, F1-score, and Accuracy
r_len = 6
tr_conf_mat = sum(best_cm_list)/len(best_cm_list)
print('------------------------------------------------')
print('KPI Results, on k-fold Test sets:')
print('f1_score {average="weighted"} , accuracy_score -->',
      round(sum(best_f1_list)/len(best_f1_list), r_len), ',',
      round(sum(best_acc_list)/len(best_acc_list), r_len))
print('-----------------')
print('Confusion Matrix:')
print('-----------------')
print(tr_conf_mat / sum(tr_conf_mat))


# %%
