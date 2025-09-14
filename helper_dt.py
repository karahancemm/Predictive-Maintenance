from prettytable import PrettyTable
from datetime import date, datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def print_test_confusion_matrix(test_cm):

    pt = PrettyTable()
    pt.field_names = ['Test', 'Predicted : Success', 'Predicted: Failure']

    pt.add_row(['Actual: Success', test_cm[0, 0], test_cm[0, 1]])
    pt.add_row(['Actual: Failure', test_cm[1, 0], test_cm[1, 1]])

    return print(pt)

def print_val_confusion_matrix(val_cm):

    pt = PrettyTable()
    pt.field_names = ['Validation', 'Predicted : Success', 'Predicted: Failure']

    pt.add_row(['Actual: Success', val_cm[0, 0], val_cm[0, 1]])
    pt.add_row(['Actual: Failure', val_cm[1, 0], val_cm[1, 1]])

    return print(pt)

def time_now():
    now = datetime.now()
    mili = now.microsecond // 1000
    mili_two = mili // 10
    return print(f"{now.strftime('%H:%M:%S')}.{mili_two}")

def print_multiple_class_confusion_matrix(test_cm):
    
    pt = PrettyTable()
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF'] #'RNF',
    pt.field_names = ["Confusion matrix"] + [f"Pred: {lbl}" for lbl in labels]
    for i, actual_label in enumerate(labels):
        row_counts = test_cm[i]  # a list of counts for each predicted label
        pt.add_row([f"Actual: {actual_label}"] + list(row_counts))

    return print(pt)
"""def gridsearch(dt, x, y):
    for i in range(1, 10, 0.5):
        param_grid = {'class_weight' =  [{'Safe' : , 'HDF' : , 'OSF' : , 'PWF' : , 'RND' : , 'RNF' : , 'TWF' : }]}

    clf = DecisionTreeClassifier(criterion = 'entropy', class_weight = weight_dict_manual, random_state = 42)
    grid_search = GridSearchCV(clf, param_grid, cv = 100, scoring = 'f1_macro', n_jobs = 10)
    grid_search.fit(x_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)
    h.time_now()"""

