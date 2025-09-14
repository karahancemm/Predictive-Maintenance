import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import TomekLinks
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from xgboost import XGBClassifier
from datetime import date, datetime
import helper_dt as h

def time_now():
    now = datetime.now()
    mili = now.microsecond // 1000
    mili_two = mili // 10
    return print(f"{now.strftime('%H:%M:%S')}.{mili_two}")

def read_data():
    folder = '/Users/cemkarahan/Desktop/'
    df = pd.read_feather(folder + 'thesis_data.fth') # Reading and loading the data
    return df

def data_prep_feature(df):
     #### Data Prep & Feature Engineering ###
    # One Hot Encoding - Type #
    df = df[~((df['Tool wear [min]'] < 200) & (df['TWF'] ==1))]
    ohe = OneHotEncoder(sparse_output = False)
    encoded_features = ohe.fit_transform(df[['Type']])
    df_encoded = pd.DataFrame(encoded_features, columns = ohe.get_feature_names_out(['Type']))
    df_encoded['Product ID'] = df['Product ID'].values
    df = df.merge(df_encoded, how = 'left', on = 'Product ID')
    ## Finalizing the features and responses ###
    df.drop(columns = {'UDI', 'Product ID'}, inplace = True) # # Dropping unnecessary fields
    df['Dumy'] = df['TWF'] + df['HDF'] + df['PWF'] + df['OSF']
    df['Safe'] = np.where(df['Dumy'] > 0, 0, 0.5)
    df['Failure Type'] = df[['Safe', 'HDF', 'OSF', 'PWF', 'TWF']].idxmax(axis = 1) # Gathering failure types into one column
    df['Failure Type - All'] = df[['Safe', 'HDF', 'OSF', 'PWF', 'RNF', 'TWF']].idxmax(axis = 1) # Gathering failure types into one column WITH RANDOM FAILURE
    df['Temperature Delta'] = abs(df['Process temperature [K]'] - df['Air temperature [K]'])
    df = df[['Type_H', 'Type_M', 'Type_L', 'Temperature Delta', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Failure Type', 'Safe', 'HDF', 'OSF', 'PWF', 'TWF']] #'RNF', 
    return df

def parameter_creation(df):
    tool_multipliers = {'L': 3, 'M': 2, 'H': 1}
    types = ['L', 'M', 'H']
    for i in types: # Only for Overstrain Parameter
        df[f'Overstrain Parameter_{i}'] = df['Torque [Nm]'] * df['Tool wear [min]'] 
    for i in types: # Only for Power Parameter
        df[f'Power Parameter_{i}'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] 
    tool_wear_failure_factor = df['Tool wear [min]'].between(200, 240).astype(int)
    for i in types: # Only for Tool Parameter
            df[f'Tool Parameter_{i}'] = df['Tool wear [min]'] * tool_multipliers[i] * tool_wear_failure_factor * df[f'Type_{i}']
    return df

def weights(twf, osf, pwf, hdf, safe):
    return twf, osf, pwf, hdf, safe

def class_weight_selection(start, end, label_name):
    others = [1.0, 3.0, 5.0, 9.0] #, 5.0, 10.0]
    safe = [0.7, 1.0, 2.0, 4.0]
    c_w = []
    for x in np.arange(start, end, 4):
        for o1 in others:
            for s in safe:
                c_w.append({label_name: x, 'HDF': o1, 'OSF': o1, 'PWF': o1, 'Safe': s})
    return c_w
        
def split_data_cv(x, y, train_idx, test_idx):
    x_train, y_train = x.loc[train_idx], y.iloc[train_idx]
    x_test, y_test= x.iloc[test_idx], y.iloc[test_idx]
    return x_train, y_train, x_test, y_test

def find_f1_weight(cw_f1_list):
    f1 = np.nan
    condition = 0
    settings = {}
    for i in cw_f1_list:
        if i[1] >= condition:
            f1 = i[1]
            condition = f1
            settings = i[0]
    return settings, condition
        
    #return max(cw_f1_list, key = lambda x: x[1])

    """cw_f1_list = np.array(cw_f1_list)

    f1 = np.nan
    condition = 0
    for i in cw_f1_list:
        if i[1] >= condition:
            f1 = i[1]
            condition = f1
    return f1."""

def tune_threshold(p_pos, y_true):
    label = 'TWF'
    thresholds = np.linspace(0.00, 1.00, 1001)
    best_threshold, best_cost = 0.0, np.inf
    for t in thresholds:
        preds = (p_pos >= t).astype(int)
        fn = np.sum((y_true == label) & (preds == 0))
        fp = np.sum((y_true != label) & (preds == 1))
        cost = fn * 5 + fp
        #print(f"t={t:.2f}  FN={fn:3d}  FP={fp:3d}  cost={cost:5d}")
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold

def evaluate_rf_folds(x, y, rf_params = None, cal_params = None, svm_params = None, do_sample_weights = False, sample_weights = None, do_pca_anomaly = False, pca_parameters_twf = None, pca_parameters_hdf = None, do_oversampling = False, twf_oversample = None, do_undersampling = False):
    rf_params = rf_params or {}
    cal_params = cal_params or {}
    sample_weights = sample_weights or {}
    pca_parameters_twf = pca_parameters_twf or {}
    pca_parameters_hdf = pca_parameters_hdf or {}
    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    f1_per_fold = []
    y_true_list = []
    y_pred_list = []
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF']
    false_positives = {lbl: [] for lbl in labels}
    false_negatives = {lbl: [] for lbl in labels}

    for fold, (train_idx,test_idx) in enumerate(outer_cv.split(x, y), 1):
        ### splitting data ###
        x_train, y_train, x_test, y_test = split_data_cv(x, y, train_idx, test_idx)
        if do_oversampling:
            x_train, y_train = svm_smote(x_train, y_train, twf_oversample)
        if do_undersampling:
            undersampling = TomekLinks(sampling_strategy = ['Safe'])
            mask_remove = (y_train == 'TWF') | (y_train == 'Safe')
            x_twf, y_twf = x_train[mask_remove], y_train[mask_remove]
            x_train = x_train[~mask_remove]
            y_train = y_train[~mask_remove]
            x_twf, y_twf = undersampling.fit_resample(x_twf, y_twf)
            x_train = pd.concat([x_train, x_twf])
            y_train = pd.concat([pd.Series(y_train), pd.Series(y_twf)])
        if do_sample_weights:
            sw = sample_weights_gen_combined(x_train, y_train, **sample_weights)
        else:
            sample_weights = None
        
        if do_pca_anomaly:
            anomalies_test_twf = anomaly_detect_pca(x_train, x_test, **pca_parameters_twf)
            anomalies_test_hdf = anomaly_detect_pca(x_train, x_test, **pca_parameters_hdf)
        base_rf = RandomForestClassifier(**rf_params)
        calibrator = CalibratedClassifierCV(base_rf, **cal_params)
        calibrator.fit(x_train, y_train, sample_weight = sw) if do_sample_weights else calibrator.fit(x_train, y_train)

        y_pred = calibrator.predict(x_test)
        if do_pca_anomaly:
            y_pred[(anomalies_test_twf == False) & (y_pred == 'TWF')] = 'Safe'
            y_pred[(anomalies_test_hdf == False) & (y_pred == 'HDF')] = 'Safe'
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        f1 = f1_score(y_test, y_pred, average = 'macro', zero_division = 0)
        f1_per_fold.append(f1)

        ##Tracking the misclassifications
        for l in ['HDF', 'OSF', 'PWF', 'TWF']:
            fp = (y_pred == l) & (y_test == 'Safe')
            fn = (y_pred == 'Safe') & (y_test== l)
            false_positives[l].extend(test_idx[fp])
            false_negatives[l].extend(test_idx[fn])

    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    print("Classification Report Test:\n", classification_report(y_true_all, y_pred_all, zero_division = 0))
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF'] #'RNF', 
    test_cm = confusion_matrix(y_true_all, y_pred_all, labels = labels) # Confusion matrix preperation
    h.print_multiple_class_confusion_matrix(test_cm) # Confusion matrix, layout

    mean_f1 = np.mean(f1_per_fold)
    std_f1 = np.std(f1_per_fold, ddof = 1)
    print(f'Macro-F1= {mean_f1:.3f} ± {std_f1:.3f} (5-fold CV)')
    print(base_rf.get_params())
    if do_sample_weights:
        print(sample_weights)
    
    # Prepare full DataFrames for export
    records_fp = []
    records_fn = []

    ### EXPORT TO EXCEL ###
    for label in labels:
        for idx in false_positives[label]:
            row = x.iloc[idx].to_dict()
            row.update({'true_label': y.iloc[idx], 'pred_label': label})
            records_fp.append(row)
        for idx in false_negatives[label]:
            row = x.iloc[idx].to_dict()
            row.update({'true_label': label, 'pred_label': y_pred_all[np.where(np.concatenate(y_true_list) == y_true_all)[0][list(false_negatives[label]).index(idx)]]})
            records_fn.append(row)

    df_fp_all = pd.DataFrame.from_records(records_fp)
    df_fn_all = pd.DataFrame.from_records(records_fn)

    # Export to Excel: two sheets
    folder = '/Users/cemkarahan/Desktop/tez/Python Exports/Faults - RF.xlsx'
    """with pd.ExcelWriter(folder) as writer:
        df_fp_all.to_excel(writer, sheet_name='False_Positives', index=False)
        df_fn_all.to_excel(writer, sheet_name='False_Negatives', index=False)"""

    print(f"Exported full false positives ({len(df_fp_all)}) and false negatives ({len(df_fn_all)}) to '{folder}'.")

def evaluate_rf_train_test(x, y, rf_params = None, cal_params = None, svm_params = None, do_sample_weights = False):
    rf_params = rf_params or {}
    cal_params = cal_params or {}
    

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 42)
    rf = RandomForestClassifier(**rf_params)
    cal = CalibratedClassifierCV(rf, **cal_params)
    cal.fit(x_train, y_train)

    y_pred = cal.predict(x_test)
    print("Classification Report Test:\n", classification_report(y_test, y_pred, zero_division = 0))
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF'] #'RNF', 
    test_cm = confusion_matrix(y_test, y_pred, labels = labels) # Confusion matrix preperation
    h.print_multiple_class_confusion_matrix(test_cm) # Confusion matrix, layout

def anomaly_detect_pca(x_train, x_test, n_pca, t):
        binary = ['Type_H', 'Type_M', 'Type_L']
        num = [col for col in x_train.columns if col not in binary]
        preprocessor = ColumnTransformer(transformers = [('num', StandardScaler(), num), ('bin', 'passthrough', binary)])
        scaler = preprocessor.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        pca = PCA(n_components = n_pca, svd_solver = 'full')
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_recon = pca.inverse_transform(x_train_pca)
        reconstruction_error = ((x_train_scaled - x_recon)**2).mean(axis = 1)
        threshold = np.percentile(reconstruction_error, t)
        anomalies = reconstruction_error > threshold

        x_test_pca = pca.transform(x_test_scaled)
        x_test_recon = pca.inverse_transform(x_test_pca)
        reconstruction_error_test = ((x_test_scaled - x_test_recon)**2).mean(axis=1)
        anomalies_test = reconstruction_error_test > threshold
        return anomalies_test

def anomaly_detect_pca_xgb(x_train, x_test, n_c, th):
        #Preparing data
        binary = ['Type_H', 'Type_M', 'Type_L']
        num = [col for col in x_train.columns if col not in binary]
        preprocessor = ColumnTransformer(transformers = [('num', StandardScaler(), num), ('bin', 'passthrough', binary)])
        scaler = preprocessor.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        #Actual Steps of PCA
        pca = PCA(n_components = n_c, svd_solver = 'full')
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_recon = pca.inverse_transform(x_train_pca)
        reconstruction_error = ((x_train_scaled - x_recon)**2).mean(axis = 1)
        threshold = np.percentile(reconstruction_error, th)
        anomalies = reconstruction_error > threshold

        x_test_pca = pca.transform(x_test_scaled)
        x_test_recon = pca.inverse_transform(x_test_pca)
        reconstruction_error_test = ((x_test_scaled - x_test_recon)**2).mean(axis=1)
        anomalies_test = reconstruction_error_test > threshold
        return anomalies_test


def evaluate_dt_folds(x, y, dt_params = None, smote_params = None, do_oversampling = False):
    f1_per_fold = []
    y_true_list = []
    y_pred_list = []
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF']
    false_positives = {lbl: [] for lbl in labels}
    false_negatives = {lbl: [] for lbl in labels}
    dt_params = dt_params or {}
    smote_params = smote_params or {}
    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    for fold, (train_idx,test_idx) in enumerate(outer_cv.split(x, y), 1):
        x_train, y_train, x_test, y_test = split_data_cv(x, y, train_idx, test_idx)
        dt = DecisionTreeClassifier(**dt_params)

        if do_oversampling:
            oversampling = SVMSMOTE(**smote_params)
            x_train, y_train = oversampling.fit_resample(x_train, y_train)
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_test)

        ##Tracking the misclassifications
        for l in ['HDF', 'OSF', 'PWF', 'TWF']:
            fp = (y_pred == l) & (y_test == 'Safe')
            fn = (y_pred == 'Safe') & (y_test== l)
            false_positives[l].extend(test_idx[fp])
            false_negatives[l].extend(test_idx[fn])

        f1 = f1_score(y_test, y_pred, average = 'macro', zero_division = 0)
        f1_per_fold.append(f1)
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
    
    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    print("Classification Report Test:\n", classification_report(y_true_all, y_pred_all, zero_division = 0))
    
    test_cm = confusion_matrix(y_true_all, y_pred_all, labels = labels) # Confusion matrix preperation
    h.print_multiple_class_confusion_matrix(test_cm) # Confusion matrix, layout
    mean_f1 = np.mean(f1_per_fold)
    std_f1 = np.std(f1_per_fold, ddof = 1)
    print(f'Macro-F1= {mean_f1:.3f} ± {std_f1:.3f} (5-fold CV)')
    print(dt.get_params())

        # Prepare full DataFrames for export
    records_fp = []
    records_fn = []

    for label in labels:
        for idx in false_positives[label]:
            row = x.iloc[idx].to_dict()
            row.update({'true_label': y.iloc[idx], 'pred_label': label})
            records_fp.append(row)
        for idx in false_negatives[label]:
            row = x.iloc[idx].to_dict()
            row.update({'true_label': label, 'pred_label': y_pred_all[np.where(np.concatenate(y_true_list) == y_true_all)[0][list(false_negatives[label]).index(idx)]]})
            records_fn.append(row)

    df_fp_all = pd.DataFrame.from_records(records_fp)
    df_fn_all = pd.DataFrame.from_records(records_fn)
    #print(df_fp_all)
    #print(df_fn_all)

        

def evaluate_xgb_folds(x, y, xgb_params = None, oversampling_params = None, do_sample_weights = False, do_oversampling = False, do_undersampling = False, weights = None, do_PCA = False, pca_parameters_twf = None, pca_parameters_hdf = None):
    f1_per_fold = []
    y_true_list = []
    y_pred_list = []
    # label setup
    class_map    = {0: 'Safe', 1: 'HDF', 2: 'OSF', 3: 'PWF', 4: 'TWF'}
    inv_map      = {v:k for k,v in class_map.items()}
    labels       = list(class_map.values())
    false_pos   = {lbl: [] for lbl in labels}
    false_neg   = {lbl: [] for lbl in labels}
    xgb_params = xgb_params or {}
    oversampling_params = oversampling_params or {}
    twf_w, osf_w, pwf_w, hdf_w, safe_w = weights if weights is not None else (None, None, None, None, None)
    n_pca_twf, t_twf = pca_parameters_twf if pca_parameters_twf is not None else (None, None, None)
    n_pca_hdf, t_hdf = pca_parameters_hdf if pca_parameters_hdf is not None else (None, None, None)

    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    for fold, (train_idx,test_idx) in enumerate(outer_cv.split(x, y), 1):
        x_train, y_train, x_test, y_test = split_data_cv(x, y, train_idx, test_idx)
        xgb = XGBClassifier(**xgb_params)

        if do_PCA:
            anomalies_test_twf = anomaly_detect_pca_xgb(x_train, x_test, n_c = n_pca_twf, th = t_twf)
            anomalies_test_hdf = anomaly_detect_pca_xgb(x_train, x_test, n_c = n_pca_hdf, th = t_hdf)

        if do_oversampling:
            oversampling = SVMSMOTE(**oversampling_params)
            x_train, y_train = oversampling.fit_resample(x_train, y_train)

        if do_undersampling:
            undersampling = TomekLinks(sampling_strategy = [0])
            mask_remove = (y_train == 4) | (y_train == 0)
            x_twf, y_twf = x_train[mask_remove], y_train[mask_remove]
            x_train = x_train[~mask_remove]
            y_train = y_train[~mask_remove]
            x_twf, y_twf = undersampling.fit_resample(x_twf, y_twf)
            x_train = pd.concat([x_train, x_twf])
            y_train = pd.concat([pd.Series(y_train), pd.Series(y_twf)])

        if do_sample_weights:
            sample_weights = sample_weights_gen_xgb_combined(x_train, y_train, twf_weight = twf_w, osf_weight = osf_w, pwf_weight = pwf_w, hdf_weight = hdf_w, safe_weight = safe_w)
        else:
            sample_weights = None
        
        xgb.fit(x_train, y_train, sample_weight = sample_weights)
        y_pred = xgb.predict(x_test)
        if do_PCA:
            y_pred[(anomalies_test_twf == False) & (y_pred == 4)] = 0
            y_pred[(anomalies_test_hdf == False) & (y_pred == 1)] = 0

        # track misclassifications
        safe_id = inv_map['Safe']
        for fault in ['HDF','OSF','PWF','TWF']:
            fid = inv_map[fault]
            fp = (y_pred==fid) & (y_test==safe_id)
            fn = (y_pred==safe_id) & (y_test==fid)
            false_pos[fault].extend(test_idx[fp])
            false_neg[fault].extend(test_idx[fn])

        f1 = f1_score(y_test, y_pred, average = 'macro', zero_division = 0)
        f1_per_fold.append(f1)
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)


    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    targets = [class_map[i] for i in range(len(class_map))]
    print("Classification Report Test:\n", classification_report(y_true_all, y_pred_all, target_names = targets, zero_division = 0))
    test_cm = confusion_matrix(y_true_all, y_pred_all) #, labels = labels) # Confusion matrix preperation
    h.print_multiple_class_confusion_matrix(test_cm) # Confusion matrix, layout
    print(xgb.get_params())
    mean_f1 = np.mean(f1_per_fold)
    std_f1 = np.std(f1_per_fold, ddof = 1)
    print(f'Macro-F1= {mean_f1:.3f} ± {std_f1:.3f} (5-fold CV)')
    if do_sample_weights:
        print('Sample Weights (TWF, All, Safe): ',weights)
    if do_oversampling:
        print(oversampling.get_params())

    # build full DataFrames & map back to text labels
    records_fp = []
    records_fn = []

    for fault in ['HDF','OSF','PWF','TWF']:
        # false positives: true_label was Safe, pred_label is the fault
        for idx in false_pos[fault]:
            row = x.iloc[idx].to_dict()
            row.update({
                'true_label': 'Safe',
                'pred_label':   fault
            })
            records_fp.append(row)

        # false negatives: true_label was the fault, pred_label was Safe
        for idx in false_neg[fault]:
            row = x.iloc[idx].to_dict()
            row.update({
                'true_label': fault,
                'pred_label': 'Safe'
            })
            records_fn.append(row)

    df_fp_all = pd.DataFrame(records_fp)
    df_fn_all = pd.DataFrame(records_fn)

    # write Excel (requires openpyxl/xlsxwriter installed)
    folder = '/Users/cemkarahan/Desktop/tez/Python Exports/Faults - XGB.xlsx'
    with pd.ExcelWriter(folder, engine='openpyxl') as w:
        df_fp_all.to_excel(w, sheet_name='False_Positives', index=False)
        df_fn_all.to_excel(w, sheet_name='False_Negatives', index=False)

    print(f"Exported {len(df_fp_all)} FP and {len(df_fn_all)} FN to '{folder}'")


def evaluate_dt_test(x, y, dt_params = None, smote_params = None):
    dt_params = None or {}
    smote_params = None or {}
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)
    dt = DecisionTreeClassifier(**dt_params)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print("Classification Report Test:\n", classification_report(y_test, y_pred, zero_division = 0))
    labels = ['Safe', 'HDF', 'OSF', 'PWF', 'TWF'] #'RNF', 
    test_cm = confusion_matrix(y_test, y_pred, labels = labels) # Confusion matrix preperation
    h.print_multiple_class_confusion_matrix(test_cm) # Confusion matrix, layout

def sample_weight_TWF(x_train, y_train, twf_weights, all_weights, safe_weights):
    weight_list = []
    twf_list = []
    aw_list = []
    safe_list = []

    for sw in safe_weights:
        for aw in all_weights:
            for twf in twf_weights:
                sample_weight = np.ones(len(y_train)) * aw
                for i, cls in enumerate(y_train):
                    if cls == 'Safe':
                        sample_weight[i] = sw
                    elif (cls == 'TWF') and (x_train['Tool wear [min]'].iloc[i] >= 200 and x_train['Tool wear [min]'].iloc[i] <= 240):
                        sample_weight[i] = twf
                    elif (cls == 'TWF') and (x_train['Tool wear [min]'].iloc[i] < 200 or x_train['Tool wear [min]'].iloc[i] > 240):
                        sample_weight[i] = 0.1
                weight_list.append(sample_weight)
                twf_list.append(twf)
                aw_list.append(aw)
                safe_list.append(sw)

    return weight_list, twf_list, aw_list, safe_list

def sample_weights_gen(x_train, y_train, twf_weight, all_weights, safe_weight):

    wear = x_train['Tool wear [min]'].values
    heat = x_train['Temperature Delta'].values
    power = x_train['Rotational speed [rpm]'].values * x_train['Torque [Nm]'].values
    overstrain = wear * x_train['Torque [Nm]']
    rot_speed = x_train['Rotational speed [rpm]'].values

    mask_twf = (y_train == 'TWF') & (wear >= 200) & (wear <= 240)
    mask_heat = ((heat < 8.59) & (rot_speed < 1279))
    mask_power = ((power < 3500 ) | (power > 9000))
    mask_overstrain = (overstrain > 11000)
    
    sample_weight = np.ones(len(y_train))
    sample_weight[mask_heat] = all_weights
    sample_weight[mask_power] = all_weights
    sample_weight[mask_overstrain] = all_weights
    sample_weight[mask_twf] = twf_weight
    sample_weight[y_train == 'Safe'] = safe_weight

    """sample_weight = np.ones(len(y_train)) * all_weights
    for i, cls in enumerate(y_train):
        if cls == 'Safe':
            sample_weight[i] = safe_weight
        elif (cls == 'TWF') and (x_train['Tool wear [min]'].iloc[i] >= 200 and x_train['Tool wear [min]'].iloc[i] <= 240):
            sample_weight[i] = twf_weight
        elif (cls == 'TWF') and (x_train['Tool wear [min]'].iloc[i] < 200 or x_train['Tool wear [min]'].iloc[i] > 240):
            sample_weight[i] = 0.1"""
    return sample_weight

def sample_weights_gen_combined(x_train, y_train, twf_weight, osf_weight, pwf_weight, hdf_weight, safe_weight):

    wear = x_train['Tool wear [min]'].values
    heat = x_train['Temperature Delta'].values
    power = (2 * np.pi / 60) *  x_train['Rotational speed [rpm]'].values * x_train['Torque [Nm]'].values # 
    overstrain = wear * x_train['Torque [Nm]'].values
    rot_speed = x_train['Rotational speed [rpm]'].values

    mask_twf = (y_train == 'TWF') & (wear >= 200) & (wear <= 240)
    mask_heat = ((heat < 8.6) & (rot_speed < 1379))
    mask_power = ((power < 3500 ) | (power > 9000))
    mask_overstrain = (overstrain > 11000)
    
    sample_weight = np.ones(len(y_train))
    sample_weight[mask_twf] = twf_weight
    sample_weight[mask_power] = pwf_weight
    sample_weight[mask_heat] = hdf_weight
    sample_weight[mask_overstrain] = osf_weight
    sample_weight[y_train == 'Safe'] = safe_weight
    return sample_weight

def sample_weights_gen_xgb(x_train, y_train, twf_weight, all_weights, safe_weight):

    wear = x_train['Tool wear'].values
    heat = x_train['Temperature Delta'].values
    power = x_train['Rotational speed'].values * x_train['Torque'].values
    overstrain = wear * x_train['Torque']
    rot_speed = x_train['Rotational speed'].values

    mask_twf = (y_train == 4) & (wear >= 200) & (wear <= 240)
    mask_heat = ((heat < 8.59) & (rot_speed < 1279))
    mask_power = ((power < 3500 ) | (power > 9000))
    mask_overstrain = (overstrain > 11000)
    
    sample_weight = np.ones(len(y_train))
    sample_weight[mask_heat] = all_weights
    sample_weight[mask_power] = all_weights
    sample_weight[mask_overstrain] = all_weights
    sample_weight[mask_twf] = twf_weight
    sample_weight[y_train == 0] = safe_weight

    return sample_weight

def sample_weights_gen_xgb_combined(x_train, y_train, twf_weight, osf_weight, pwf_weight, hdf_weight, safe_weight):

    wear = x_train['Tool wear'].values
    heat = x_train['Temperature Delta'].values
    power = (2 * np.pi / 60) *  x_train['Rotational speed'].values * x_train['Torque'].values #
    overstrain = wear * x_train['Torque']
    rot_speed = x_train['Rotational speed'].values

    mask_twf = (y_train == 4) & (wear >= 200) & (wear <= 240)
    mask_heat = ((heat < 8.59) & (rot_speed < 1279))
    mask_power = ((power < 3500 ) | (power > 9000))
    mask_overstrain = (overstrain > 11000)
    
    sample_weight = np.ones(len(y_train))
    sample_weight[mask_twf] = twf_weight
    sample_weight[mask_heat] = hdf_weight
    sample_weight[mask_power] = pwf_weight
    sample_weight[mask_overstrain] = osf_weight
    sample_weight[y_train == 0] = safe_weight

    return sample_weight

def svm_smote(x_train, y_train, twf):
    smote = SVMSMOTE(sampling_strategy = {'TWF': twf}, random_state = 42)
    x_train_oversampled, y_train_oversampled = smote.fit_resample(x_train, y_train)
    return x_train_oversampled, y_train_oversampled 

def run_experiment(params):
    # unpack params
    fold, x_train, y_train, x_test, y_test, twf_w, aw_w, safe_w = params

    print(f"[DEBUG] using twf_w={twf_w}, aw_w={aw_w}, safe_w={safe_w}")

    # oversample
    """sampler = SVMSMOTE(sampling_strategy=ss, k_neighbors=k, m_neighbors=10, random_state=42)
    x, y = sampler.fit_resample(x_train, y_train)"""

    sw = np.ones(len(y_train)) * 1.0
    y_arr = np.array(y_train)
    wear = x_train['Tool wear [min]'].values
    heat = x_train['Temperature Delta'].values
    power = x_train['Rotational speed [rpm]'].values * x_train['Torque [Nm]'].values
    overstrain = wear * x_train['Torque [Nm]']

    mask_twf = (y_arr == 'TWF') & (wear >= 200) & (wear <= 240)
    mask_heat = (heat <= 8.6)
    mask_power = ((power < 3500 ) | (power > 9000))
    mask_overstrain = (overstrain > 11000)
    mask_no_twf = (y_arr == 'TWF') & (wear < 200)

    
    sw[mask_heat] = aw_w
    sw[mask_power] = aw_w
    sw[mask_overstrain] = aw_w
    sw[mask_twf] = twf_w
    sw[mask_no_twf] = 0.1
    sw[y_arr == 'Safe'] = safe_w
    return sw