import functions_thesis as ft
import numpy as np

df = ft.read_data()
df = ft.data_prep_feature(df)
df = ft.parameter_creation(df)


### X & Y - Splitting Data ###
x_fields = [col for col in df.columns if col not in ['Failure Type', 'Safe', 'HDF', 'OSF', 'PWF',  'TWF']]
x = df[x_fields]
y = df['Failure Type']


ft.evaluate_rf_folds(x, y, rf_params = {'criterion': 'gini', 'random_state': 42, 'n_jobs': 8}, cal_params= {'cv': 5}, 
                     do_sample_weights = True, sample_weights = {'twf_weight': 9.0, 'osf_weight': 3.0, 'pwf_weight': 7.0, 'hdf_weight': 9.0, 'safe_weight': 0.8}, 
                     do_pca_anomaly = True, pca_parameters_twf = {'n_pca': 0.8, 't': 99.0}, pca_parameters_hdf = {'n_pca': 0.7, 't': 71.0}, do_oversampling = False, twf_oversample = 400, do_undersampling = False)



