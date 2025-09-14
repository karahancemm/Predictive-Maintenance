import functions_thesis as ft
from collections import Counter

df = ft.read_data()
df = ft.data_prep_feature(df)
df = ft.parameter_creation(df)


### X & Y - Splitting Data ###
x_fields = [col for col in df.columns if col not in ['Failure Type', 'Safe', 'HDF', 'OSF', 'PWF',  'TWF']] # 'RNF',
x = df[x_fields]
y = df['Failure Type']
class_map = {'Safe': 0, 'HDF': 1, 'OSF': 2, 'PWF': 3, 'TWF': 4}
y = y.map(class_map)
x.rename(columns = {'Rotational speed [rpm]': 'Rotational speed', 'Torque [Nm]': 'Torque', 'Tool wear [min]': 'Tool wear'}, inplace = True)

ratio = 0.05
counts = Counter(y)
sampling_twf = int(counts[0] * ratio)
sampling_pwf = int(counts[0] * 0.2)
sampling_osf = int(counts[0] * 0.05)
sampling_hdf = int(counts[0] * 0.3)

ft.evaluate_xgb_folds(x, y, 
                      xgb_params = {'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.3, 'min_child_weight': 3.0, 'max_depth': 6, 'gamma': 0.30, 'reg_alpha': 0.7, 'reg_lambda': 2.5},
                      oversampling_params = {'sampling_strategy': {4: 700}, 'random_state': 42}, #, 3: sampling_pwf, 2: sampling_osf, 1: sampling_hdf}},  
                        do_sample_weights = True,
                        do_oversampling = False,
                        do_undersampling = False, 
                        weights = ft.weights(6.6, 8.0, 5.2, 1.0, 0.8),
                                           # twf, osf, pwf, hdf, safe  
                        do_PCA = True,
                        pca_parameters_twf = (0.80, 98.5), 
                        pca_parameters_hdf = (0.70, 67.0)
                        )

