import sys
module_dir = '/home/b/b381993'

sys.path.append(module_dir)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
import joblib
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import GenericUnivariateSelect, f_regression

from DeepFate.important_features.utils_important_features import get_datasets
from DeepFate.models.RemoveCorrelatedFeatures import RemoveCorrelatedFeatures

if __name__ == '__main__':
    train_dataset_raw = pd.read_csv('/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/train_dataset.csv')
    test_dataset_raw = pd.read_csv('/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/test_dataset.csv')

    ########ONLY GROWTH RATE
    dict_exp1 = {'df_train': train_dataset_raw, 
                 'df_test': test_dataset_raw, 
                 'all_features': False, 
                 'one_features_name': 'gradient_area'}
    
    for nt_eval in range(1, 11):
    
        # Get the processed datasets for the given timestep (nt_eval)
        train_dataset, test_dataset = get_datasets(
            all_features=dict_exp1['all_features'], 
            df_train=dict_exp1['df_train'], 
            df_test=dict_exp1['df_test'], 
            one_features_name=dict_exp1['one_features_name'], 
            nb_timesteps=nt_eval
        )

        y_train = train_dataset_raw['y_max_extend']

        # Define MLP estimator with the given parameters
        mlp_estimator = MLPRegressor(hidden_layer_sizes=(100,), 
                                     max_iter=1000, 
                                     alpha=0.001, 
                                     random_state=42, 
                                     early_stopping=True, 
                                     verbose=False)

        # Create pipeline with scaling and MLP
        pipeline_mlp = Pipeline([
            ('scaler', StandardScaler()),  
            ('mlp', mlp_estimator)
        ])

        # Fit the pipeline to the training data
        pipeline_mlp.fit(train_dataset, y_train)

        # Save the trained pipeline
        joblib.dump(pipeline_mlp, f'/home/b/b381993/DeepFate/models/saved_models/pipeline_mlp_only_growth_rate_{nt_eval}.pkl')

        # Make predictions
        y_pred = pipeline_mlp.predict(test_dataset)

        # Compare predictions to true values
        y_test = test_dataset_raw['y_max_extend']
        pearsonr_score = pearsonr(y_test, y_pred)
        print(f'Check for timestep {nt_eval}: Pearson correlation = {pearsonr_score[0]}')

    
    #######ALL FEATURES (for completeness, you can choose to activate this part)
    
    blacklist_exp = ['gradient_area', 'average_diameter', 'mcs_area']
    dict_exp = {'df_train': train_dataset_raw,
                'df_test': test_dataset_raw,
                'all_features': True,
                'one_features_name': '',
                'blacklist': blacklist_exp}
    
    #for nt_eval in range(1, 10):
    for nt_eval in [10]:
        train_dataset, test_dataset = get_datasets(
            all_features=dict_exp['all_features'], 
            df_train=dict_exp['df_train'], 
            df_test=dict_exp['df_test'], 
            one_features_name=dict_exp['one_features_name'], 
            blacklist=dict_exp['blacklist'], 
            nb_timesteps=nt_eval
        )

        y_train = train_dataset_raw['y_max_extend']

        # Feature selection and correlation removal
        selector = GenericUnivariateSelect(score_func=f_regression, mode='percentile', param=30)

        remove_correlated = RemoveCorrelatedFeatures(threshold=0.85, 
                                                    priority_vars=['var_9', 'var_13', 'var_19', 'var_14', 'var_17', 'var_18'],
                                                    blacklist_vars=[])

        # MLP Regressor pipeline
        mlp_estimator = MLPRegressor(hidden_layer_sizes=(100,), 
                                     max_iter=1000, 
                                     alpha=0.001, 
                                     random_state=42, 
                                     early_stopping=True, 
                                     verbose=False)
        
        pipeline_mlp = Pipeline([
            ('remove_correlated', remove_correlated),
            ('feature_selection', selector),
            ('scaler', StandardScaler()),  
            ('mlp', mlp_estimator)
        ])

        print(f'Start training for timestep {nt_eval}')
        pipeline_mlp.fit(train_dataset, y_train)
        print(f'End training for timestep {nt_eval}')

        # Predictions and evaluation
        y_pred = pipeline_mlp.predict(test_dataset)

        y_test = test_dataset_raw['y_max_extend']
        pearsonr_score = pearsonr(y_test, y_pred)
        print(f'Check for timestep {nt_eval}: Pearson correlation = {pearsonr_score[0]}')

        # Save the trained pipeline
        joblib.dump(pipeline_mlp, f'/home/b/b381993/DeepFate/models/saved_models/pipeline_mlp_{nt_eval}.pkl')

