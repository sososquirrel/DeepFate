import argparse
import pandas as pd
import joblib
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from scipy.stats import pearsonr
from DeepFate.models.RemoveCorrelatedFeatures import RemoveCorrelatedFeatures
from DeepFate.important_features.utils_important_features import get_datasets
import os

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with dataset and save it")
    
    parser.add_argument('--train_data', type=str, default='/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/train_dataset.csv', 
                        help="Path to the training dataset CSV file")
    parser.add_argument('--test_data', type=str, default='/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/test_dataset.csv', 
                        help="Path to the testing dataset CSV file")
    parser.add_argument('--save_model', type=str, default='/home/b/b381993/DeepFate/models/saved_models', 
                        help="Directory where the trained model should be saved")
    
    return parser.parse_args()

# Main function
if __name__ == '__main__':
    args = parse_args()

    # Load datasets
    train_dataset_raw = pd.read_csv(args.train_data)
    test_dataset_raw = pd.read_csv(args.test_data)

    ########ONLY GROWTH RATE
    dict_exp1 = {'df_train': train_dataset_raw, 
                 'df_test': test_dataset_raw, 
                 'all_features': False, 
                 'one_features_name': 'gradient_area'}
    
    for nt_eval in range(1, 11):
        train_dataset, test_dataset = get_datasets(
            all_features=dict_exp1['all_features'], 
            df_train=dict_exp1['df_train'], 
            df_test=dict_exp1['df_test'], 
            one_features_name=dict_exp1['one_features_name'], 
            nb_timesteps=nt_eval
        )

        y_train = train_dataset_raw['y_max_extend']

        lasso_estimator = Lasso(alpha=0.1)
        pipeline_lasso = Pipeline([
            ('scaler', StandardScaler()),  
            ('lasso', lasso_estimator)
        ])

        pipeline_lasso.fit(train_dataset, y_train)

        model_save_path = os.path.join(args.save_model, f'pipeline_lasso_only_growth_rate_{nt_eval}.pkl')
        joblib.dump(pipeline_lasso, model_save_path)

        y_pred = pipeline_lasso.predict(test_dataset)

        # Compare predictions to true values
        y_test = test_dataset_raw['y_max_extend']
        pearsonr_score = pearsonr(y_test, y_pred)
        print('check!!!', pearsonr_score)

    #######ALL FEATURES
    print(len(train_dataset_raw), len(test_dataset_raw))

    blacklist_exp = ['gradient_area', 'average_diameter', 'mcs_area']
    dict_exp = {'df_train': train_dataset_raw,
                'df_test': test_dataset_raw,
                'all_features': True,
                'one_features_name': '',
                'blacklist': blacklist_exp}
    
    for nt_eval in [10]:
        train_dataset, test_dataset = get_datasets(
            all_features=dict_exp['all_features'], 
            df_train=dict_exp['df_train'], 
            df_test=dict_exp['df_test'], 
            one_features_name=dict_exp['one_features_name'], 
            blacklist=dict_exp['blacklist'], 
            nb_timesteps=nt_eval
        )

        print(len(train_dataset), len(test_dataset))

        y_train = train_dataset_raw['y_max_extend']

        # Define the feature selection and correlation removal steps
        selector = GenericUnivariateSelect(score_func=f_regression, mode='percentile', param=30)

        # Assuming RemoveCorrelatedFeatures is a custom function you implemented
        remove_correlated = RemoveCorrelatedFeatures(threshold=0.85, 
                                                    priority_vars=['var_9', 'var_13', 'var_19', 'var_14', 'var_17', 'var_18'],
                                                    blacklist_vars=[])

        # Create the pipeline
        lasso_estimator = Lasso(alpha=0.1)

        pipeline_lasso = Pipeline([
            ('remove_correlated', remove_correlated),
            ('feature_selection', selector),
            ('scaler', StandardScaler()),  
            ('lasso', lasso_estimator)
        ])

        print('start training')
        pipeline_lasso.fit(train_dataset, y_train)
        print('end training')

        y_pred = pipeline_lasso.predict(test_dataset)

        # Compare predictions to true values
        y_test = test_dataset_raw['y_max_extend']
        pearsonr_score = pearsonr(y_test, y_pred)
        print('check!!!', pearsonr_score)

        model_save_path = os.path.join(args.save_model, f'pipeline_lasso_{nt_eval}.pkl')
        joblib.dump(pipeline_lasso, model_save_path)
