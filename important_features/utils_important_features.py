
import matplotlib.pyplot as plt
import numpy as np
from DeepFate.models.utils_models import get_X_cols, get_one_feature
import pandas as pd
#import DeepFate
from DeepFate.models.utils_models import get_model
import os
import joblib


def get_datasets(all_features: bool, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 nb_timesteps: int, one_features_name: str, blacklist=[], wantlist=None):

    if all_features:
        cols = get_X_cols(df_train.columns, nb_timesteps=nb_timesteps, blacklist=blacklist, wantlist=wantlist)
    else:
        cols = get_one_feature(df_train, one_features_name)
        cols = get_X_cols(cols, nb_timesteps=nb_timesteps)
    
    train_dataset = df_train[cols]
    test_dataset = df_test[cols]

    return train_dataset, test_dataset


def get_features_importance(path_model, model_str, df_train, nb_timesteps, exp='ALL_FEATURES', no_corr_input=False):
    
    #print(len(df_train.columns))
    if not no_corr_input:
        path_file = os.path.join(path_model,f'{model_str}_{exp}_{nb_timesteps}.joblib')
    else:
        path_file = os.path.join(path_model,f'{model_str}_{exp}_{nb_timesteps}_presel.joblib')
    model = get_model(model_str)
    model = joblib.load(path_file)

    #print(path_file)
    
    if model_str=='Lasso':
        feature_importances = model.coef_
        feature_importances_abs = np.abs(model.coef_)
        #print(len(feature_importances_abs))

    elif model_str=='RandomForest':
        feature_importances = model.feature_importances_
        feature_importances_abs = np.abs(model.feature_importances_)
        
    index_argsort = np.argsort(feature_importances_abs)[::-1]
    #print(index_argsort)
    values_features = feature_importances[index_argsort]
    
    #columns_cut = get_X_cols(df_train.columns, nb_timesteps=nb_timesteps)
    #df_train_cut = df_train[columns_cut]
    features_selected_names = df_train.columns[index_argsort]

    return features_selected_names, values_features, index_argsort

def get_features_importance_pipeline(pipeline_path, model_str, df_train):
    #get pipeline    
    pipeline = joblib.load(pipeline_path)
    #get model
    model = pipeline.named_steps[model_str]
    if model_str=='lasso':
        feature_importances = model.coef_
        feature_importances_abs = np.abs(model.coef_)
    
    elif model_str == 'rf':
        # Random forest feature importance is given by feature_importances_
        feature_importances = model.feature_importances_
        feature_importances_abs = np.abs(feature_importances)
        
    index_argsort = np.argsort(feature_importances_abs)[::-1]
    values_features = feature_importances[index_argsort]
    
    #get intput
    dropped_columns = pipeline.named_steps['remove_correlated'].to_drop_
    remaining_columns = df_train.drop(columns=dropped_columns).columns
    final_feature_mask = pipeline.named_steps['feature_selection'].get_support()
    features_selected_for_train = remaining_columns[final_feature_mask]

    features_selected_names = features_selected_for_train[index_argsort]

    return features_selected_names, values_features, index_argsort

def plot_features_importance(model_str, nb_selected, path_model, df_train, nb_timesteps, automatic_name:bool=False, exp='ALL_FEATURES',no_corr_input=False):
    features_selected_names, values_features, index_argsort = get_features_importance(path_model=path_model,
                        model_str=model_str,
                        df_train=df_train,
                        nb_timesteps=nb_timesteps,
                        exp=exp,
                        no_corr_input=no_corr_input)
    
    #print(len(features_selected_names))
    
    if automatic_name:
        #print('yes automatic')
        df_names = pd.read_csv('/home/b/b381993/DeepFate/notebooks/name_dict_new_var.csv')
        features_selected_names= [df_names[df_names['Old Name']==features_selected_names[i]]['Name Plot'].values[0] for i in range(len(features_selected_names))]


    fig = plt.figure(figsize=(10,6), constrained_layout=True)

    sel_features = features_selected_names[:nb_selected][::-1]
    #print(len(sel_features))
    y_pos = np.arange(len(sel_features))

    performance = values_features[:nb_selected][::-1]

    plt.barh(y_pos, performance, align='center', color=plt.get_cmap('Spectral')(0.85))


    #labels= [df_names[df_names['Old Name']==sel_features[i]]['Name Plot'].values[0] for i in range(15)]

    plt.yticks(y_pos, labels=sel_features)

    #plt.invert_yaxis()  # labels read top-to-bottom
    plt.xlabel('Coefficients')
    plt.title(f'Important Features and attributed coefficients for {model_str} on {exp}', fontstyle='italic')
    plt.grid(True)


    plt.tight_layout()
    plt.show()

def get_sys_env_terms(X_input, coefs, index_sys, index_env):
    X_input_env = X_input[:,index_env]
    X_input_sys = X_input[:,index_sys]

    sum_env = np.sum(X_input_env*coefs[index_env], axis=1)
    sum_sys = np.sum(X_input_sys*coefs[index_sys], axis=1)

    return sum_sys, sum_env



def get_variances(X_input, coefs, index_sys, index_env):
    #print(X_input.shape)
    X_input_env = X_input[:,index_env]
    X_input_sys = X_input[:,index_sys]
    #print('coefs', coefs)
    var_vector_all = []

    for i in range(X_input.shape[1]):
        var_vector_all.append(coefs[i]**2 * np.var(X_input[:, i]))
    var_vector_all = np.array(var_vector_all)
    
    var_sum_env = np.var(np.sum(X_input_env*coefs[index_env], axis=1))
    #print(var_sum_env)
    sum_var_env = np.sum(var_vector_all[index_env])
    covar_env = var_sum_env - sum_var_env
    
    var_sum_sys = np.var(np.sum(X_input_sys*coefs[index_sys], axis=1))
    #print(var_sum_sys)
    sum_var_sys = np.sum(var_vector_all[index_sys])
    covar_sys = var_sum_sys - sum_var_sys
    
    var_sum_all = np.var(np.sum(X_input*coefs, axis=1))
    covar_env_vs_sys = var_sum_all - var_sum_sys - var_sum_env
    
    return {'var_env':var_sum_env, 'var_sys':var_sum_sys, 'var_all':var_sum_all, 
            'covar_sys':covar_sys, 'covar_env':covar_env, 'covar_env_vs_sys':np.abs(covar_env_vs_sys), 
            'var_vector_all':var_vector_all}