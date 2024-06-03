import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import DeepFate
from DeepFate.model.utils_model import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse
from matplotlib.gridspec import GridSpec
import ast
from scipy.stats import gaussian_kde



if __name__ == '__main__':
   
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()
    
    #### GET the data
    # Get the folder path from the command line argument
    folder_path = args.pathfolder
    
        
    folder_path_preds = os.path.join(folder_path, 'saved_preds')
    path_to_preds_df = os.path.join(folder_path_preds, 'preds_all_models_all_features.parquet')
    df_all_features=pd.read_parquet(path_to_preds_df)

    ############CREATE DIR FOR FIGURES
    folder_path_figure_preds = os.path.join(folder_path, 'saved_preds_figures')
    os.makedirs(folder_path_figure_preds, exist_ok=True)

    
    #### PLOTS
    for i_model,model_str in enumerate(['Lasso', 'RandomForest', 'MLPRegressor']):

        fig = plt.figure(layout="constrained",  figsize=(8,18))

        gs = GridSpec(100, 6, figure=fig)

        ax1 = fig.add_subplot(gs[:30, :])
        ax2 = fig.add_subplot(gs[30:60, :])
        ax4 = fig.add_subplot(gs[60:70, :])
        ax3 = fig.add_subplot(gs[70:99, :])
        ax33 = fig.add_subplot(gs[99:100, :])

        #### ax1
        nb_timesteps=3
        str_dict = f'{model_str}_{nb_timesteps}'

        y_preds =df_all_features[df_all_features['model']==str_dict]['preds'].values
        y_test =df_all_features[df_all_features['model']==str_dict]['truth'].values


        x=y_preds[0]
        y = y_test[0]

        xy = np.vstack([x,y])
        xy=np.array(xy)
        z = gaussian_kde(xy)(xy)


        im = ax1.scatter( y_test[0],  y_preds[0], marker = '+', c =1000*z, alpha=0.7, cmap='YlGnBu_r')

        ax1.set_xlim(40,220)
        ax1.set_ylim(40,220)
        #plt.title('Multi-Linear Model, all features, 1h', fontstyle='italic')
        ax1.set_xlabel('Maximal Extension (Ground Truth) [km]')
        ax1.set_ylabel('Prediction [km]')



        ax1.plot(y_test[0], y_test[0], color='k', linestyle='dashed')
        #plt.plot(y_test, 1.13*y_test-13.78967328)
        ax1.grid(True)
        #plt.plot(y_test, y_test)

        ax1.text(50, 180, f'All features, {nb_timesteps/2}h', style='italic', bbox={'facecolor': 'tab:blue', 'alpha': 0.5, 'pad': 10})



        ### ax2
        nb_timesteps=4
        str_dict = f'{model_str}_{nb_timesteps}'
        y_preds =df_all_features[df_all_features['model']==str_dict]['preds'].values
        y_test =df_all_features[df_all_features['model']==str_dict]['truth'].values

        x=y_preds[0]
        y = y_test[0]
        xy = np.vstack([x,y])
        xy = np.array(xy)
        
        print(xy.shape)
        
        z = gaussian_kde(xy)(xy)


        #im = plt.scatter( y_test**2,  0.5*(y_preds_lr**2+y_preds_rf**2), marker = '+', c = df_test['y_duration'], alpha=0.6)
        #im = plt.scatter( y_test, 0.5*(y_preds_lr+y_preds_rf), marker = '+', c =z)
        im = ax2.scatter( y_test[0],  y_preds[0], marker = '+', c =1000*z, alpha=0.7, cmap='YlGnBu_r')

        ax2.set_xlim(40,220)
        ax2.set_ylim(40,220)
        #plt.title('Multi-Linear Model, all features, 1.5h', fontstyle='italic')
        ax2.set_xlabel('Maximal Extension (Ground Truth) [km]')
        ax2.set_ylabel('Prediction [km]')

        #ax[1].clim(0,0.5)


        ax2.plot(y_test[0], y_test[0], color='k', linestyle='dashed')
        #plt.plot(y_test, 1.13*y_test-13.78967328)
        ax2.grid(True)
        #plt.plot(y_test, y_test)

        ax2.text(50, 180, f'All features, {nb_timesteps/2}h', style='italic', bbox={'facecolor': 'tab:blue', 'alpha': 0.5, 'pad': 10})

        fig.colorbar(im, orientation='horizontal', label=r'Density [$10^{-3}$]', aspect=110, pad=-0.7, ax=ax4)


        ######## ax3
        """
        cmap=plt.get_cmap('PuBuGn',5)
        N=100
        quantile_size = [np.quantile(y_test, i/N) for i in range(1,N+1)]

        N2=10
        quantile_size2 = [np.quantile(y_test, i/N2) for i in range(1,N2+1)]
        quantile_size2.insert(0, 50)



        rmse_values=[]
        rmse_values_list=[]

        for nb_timesteps in [1, 2, 3, 4]:
            str_dict = f'{model_str}_{nb_timesteps}'
            y_preds = df_all_features[df_all_features['model']==str_dict]['preds'].values
            y_test =df_all_features[df_all_features['model']==str_dict]['truth'].values

            rmse_values = []

            y_preds = np.array(y_preds[0])
            y_test = np.array(y_test[0])

            for size in quantile_size2:
                idx = np.where(y_test <= size)[0]
                rmse_values.append(mean_squared_error(y_preds[idx], y_test[idx]))
            ax3.plot(quantile_size2, rmse_values, '+-', color=cmap(nb_timesteps), label=f'Period : {nb_timesteps*0.5}h')


            n_start=2
            n_end=8
            a = (rmse_values[n_end]-rmse_values[n_start])/(quantile_size2[n_end]-quantile_size2[n_start])
            ax3.plot(quantile_size2[n_start:n_end+1], a*(quantile_size2[n_start:n_end+1]-quantile_size2[n_start])+rmse_values[n_start], color=cmap(nb_timesteps), ls='--', linewidth=2, alpha=0.4)
            #ax3.text( quantile_size2[n_start], rmse_values[n_start]+0.065, f'{np.round(a*100, 2)}%', fontstyle='italic', color=cmap(nb_timesteps),  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))


            n_start=8
            n_end=10
            a = (rmse_values[n_end]-rmse_values[n_start])/(quantile_size2[n_end]-quantile_size2[n_start])
            ax3.plot(quantile_size2[n_start:], a*(quantile_size2[n_start:]-quantile_size2[n_start])+rmse_values[n_start], color=cmap(nb_timesteps), ls='--', linewidth=2, alpha=0.4)
            #ax3.text(quantile_size2[-2], rmse_values[-1]+0.03, f'{np.round(a*100, 2)}%', fontstyle='italic', color=cmap(nb_timesteps))

            ax33.plot(quantile_size2, rmse_values, alpha=0)

                #ax3.plot(n)#, capsize=4, capthick=2, color=cmap(nb_timesteps), label=f'Period : {nb_timesteps*0.5}h')


        #im2=ax3.scatter(y_test, 100*(np.sqrt((np.mean(y_test)-y_test)**2)/y_test), marker='+', c='k', s=1, label='Constant Prediction')
        #ax3.set_ylim(-5,80)
        

            
        #plt.title('Lasso, only growth rate area', fontstyle='italic')

        data = quantile_size
        # Calculate percentiles for ticks
        percentiles_for_ticks = [1,  25, 50, 75, 90, 100]


        ax3.vlines(quantile_size2[8],0.15,1, color='k', ls='--')
        # Calculate the values at the specified percentiles

        ticks_values = np.percentile(data, percentiles_for_ticks)

        # Create a mapping of percentile to clean label
        percentile_label_mapping = {p: f'{p}%' for p in percentiles_for_ticks}

        # Calculate the percentile rank for each tick value
        tick_percentile_ranks = [percentileofscore(data, value) for value in ticks_values]



    
        ax33.yaxis.set_visible(False) # hide the yaxis
        ax33.set_xticks(ticks_values)
        ax33.set_xticklabels([percentile_label_mapping[p] for p in percentiles_for_ticks])


        ax3.set_xlim(40,220)
        #ax3.set_ylim(-0.1,1)


        ax33.set_xlim(40,220)
        ax33.spines['top'].set_visible(False)
        ax33.spines['right'].set_visible(False)
        ax33.spines['left'].set_visible(False)
        ax33.spines['bottom'].set_position('zero')
        ax33.set_xlabel('Percentile Rank [%]')


        ax3.set_xlabel('Maximal Extension [km]')
        ax3.set_ylabel('Cumulative Mean-Square Error')
        ax3.legend(ncol=2,  loc= 'upper right')
        ax3.grid(True)
        """
        ax4.set_axis_off()


        path_to_save = os.path.join(folder_path_figure_preds, f'{model_str}_preds.pdf')
        plt.savefig(path_to_save)

