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



    
if __name__ == '__main__':
    
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()


    # Get the folder path from the command line argument
    folder_path = args.pathfolder
    
    ############CREATE DUR FOR FIGURES
    folder_path_figure_stats = os.path.join(folder_path, 'saved_stats_figures')
    os.makedirs(folder_path_figure_stats, exist_ok=True)

    
    ############ GET WHERE STATS ARE
    folder_path_stats = os.path.join(folder_path, 'saved_stats')
    all_features_stats_files = [f for f in os.listdir(folder_path_stats) if 'all_features' in f]
    only_growth_rate_stats_files = [f for f in os.listdir(folder_path_stats) if 'only_growth' in f]

    
    
    ##########LOAD STATS
    df_all_features_stats = pd.read_csv(os.path.join(folder_path_stats,all_features_stats_files[0]))
    df_only_growth_rate_stats = pd.read_csv(os.path.join(folder_path_stats,only_growth_rate_stats_files[0]))
    
    
    ####lsit of stats
    list_pearsonr_only_growth_rate=[]
    list_rmse_only_growth_rate=[]
    for model_str in ['Lasso', 'RandomForest', 'MLPRegressor']:
        pearsonr_list=[]
        rmse_list=[]
        for i_t in range(DeepFate.config.NB_TIMESTEPS):
            model_t = f'{model_str}_{i_t}'
            pearsonr_list.append(df_only_growth_rate_stats[df_only_growth_rate_stats['model']==model_t]['pearsonr'].values)
            rmse_list.append(df_only_growth_rate_stats[df_only_growth_rate_stats['model']==model_t]['rmse'].values)
        
                      
        list_pearsonr_only_growth_rate.append(pearsonr_list)
        list_rmse_only_growth_rate.append(rmse_list)
        
    list_pearsonr_all_features=[]
    list_rmse_all_features=[]
    for model_str in ['Lasso', 'RandomForest', 'MLPRegressor']:
        pearsonr_list=[]
        rmse_list=[]
        for i_t in range(DeepFate.config.NB_TIMESTEPS):
            model_t = f'{model_str}_{i_t}'
            pearsonr_list.append(df_all_features_stats[df_only_growth_rate_stats['model']==model_t]['pearsonr'].values)
            rmse_list.append(df_all_features_stats[df_only_growth_rate_stats['model']==model_t]['rmse'].values)
        
    
        list_pearsonr_all_features.append(pearsonr_list)
        list_rmse_all_features.append(rmse_list)
    
    
    ##########PLOT
    
    
    fig, ax = plt.subplots(2,1, figsize=(7,12), constrained_layout=True)
    cmap_color=plt.get_cmap('PuBuGn')
    color_rmse = cmap_color(0.6)
    color_r = cmap_color(0.99)



    legend_elements = [Line2D([0], [0], color='k', lw=1,linestyle='-', label='Lasso'),
                        Line2D([0], [0], color='k', lw=1,linestyle='--', label='Random Forest'),
                      Line2D([0], [0], color='k', lw=1,linestyle='-.', label='MLP')]


    #######top
    ax2 = plt.twinx(ax[0])

    label_list = ['Lasso','Random Forest','MLP']
    linestyle_list = ['-', '--', '-.']
    i=0

    x = ['0.5', '1', '1.5', '2', '2.5', '3','3.5','4','4.5','5']

    for list_rmse, list_pearson in zip(list_rmse_only_growth_rate, list_pearsonr_only_growth_rate):
        label=label_list[i]
        ax[0].plot(x,list_rmse, linestyle_list[i], color=color_rmse)
        ax2.plot(x,list_pearson,  linestyle_list[i], color=color_r)
        ax2.plot([], [], linestyle_list[i], label=label, color=color_r)
        i=i+1


    ax[0].set_xticks(['0.5', '1', '1.5', '2', '2.5', '3','3.5','4','4.5','5'])

    ax2.yaxis.label.set_color(color_r)
    ax2.tick_params(axis='y', colors=color_r)

    ax[0].yaxis.label.set_color(color_rmse)
    ax[0].tick_params(axis='y', colors=color_rmse)

    ax[0].hlines(30, 0.5, 5.0, linestyle='dotted')

    ax[0].set_ylim(3,33)
    ax2.set_ylim(0,1)

    ax[0].grid(True)

    plt.legend(handles=legend_elements,loc='upper center', borderaxespad=-3, ncol=3)

    ax[0].set_xlabel('Time of Observation [h]')
    ax[0].set_ylabel(r'Mean Square error [$km$]')
    ax2.set_ylabel(r'Pearson-r regression score [1]')

    ax2.set_ylim(0,1)
    ax[0].set_ylim(5,45)

    ax[0].text(5.0, 22.0, 'Only Growth Rate', style='italic', bbox={'facecolor': 'tab:blue', 'alpha': 0.3, 'pad': 10})


    ######### bottom

    label_list = ['Lasso', 'Random Forest','MLP']
    linestyle_list = ['-', '--', '-.']
    i=0

    x = ['0.5', '1', '1.5', '2', '2.5', '3','3.5','4','4.5','5']

    ax3 = plt.twinx(ax[1])


    for list_rmse, list_pearson in zip(list_rmse_all_features, list_pearsonr_all_features):
        label=label_list[i]
        ax[1].plot(x,list_rmse, linestyle_list[i], color=color_rmse)
        ax3.plot(x,list_pearson,  linestyle_list[i], color=color_r)
        ax3.plot([], [], linestyle_list[i], color='k')
        i=i+1

    i=-1
    for list_rmse, list_pearson in zip(list_rmse_all_features, list_pearsonr_all_features):
        label=label_list[i]
        ax[1].plot(x,list_rmse, linestyle_list[i], color=color_rmse, alpha=0.5)
        ax3.plot(x,list_pearson,  linestyle_list[i], color=color_r, alpha=0.5)
        ax3.plot([], [], linestyle_list[i], color='k')
        i=i+1

    ax[1].set_xlabel('Time of Observation [h]')
    ax[1].set_ylabel(r'Mean Square error [$km$]')
    ax3.set_ylabel(r'Pearson-r regression score [1]')


    ax3.set_ylim(0,1)
    ax[1].set_ylim(5,45)




    ax[1].grid(True)
    ax3.yaxis.label.set_color(color_r)
    ax3.tick_params(axis='y', colors=color_r)

    ax[1].yaxis.label.set_color(color_rmse)
    ax[1].tick_params(axis='y', colors=color_rmse)

    ax[1].text(5.0, 22.0, 'All features', style='italic', bbox={'facecolor': 'tab:blue', 'alpha': 0.3, 'pad': 10})
    
    
    path_save_fig=os.path.join(folder_path_figure_stats, 'figure_1.pdf')
    plt.savefig(path_save_fig)