import argparse
import os
import argh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm
import matplotlib.lines as mlines


from helper_files.data_loader import load_data


def _old_plot_proportions(save_folder, proportions_dict, iteration_range, baselines, model_config, params_dict,
                          focus_country, plot_type=None):
    """
    Plot the proportions of local and focus_country (mostly US) track recommendations.

    Parameters:
    - proportions_dict: A dictionary containing the proportions of local and focus_country (mostly US) tracks.
    - iteration_range: A list of iteration numbers.
    - baselines: A dictionary containing the baseline proportions.
    - model_config: A tuple containing the model name and the model type.
    - params_dict: A dictionary containing the parameters used for the experiment.
    - focus_country: The country code for the focus group.
    - plot_type: The type of plot to be generated (interaction or recommendation).

    Returns:
    A plot showing the proportions of local and focus_country (mostly US) track recommendations.
    """

    plt.figure(figsize=(15, 7))
    plt.plot(iteration_range, proportions_dict, label=f'{focus_country} Proportion', color='orange', linestyle='-')

    # Filling the areas under the curves
    plt.fill_between(iteration_range, proportions_dict, alpha=0.1, color='orange')

    # Plotting the baseline proportions as horizontal lines
    plt.hlines(y=baselines.query("country == 'global'")[focus_country.lower()].values[0], xmin=iteration_range[0],
               xmax=iteration_range[-1], colors='orange', linestyles='--', label=f'Global Baseline {focus_country}')

    # Ensure the first tick is displayed on the x-axis
    if iteration_range[0] not in plt.xticks()[0]:
        plt.xticks([iteration_range[0]] + list(plt.xticks()[0]))

    plt.xlim(iteration_range[0], iteration_range[-1])
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    plt.xlabel('Iteration')
    plt.ylabel(f'Proportion of tracks to {focus_country}')
    plt.legend(loc='upper right')

    if plot_type == 'interaction':
        plt.title(
            f'Country Interaction Distribution ({model_config[0]}, {model_config[1]}, {params_dict["dataset_name"]})')
        plt.savefig(os.path.join(save_folder, f'{model_config[0]}_{model_config[1]}_interaction_proportions.png'))
    elif plot_type == 'recommendation':
        plt.title(
            f'Country Recommendation Distribution ({model_config[0]}, {model_config[1]}, {params_dict["dataset_name"]})')
        plt.savefig(os.path.join(save_folder, f'{model_config[0]}_{model_config[1]}_recommendation_proportions.png'))


def _old_plot_jsd(save_folder, iteration_range, jsd_values, model_config, params_dict, focus_country, plot_type=None):
    """
    Plot the progression of average JSD between history and recommendations at each iteration.

    Parameters:
    - iteration_range: A list of iteration numbers.
    - jsd_values: A list of JSD values for each iteration.
    - save_folder: Directory where the plot will be saved.
    - model_config: A tuple containing the model name and the model type.
    - params_dict: A dictionary containing the parameters used for the experiment.
    - focus_country: The country code for the focus group.
    - plot_type: The type of plot to be generated (interaction or recommendation).

    Returns:
    A plot showing the progression of average JSD between history and recommendations at each iteration.
    """

    plt.figure(figsize=(15, 7))
    plt.plot(iteration_range, jsd_values, label='Average JSD', color='green', linestyle='-')

    plt.xlabel('Iteration')

    # Ensure the first tick is displayed on the x-axis
    if iteration_range[0] not in plt.xticks()[0]:
        plt.xticks([iteration_range[0]] + list(plt.xticks()[0]))

    # Adding labels and title
    plt.ylabel('Average JSD')
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlim(iteration_range[0], iteration_range[-1])

    if plot_type == 'interaction':
        plt.title(
            f'JSD between History and Interactions of {focus_country} ({model_config[0]}, {model_config[0]}, {params_dict["dataset_name"]})')
        plt.savefig(os.path.join(save_folder, f'{model_config[0]}_{model_config[1]}_interaction_jsd.png'))
    elif plot_type == 'recommendation':
        plt.title(
            f'JSD between History and Recommendations of {focus_country} ({model_config[0]}, {model_config[1]}, {params_dict["dataset_name"]})')
        plt.savefig(os.path.join(save_folder, f'{model_config[0]}_{model_config[1]}_recommendation_jsd.png'))


def plot_proportions(experiment_data, save_folder, iteration_range, control_country, colors):
    """2x2 layout plotting US/local proportions for recommendations and interaction history"""
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    # fig.tight_layout()

    # plt.suptitle(f'US and Local Proportions from {control_country} users', size=22)
    # fig.subplots_adjust(top=0.95)

    label_font_size = 22
    legend_font_size = 18
    tick_font_size = 22

    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].set_xlim(iteration_range[0], iteration_range[-1])
            ax[i, j].set_xlabel('Iteration', fontsize=label_font_size)
            ax[i, j].set_ylim(0, 1)
            if i == 0:
                ax[i, j].set_ylabel('% US items recommended', fontsize=label_font_size)
            else:
                ax[i, j].set_ylabel('% Local items recommended', fontsize=label_font_size)
            ax[i, j].grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
            ax[i, j].tick_params(axis='both', labelsize=tick_font_size)

    ax[0, 0].set_title('US items in Recommendations', fontsize=label_font_size)
    ax[0, 1].set_title('US items in Interaction History', fontsize=label_font_size)
    ax[1, 0].set_title('Local items in Recommendations', fontsize=label_font_size)
    ax[1, 1].set_title('Local items in Interaction History', fontsize=label_font_size)

    baseline_us = None
    baseline_local = None

    for idx, ((model, choice_model), entry) in enumerate(experiment_data.items()):
        metrics = entry['metrics']
        if not baseline_us:
            metrics_global = metrics[(metrics['country'] == control_country) & (metrics['iteration'] == 1)]
            baseline_us = metrics_global.interaction_us_proportion.values[0]
            baseline_local = metrics_global.interaction_local_proportion.values[0]
        metrics = metrics[metrics['country'] == control_country]
        ax[0, 0].plot(iteration_range, metrics['us_proportion'].values, label=f'{model}', color=colors[idx])
        ax[0, 1].plot(iteration_range, metrics['interaction_us_proportion'].values, label=f'{model}', color=colors[idx])
        ax[1, 0].plot(iteration_range, metrics['local_proportion'].values, label=f'{model}', color=colors[idx])
        ax[1, 1].plot(iteration_range, metrics['interaction_local_proportion'].values, label=f'{model}', color=colors[idx])

    for i in range(0, 2):
        ax[0, i].hlines(y=baseline_us, xmin=iteration_range[0], xmax=iteration_range[-1],
                        colors='black', linestyles='--', label='Dataset US Interactions', alpha=0.5)
        ax[1, i].hlines(y=baseline_local, xmin=iteration_range[0], xmax=iteration_range[-1],
                        colors='black', linestyles='--', label='Dataset Local Interactions', alpha=0.5)

    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].legend(loc='upper right', fontsize=legend_font_size)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.savefig(os.path.join(save_folder, f'{control_country}_proportions.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_country_pop_jsd_global(experiment_data, save_folder, iteration_range, colors, sorted_models_on_user_local_prop):
    """Plotting country JSD for n-th Interaction History vs. Original Interaction History and Popularity JSD n-th Interaction History vs. Original Interaction History side by side"""
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    label_font_size = 28
    legend_font_size = 24
    tick_font_size = 28

    ylim = 0.355
    def format_ax(ax, xlabel, ylabel, ylim):
        ax.set_xlim(iteration_range[0], iteration_range[-1])
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)

    format_ax(ax[0], 'Iteration', 'Country miscalibration (JSD)', ylim)
    format_ax(ax[1], 'Iteration', 'Popularity miscalibration (JSD)', ylim)

    model_color_map = {model: colors[idx] for idx, (model, choice_model) in enumerate(experiment_data.keys())}

    for idx, ((model, choice_model), entry) in enumerate(experiment_data.items()):
        metrics = entry['metrics']
        ax[0].plot(iteration_range, metrics[metrics['country'] == 'global']['interaction_jsd_summarized'].values, label=f'{model}', color=colors[idx], linewidth=4)
        ax[1].plot(iteration_range, metrics[metrics['country'] == 'global']['interaction_bin_jsd'].values, label=f'{model}', color=colors[idx], linewidth=4)

    legend_handles = [plt.Line2D([], [], color=model_color_map[model], marker='o', markersize=5, label=model, linestyle='None') for model in sorted_models_on_user_local_prop]

    ax[0].legend(handles=legend_handles, loc='upper left', fontsize=legend_font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'country_and_popularity_jsd_global.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_jsds(experiment_data, save_folder, iteration_range, control_country, colors):
    """2x2 layout plotting country and popularity JSDs for recommendations and interaction history"""
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    # fig.tight_layout()

    # plt.suptitle(f'Artist Country and Popularity JSD Divergences from {control_country} users', size=22)
    # fig.subplots_adjust(top=0.95)

    label_font_size = 22
    legend_font_size = 17
    tick_font_size = 22

    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].set_xlim(iteration_range[0], iteration_range[-1])
            ax[i, j].set_xlabel('Iteration', fontsize=label_font_size)
            ax[i, j].set_ylim(0, 1)
            ax[i, j].tick_params(axis='both', labelsize=tick_font_size)
            if i == 0:
                ax[i, j].set_ylabel('Country JSD (US/Local/Other)', fontsize=label_font_size)
            else:
                ax[i, j].set_ylabel('Popularity JSD (HighPop/MidPop/LowPop)', fontsize=label_font_size)
            ax[i, j].grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    ax[0, 0].set_title('Country JSD \nn-th Recommendations vs. \n Original Interaction History', fontsize=label_font_size)
    ax[0, 1].set_title('Country JSD \nn-th Interaction History vs. \n Original Interaction History', fontsize=label_font_size)
    ax[1, 0].set_title('Popularity JSD \nn-th Recommendations vs. \n Original Interaction History', fontsize=label_font_size)
    ax[1, 1].set_title('Popularity JSD \nn-th Interaction History vs. \n Original Interaction History', fontsize=label_font_size)

    for idx, ((model, choice_model), entry) in enumerate(experiment_data.items()):
        metrics = entry['metrics']
        metrics = metrics[metrics['country'] == control_country]
        ax[0, 0].plot(iteration_range, metrics['jsd_summarized'].values, label=f'{model}', color=colors[idx])
        ax[0, 1].plot(iteration_range, metrics['interaction_jsd_summarized'].values, label=f'{model}', color=colors[idx])
        ax[1, 0].plot(iteration_range, metrics['bin_jsd'].values, label=f'{model}', color=colors[idx])
        ax[1, 1].plot(iteration_range, metrics['interaction_bin_jsd'].values, label=f'{model}', color=colors[idx])

    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].legend(loc='upper right', fontsize=legend_font_size)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.savefig(os.path.join(save_folder, f'{control_country}_jsd.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_final_interactions(experiment_data, save_folder, iteration_range, countries):
    """ISMIR-like plot showing US/local proportions in initial and final user profiles"""
    countries = list(reversed(countries))
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    # plt.suptitle(f'Proportion of Interactions at Iteration 1 vs. Iteration 100', size=20)

    label_font_size = 13.5
    legend_font_size = 13.5
    tick_font_size = 13.5
    models = [model[0] for model in experiment_data.keys()]

    for idx, model in enumerate(models):
        baselines_us = []
        baselines_local = []
        final_us = []
        final_local = []

        metrics = experiment_data[(model, 'Rank Based')]["metrics"]
        # Reversed because the plotting is from bottom to top
        for country in countries:
            first_entry = metrics.query(f"country == '{country}' and iteration == {iteration_range[0]}")
            baselines_local.append(first_entry['interaction_local_proportion'].values[0])
            baselines_us.append(first_entry['interaction_us_proportion'].values[0] if country != 'US' else 0)

            metrics_entry = metrics.query(f"country == '{country}' and iteration == {iteration_range[-1]}")
            final_local.append(metrics_entry['interaction_local_proportion'].values[0])
            final_us.append(metrics_entry['interaction_us_proportion'].values[0] if country != 'US' else 0)

        i = idx // 3
        j = idx % 3

        ax[i, j].set_xlim(0, 1)
        ax[i, j].set_xlabel('Proportions of Interactions', fontsize=label_font_size)
        ax[i, j].set_title(model, fontsize=label_font_size)
        ax[i, j].barh(y=countries, width=final_us, left=0, color='#fe9f8e', label='Final US music', alpha=0.8)
        ax[i, j].barh(y=countries, width=final_local, left=np.ones(len(final_local)) - np.array(final_local),
                      color='#2784ba', label='Final domestic', alpha=0.8)

        ax[i, j].barh(y=countries, width=baselines_us, left=0, color='None', edgecolor='#a03112',
                      label='Input US music', linewidth=1.5)
        ax[i, j].barh(y=countries, width=baselines_local,
                      left=np.ones(len(baselines_local)) - np.array(baselines_local), color='None', edgecolor='#2784ba',
                      linewidth=1.5, label='Input domestic')
        ax[i, j].tick_params(axis='both', labelsize=tick_font_size)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', fontsize=legend_font_size)
    fig.tight_layout(rect=[0, 0, 1, 0.925])

    fig.savefig(os.path.join(save_folder, f'final_results.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_user_local_proportions(user_experiment_data, save_folder, colors, models, sorted_models_on_user_local_prop, local_baseline: float):
    """Regression plot of local proportions (globally across all users) over iterations for each model."""
    fig, ax = plt.subplots(figsize=(12, 8))

    label_font_size = 25
    legend_font_size = 18
    tick_font_size = 26

    for idx, model in enumerate(tqdm(models, desc='Creating Regression plots categorized by model using per-user metrics')):
        metrics = user_experiment_data[(model, 'Rank Based')]["metrics"]
        if not metrics.empty:
            sns.regplot(ax=ax, data=metrics, x="iteration", y="local_proportion", x_estimator=np.mean,
                        order=2, line_kws={'color': colors[idx]}, scatter_kws={'alpha': 0.2, 'color': colors[idx]}, x_ci=None, ci=None)

    ax.set_ylim(-0.01, 0.225)
    ax.set_ylabel('Proportion of local items recommended', fontsize=label_font_size)
    ax.set_xlabel('Iteration', fontsize=label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)

    # Update hlines for the baseline to have longer dashes and shorter gaps
    ax.hlines(y=local_baseline, xmin=0, xmax=100,
              colors='black', linestyles=(0, (10, 5)), linewidth=2, label='Original Dataset', alpha=0.5)

    legend_handles = [plt.Line2D([], [], color=colors[models.index(model)], marker='o', markersize=5, label=model, linestyle='None') for model in sorted_models_on_user_local_prop]
    legend_handles.append(
        plt.Line2D([], [], color='black', linestyle=(0, (10, 5)), linewidth=2, label='Baseline', alpha=1.)
    )

    ax.legend(handles=legend_handles, loc='lower right', fontsize=legend_font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'user_based_local_proportions.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_country_specific_local_proportions(user_experiment_data, save_folder, countries_of_interest_local_prop, colors, models):
    """2x3 layout of regression plots showing one model per plot and multiple countries of interest."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Define colors for each country to ensure they are consistent across all models
    country_colors = {country: color for country, color in zip(countries_of_interest_local_prop, colors)}

    label_font_size = 17
    legend_font_size = 17
    tick_font_size = 17

    for idx, model in enumerate(tqdm(models, desc='Creating Local proportions plots categorized by model and country using per-user metrics')):
        ax = axes[idx]

        data = user_experiment_data[(model, 'Rank Based')]["metrics"]
        data = data[data['country'].isin(countries_of_interest_local_prop)]

        legend_handles = []
        for country in tqdm(countries_of_interest_local_prop, desc='Countries'):
            country_data = data[data['country'] == country]
            sns.regplot(ax=ax, data=country_data, x="iteration", y="local_proportion", x_estimator=np.mean,
                        order=2, line_kws={'color': country_colors[country]},
                        scatter_kws={'alpha': 0.2, 'color': country_colors[country]}, label=f"{country}",
                        x_ci=None, ci=None)
            legend_handles.append(mlines.Line2D([], [], color=country_colors[country], label=country))

        ax.set_title(f'{model}', fontsize=label_font_size)
        ax.set_ylim(-0.05, 0.9)
        ax.set_ylabel('Local items recommended', fontsize=label_font_size)
        ax.set_xlabel('Iteration', fontsize=label_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.legend(handles=legend_handles, title="Country", loc='upper right', fontsize=legend_font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'user_based_local_proportions_country_specific.pdf'), format="pdf", bbox_inches="tight")
    plt.close()


@argh.arg('-ef', '--experiments-folder', type=str, help='Path to the experiments folder')
@argh.arg('-ex', '--experiment-name', type=str, help='Name of the specific experiment')
@argh.arg('-fc', '--focus-country', type=str, help='Focus country code in uppercase (e.g. US)')
@argh.arg('-f', '--force-recompute', action=argparse.BooleanOptionalAction, help='Force recompute the metrics')
@argh.arg('-c', '--control-country', type=str, help='Control country code for the plots in uppercase (e.g. DE)')
def plot_main(experiments_folder="experiments", experiment_name="sample1", focus_country="US", force_recompute=False,
              control_country=None):
    experiment_folder = os.path.join(experiments_folder, experiment_name)
    plot_save_folder = os.path.join(experiments_folder, experiment_name, 'plots')

    if force_recompute:
        # delete previous metrics file
        if os.path.exists(os.path.join(experiment_folder, 'metrics.csv')):
            os.remove(os.path.join(experiment_folder, 'metrics.csv'))
        if os.path.exists(os.path.join(experiment_folder, 'user_based_metrics.csv')):
            os.remove(os.path.join(experiment_folder, 'user_based_metrics.csv'))
        if os.path.exists(os.path.join(experiment_folder, 'baselines.csv')):
            os.remove(os.path.join(experiment_folder, 'baselines.csv'))

    if not os.path.exists(plot_save_folder):
        os.makedirs(plot_save_folder)

    print("Loading data...")

    # Load the data
    experiment_data, user_experiment_data = load_data(experiments_folder, experiment_name, focus_country, control_country)
    iterations_range = list(range(1, max(map(lambda x: max(x["metrics"]["iteration"]), experiment_data.values())) + 1))

    # countries of interest used in the ISMIR paper (all countries >100 users)
    countries_of_interest_ismir_style = ['CA', 'AU', 'UK', 'BR', 'IT', 'NO', 'NL', 'MX', 'ES', 'JP', 'SE', 'BE', 'FR', 'UA', 'DE', 'PL', 'RU', 'TR', 'FI', 'US']

    # countries to plot in the ISMIR-like plot
    # countries_of_interest_ismir_style = ['global', 'BR', 'UK', 'FI', 'DE', 'JP', 'US']
    # countries to plot in the regression line plots
    countries_of_interest_local_prop = ['BR', 'UK', 'FI', 'DE', 'JP', 'US']

    # Sorting the data for consistent plotting
    experiment_data = {k: experiment_data[k] for k in sorted(experiment_data.keys(), key=lambda x: x[0])}
    user_experiment_data = {k: user_experiment_data[k] for k in sorted(user_experiment_data.keys(), key=lambda x: x[0])}

    models = [model[0] for model in experiment_data.keys()]
    colors_models = plt.get_cmap('tab10')(np.arange(0, len(models)))

    initial_y_values = {}

    for idx, model in enumerate(tqdm(models, desc='Creating Regression plots categorized by model using per-user metrics')):
        metrics = user_experiment_data[(model, 'Rank Based')]["metrics"]
        if not metrics.empty:
            initial_y_values[model] = metrics[metrics['iteration'] == 1].local_proportion.mean()

    sorted_models_on_user_local_prop = sorted(models, key=lambda current_model: initial_y_values[current_model], reverse=True)

    if control_country:
        countries_of_interest_ismir_style.append(control_country)
        countries_of_interest_local_prop.append(control_country)

    colors_countries = plt.get_cmap('tab10')(np.arange(0, max(len(countries_of_interest_ismir_style), len(countries_of_interest_local_prop))))

    print("Creating plots...")

    for country in tqdm(countries_of_interest_local_prop, desc='Creating plots per country'):
        plot_proportions(experiment_data, plot_save_folder, iterations_range, country, colors_models)
        plot_jsds(experiment_data, plot_save_folder, iterations_range, country, colors_models)

    plot_country_pop_jsd_global(experiment_data, plot_save_folder, iterations_range, colors_models, sorted_models_on_user_local_prop)

    plot_final_interactions(experiment_data, plot_save_folder, iterations_range, countries_of_interest_ismir_style)

    # TODO: this is a hotfix while we investigate what goes wrong with the calculation of baselines / initial interaction proportions
    user_interactions_base = user_experiment_data[('Pop', 'Rank Based')]['metrics']
    local_baseline = user_interactions_base[user_interactions_base['iteration'] == 1].interaction_local_proportion.mean()

    plot_user_local_proportions(user_experiment_data, plot_save_folder, colors_models, models, sorted_models_on_user_local_prop, local_baseline)
    plot_country_specific_local_proportions(user_experiment_data, plot_save_folder, countries_of_interest_local_prop, colors_countries, models)


if __name__ == "__main__":
    argh.dispatch_command(plot_main)
