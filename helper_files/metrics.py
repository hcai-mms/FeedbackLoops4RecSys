from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def calculate_baselines(global_interactions, tracks_info, demographics, focus_country):
    """
    Calculate the global and country-specific baseline proportions for 'focus_country', 'local', and 'other' interactions.

    Parameters:
    - global_interactions: DataFrame containing global interaction history.
    - tracks_info: DataFrame containing tracks and their country of origin.
    - demographics: DataFrame containing user demographics including their country.

    Returns:
    - Saves the result to 'baselines.csv' and returns the DataFrame.
    """
    # Ensure user_id is an explicit column in demographics
    demographics = demographics.reset_index().rename(columns={'index': 'user_id'})

    # Merge interactions with track information
    merged_data = global_interactions.merge(tracks_info[['item_id', 'country']], on='item_id', how='left')
    # Merge demographics to get user's country
    merged_data = merged_data.merge(demographics[['user_id', 'country']], on='user_id', how='left', suffixes=('', '_user'))

    # Calculate proportions of 'focus_country', 'local', and 'other'
    merged_data[f'is_{focus_country}'] = merged_data['country'] == focus_country
    merged_data['is_local'] = merged_data['country'] == merged_data['country_user']
    merged_data['is_other'] = ~(merged_data[f'is_{focus_country}'] | merged_data['is_local'])

    # Aggregate these proportions at the country level
    baseline_data = merged_data.groupby('country_user').agg({
        f'is_{focus_country}': 'mean',
        'is_local': 'mean',
        'is_other': 'mean'
    }).reset_index().rename(columns={
        'country_user': 'country',
        f'is_{focus_country}': f'{focus_country.lower()}',
        'is_local': 'local',
        'is_other': 'other'
    })

    # Calculate global proportions for all users
    global_proportions = {
        'country': 'global',
        f'{focus_country.lower()}': merged_data[f'is_{focus_country}'].mean(),
        'local': merged_data['is_local'].mean(),
        'other': merged_data['is_other'].mean()
    }

    # Append the global proportions as the first row in the DataFrame
    global_row = pd.DataFrame([global_proportions])
    baseline_data = pd.concat([global_row, baseline_data], ignore_index=True)

    return baseline_data


def calculate_proportions(top_k_data, tracks_info, demographics, focus_country):
    """
    Calculate the proportions of focus_country track recommendations for each user and optionally aggregate by country.

    Parameters:
    - top_k_data: DataFrame containing top k interactions.
    - tracks_info: DataFrame containing tracks information.
    - demographics: DataFrame containing user demographics.

    Returns:
    Tuple of DataFrames with focus_country and local proportion values for each user and aggregated by country.
    """
    # Prepare the data by merging necessary information
    top_k_data = top_k_data.merge(demographics[['country']], left_on='user_id', right_index=True, how='left').rename(columns={'country': 'user_country'})
    top_k_data = top_k_data.merge(tracks_info[['item_id', 'country']], on='item_id', how='left').rename(columns={'country': 'artist_country'})

    # Calculate user-level proportions
    user_proportions = top_k_data.groupby('user_id').apply(
        lambda df: pd.Series({
            f'{focus_country.lower()}_proportion': (df['artist_country'] == focus_country).mean(),
            'local_proportion': (df['user_country'] == df['artist_country']).mean()
        })
    ).reset_index()

    # Aggregate by country
    country_proportions = top_k_data.groupby('user_country').apply(
        lambda df: pd.Series({
            f'{focus_country.lower()}_proportion': (df['artist_country'] == focus_country).mean(),
            'local_proportion': (df['artist_country'] == df['user_country']).mean()
        })
    ).reset_index().rename(columns={'user_country': 'country'})

    # Add global proportions
    global_proportions = pd.Series({
        'country': 'global',
        f'{focus_country.lower()}_proportion': (top_k_data['artist_country'] == focus_country).mean(),
        'local_proportion': (top_k_data['user_country'] == top_k_data['artist_country']).mean()
    })

    country_proportions = pd.concat([pd.DataFrame([global_proportions]), country_proportions], ignore_index=True)

    return user_proportions, country_proportions


def join_interaction_with_country(interaction_history, demographics, tracks_info, tracks_with_popularity):
    """
    Join interaction history with demographics to associate each interaction with a country.

    Parameters:
    - interaction_history: DataFrame of interaction histories without country information.
    - demographics: DataFrame of user demographics, including country.
    - tracks_info: DataFrame of tracks information, which is used to ensure item_id compatibility.

    Returns:
    DataFrame of interaction histories enriched with country information.
    """

    # Merge to get country and other demographic details for each interaction
    interaction_history = interaction_history.merge(demographics, left_on='user_id', right_index=True, how='left')
    interaction_history = interaction_history.rename(columns={'country': 'user_country'})

    # Merge to include track details
    interaction_history = interaction_history.merge(tracks_info, on='item_id', how='left')
    interaction_history = interaction_history.rename(columns={'country': 'artist_country'})

    # Merge to include popularity bins
    interaction_history = interaction_history.merge(tracks_with_popularity[['item_id', 'popularity_bin']], on='item_id', how='left')

    return interaction_history


def calculate_iteration_jsd_per_user(recs_merged, unique_item_countries, history_distribution, model, choice_model, iteration, user_ids, focus_country):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between history and recommendations for each user and aggregate by country if needed.

    Parameters:
    - recs_merged: DataFrame containing top K interactions.
    - unique_item_countries: List of unique item countries from the tracks information.
    - history_distribution: Distribution of user interaction history across countries.
    - model: The name of the model used in the experiment.
    - choice_model: The name of the choice model used in the experiment.
    - iteration: The iteration number.
    - user_ids: List of user IDs.

    Returns:
    Tuple of DataFrames with JSD values per user and aggregated by country.
    """
    jsd_rows_per_user = []

    recs_by_user = recs_merged.groupby('user_id')

    for user_id in user_ids:
        if user_id in recs_by_user.groups:
            user_recs = recs_by_user.get_group(user_id)
            user_country = user_recs['user_country'].values[0]

            top_k_distribution = calculate_country_distribution(user_recs, unique_item_countries)

            summarized_history = np.zeros(3)
            summarized_history[0] = history_distribution[user_id][unique_item_countries == focus_country].sum()
            summarized_history[1] = history_distribution[user_id][unique_item_countries == user_country].sum()
            if user_country == focus_country:
                # through floating point errors, it might be slightly negative which causes jsd to be infinite
                # to be safe max it with 0
                summarized_history[2] = np.max([1 - summarized_history[0], 0])
            else:
                summarized_history[2] = np.max([1 - summarized_history[0] - summarized_history[1], 0])

            summarized_top_k = np.zeros(3)
            summarized_top_k[0] = top_k_distribution[unique_item_countries == focus_country].sum()
            summarized_top_k[1] = top_k_distribution[unique_item_countries == user_country].sum()
            if user_country == focus_country:
                summarized_top_k[2] = np.max([1 - summarized_top_k[0], 0])
            else:
                summarized_top_k[2] = np.max([1 - summarized_top_k[0] - summarized_top_k[1], 0])

            jsd_rows_per_user.append({
                'user_id': user_id,
                'country': user_recs['user_country'].values[0],
                'jsd': jensenshannon(history_distribution[user_id], top_k_distribution, base=2),
                'jsd_summarized': jensenshannon(summarized_history, summarized_top_k, base=2),
                'user_count': 1
            })

    jsd_user_df = pd.DataFrame(jsd_rows_per_user)

    # Aggregating by country
    jsd_country_df = jsd_user_df.groupby('country').agg({
        'jsd': 'mean',
        'jsd_summarized': 'mean',
        'user_count': 'sum'
    }).reset_index()

    # Add global row for global statistics
    global_jsd = jsd_user_df['jsd'].mean()
    global_jsd_summarized = jsd_user_df['jsd_summarized'].mean()
    global_row = pd.Series({
        'country': 'global',
        'jsd': global_jsd,
        'jsd_summarized': global_jsd_summarized,
        'user_count': jsd_user_df['user_count'].sum()
    })
    jsd_country_df = pd.concat([pd.DataFrame([global_row]), jsd_country_df], ignore_index=True)

    # Add experiment details
    for df in (jsd_user_df, jsd_country_df):
        df['model'] = model
        df['choice_model'] = choice_model
        df['iteration'] = iteration

    jsd_user_df = jsd_user_df[['user_id', 'model', 'choice_model', 'iteration', 'country', 'user_count', 'jsd', 'jsd_summarized']]
    jsd_country_df = jsd_country_df[['model', 'choice_model', 'iteration', 'country', 'user_count', 'jsd', 'jsd_summarized']]

    return jsd_user_df, jsd_country_df


def calculate_country_distribution(df, country_list):
    """
    Calculates the distribution of tracks over countries.

    Parameters:
    - df: DataFrame with 'item_id'.
    - country_list: List of countries to ensure those not in the dataframe return 0.

    Returns:
    Numpy array representing the distribution of tracks across countries.
    """
    country_counter = Counter(df['artist_country'])
    country_counts = pd.Series(country_counter)
    distribution = country_counts.reindex(country_list, fill_value=0).values
    distribution = distribution / distribution.sum()
    return distribution


def create_popularity_bins(interactions_merged, tracks_info):
    """
    Create popularity bins for tracks based on interaction counts.

    Parameters:
    - interactions_merged: DataFrame containing global interaction history.
    - tracks_info: DataFrame containing tracks information.

    Returns:
    DataFrame with popularity bins for each track.
    """
    # Calculate popularity of each track
    popularity_counts = interactions_merged['item_id'].value_counts().rename('interaction_count')
    tracks_with_popularity = tracks_info.merge(popularity_counts, left_on='item_id', right_index=True, how='left')
    tracks_with_popularity['interaction_count'] = tracks_with_popularity['interaction_count'].fillna(0)

    # Calculate quantiles for bin thresholds
    quantiles = {
        0.2: 0,
        0.8: 0
    }
    total_popularity = tracks_with_popularity['interaction_count'].sum()
    # find lower threshold
    for i in range(1, tracks_with_popularity['interaction_count'].max()):
        sum_below_threshold = tracks_with_popularity[tracks_with_popularity['interaction_count'] <= i]['interaction_count'].sum()
        if sum_below_threshold / total_popularity >= 0.2:
            quantiles[0.2] = i
            break
    # find upper threshold
    for i in reversed(range(1, tracks_with_popularity['interaction_count'].max())):
        sum_below_threshold = tracks_with_popularity[tracks_with_popularity['interaction_count'] <= i]['interaction_count'].sum()
        if sum_below_threshold / total_popularity >= 0.8:
            quantiles[0.8] = i
            break

    # Assign bins
    tracks_with_popularity['popularity_bin'] = np.select(
        [
            tracks_with_popularity['interaction_count'] <= quantiles[0.2],
            tracks_with_popularity['interaction_count'] > quantiles[0.8]
        ],
        ['Low', 'High'], default='Medium'
    )

    return tracks_with_popularity


def calculate_user_bin_jsd(recs_merged, interactions_merged):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between history and recommendations for each user, based on popularity bins.

    Parameters:
    - recs_merged: DataFrame containing top K interactions.
    - interactions_merged: DataFrame containing global interaction history.

    Returns:
    DataFrame with the JSD values per user. If user_based is False, it aggregates the JSD by country.
    """
    # Get bin counts per user for recommendations and history
    recs_bin_counts = recs_merged.groupby('user_id')['popularity_bin'].value_counts(normalize=True).unstack(fill_value=0)
    hist_bin_counts = interactions_merged.groupby('user_id')['popularity_bin'].value_counts(normalize=True).unstack(fill_value=0)

    # Ensure all bins are represented
    all_bins = ['High', 'Medium', 'Low']
    recs_bin_counts = recs_bin_counts.reindex(columns=all_bins, fill_value=0)
    hist_bin_counts = hist_bin_counts.reindex(columns=all_bins, fill_value=0)

    # Calculate JSD for each user
    jsd_results = (recs_bin_counts.apply(lambda x: jensenshannon(x, hist_bin_counts.loc[x.name], base=2), axis=1)
                   .reset_index().rename(columns={0: 'jsd'}))

    jsd_results['country'] = recs_merged.groupby('user_id')['user_country'].first()

    return jsd_results


def aggregate_jsd_by_country(bin_jsd_df, user_based=False):
    """
    Aggregate JSD by country, computing mean JSD for each country and globally, unless user_based is True.

    Parameters:
    - bin_jsd_df: DataFrame containing JSD values per user and country.
    - user_based: Boolean flag to determine if aggregation should be skipped.

    Returns:
    DataFrame with aggregated JSD values per country or the original DataFrame if user_based is True.
    """
    if user_based:
        return bin_jsd_df

    aggregated_jsd = bin_jsd_df.groupby('country')['jsd'].mean().reset_index()
    aggregated_jsd.columns = ['country', 'bin_jsd']
    global_jsd = bin_jsd_df['jsd'].mean()
    global_row = pd.DataFrame([['global', global_jsd]], columns=['country', 'bin_jsd'])
    aggregated_jsd = pd.concat([global_row, aggregated_jsd], ignore_index=True)

    return aggregated_jsd


def merge_jsd_dataframes(jsd_df, user_based_jsd_df, bin_jsd_df):
    """
    Merge JSD dataframes to include bin JSD values. Can operate at user or country level.

    Parameters:
    - jsd_df: DataFrame containing JSD values per user and country.
    - user_based_jsd_df: DataFrame containing JSD values per user.
    - bin_jsd_df: DataFrame containing JSD values per user and country, including bin JSD values.

    Returns:
    Tuple of DataFrames with bin JSD values merged into the original JSD dataframes.
    """

    user_based_jsd_df = user_based_jsd_df.merge(bin_jsd_df['jsd'].rename('bin_jsd'), left_index=True, right_index=True, how='left')

    aggregated_bin_jsd = aggregate_jsd_by_country(bin_jsd_df)
    jsd_df = jsd_df.merge(aggregated_bin_jsd, on='country', how='left')

    return user_based_jsd_df, jsd_df
