# Feedback Loop Simulation Framework for Recommender Systems
This repository hosts the code and the additional materials for the paper "Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems" by [Oleg Lesota](mailto:oleg.lesota@jku.at), [Jonas Geiger](mailto:jonasgeiger@outlook.de), [Max Walder](mailto:max@gstat.eu), Dominik Kowald and Markus Schedl (to be) published at RecSys 2024.

This framework allows to simulate closed feedback loops using the [RecBole](https://recbole.io) library. The repository includes multiple user choice models, an example dataset as well as evaluation and plotting utilities. Please refer to the paper for more. The data and models used in the paper are available on demand via an email to [Oleg Lesota](mailto:oleg.lesota@jku.at).

## Installation
Should you wish to use CUDA, install it separately according to your setup.

Ensure your current Python environment has all necessary packages installed:
```pip3 install -r requirements.txt```

## Folder structure

To run an experiment you need to add your initial dataset into the `experiments` folder. Each subfolder is considered
its own experiment and requires an input folder with 3 files:

```
experiments/
  <experiment name>/
    input/
      dataset.inter      # Interactions
      demographics.tsv   # User demographic information
      tracks.tsv         # Music Track information
```
### Sample Dataset:
#### demographics.tsv (no header):

| Column Order | Column name         | Description            |
|--------------|---------------------|------------------------|
| 1            | user_country        | Country Code of User   |
| 2            | age_at_registration | Age at registration    |
| 3            | gender              | m/f                    |
| 4            | registration_time   | Registration timestamp |

Example:

```
RU	27	m	2007-03-27 19:50:20
IT	33	m	2006-06-18 21:07:33
BR	19	m	2010-01-14 05:55:11
RU	25	m	2007-10-12 18:42:00
UK	25	m	2005-06-15 22:02:11
ES	29	m	2005-09-30 22:38:33
```

#### tracks.tsv (no header):

| Column Order | Column name  | Description             |
|--------------|--------------|-------------------------|
| 1            | artist       | Name of the artist      |
| 2            | track        | Name of the music track |
| 3            | user_country | Country Code of track   |

Example:

```
Modena City Ramblers	In Un Giorno Di Pioggia	IT
AC/DC   Thunderstruck   AU
Taylor Swift	King of My Heart	US
```

#### dataset.inter (RecBole standard header):

| Column Order | Column name   | Description                                                  |
|--------------|---------------|--------------------------------------------------------------|
| 1            | user_id:token | ID (row number, starting at 0) of user in demographics table |
| 2            | item_id:token | ID (row number, starting at 0) of track in tracks table      |

A valid example setup can be found under `experiments/example`

## Usage
### Full Simulation Run
`python3 run_loop.py -n 10 --dataset example --model ItemKNN --choice-model rank_based --config recbole_config_default.yml`

| Option         | Description                                                                                                                                                                     |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -n             | Amount of loops to run                                                                                                                                                          |
| --dataset      | Experiment name (subfolder name) with dataset to evaluate                                                                                                                       |
| --model        | `Recbole`-model to train and generate recommendations from                                                                                                                      |
| --choice-model | User choice model to simulate acceptance with. Available options are <br/> `random`, `rank_based`, `us_centric`, `non_us_centric`<br/>and are implemented in `choice_models.py` |
| --config       | Recbole config file to be used                                                                                                                                                  |

This script internally calls the `main.py` script `n` times as a subprocess due to memory leaks within Recbole encountered during development.

### Single simulation step
To run a single iteration of the tool use the `main.py` script. All options are detailed by the `main.py -h` command.

### Output
After finishing the full experiment, you'll be left with the following folder structure:
```
experiments/
    <experiment name>/
        datasets/
            iteration_1.inter     # input data at interation 1
            iteration_2.inter
            ...
            iteration_<n>.inter   # input data at interation n
            iteration_<n+1>.inter # final dataset with all new artificial interactions added
        input/
            dataset.inter         # Interactions, unmodified
            demographics.tsv      # User demographic information, unmodified
            tracks.tsv            # Music Track information, unmodified
        log/
            ...
        log_tensorboard/
            ...
        output/
            iteration_1_accepted_songs.tsv      # Accepted songs of iteration 1
            iteration_1_top_k.tsv               # Suggested recommendations of iteration 1
            iteration_2_accepted_songs.tsv
            iteration_2_top_k.tsv
            ...
            iteration_<n>_accepted_songs.tsv
            iteration_<n>_top_k.tsv
        params.json              # Internal file with saved simulation parameters (model, choice model, config)
```

#### Datasets
The dataset folder contains the data that was used to train Recbole in a given iteration. 
**This also means that iteration_1.inter is always identical to input/dataset.inter**.
The last file in this folder was not used for training, but instead contains the resulting dataset after the
experiment finished.

#### Output
For each iteration, there are two resulting output files

**iteration_\<n\>_top_k.tsv**: Tab-Separated Tab-Separated Values file (TSV) with the following columns (with header row) detailing the best k (default k=10) recommendations per user

| Column Order | Column name | Description                                                                                     |
|--------------|-------------|-------------------------------------------------------------------------------------------------|
| 1            | user_id     | ID (row number, starting at 0) of user in demographics table                                  |
| 2            | track_id    | ID (row number, starting at 0) of track in tracks table                                         |
| 3            | rank        | rank of the recommendation. Best recommendation is always in rank 1, second best in rank 2, etc. |
| 4            | score       | Score given to the recommendation by Recbole                                                    |

This table records what the model trained by recbole has determined to be the best k items to recommend to each user.
Note that repeated recommendation is not allowed and such recommendations are filtered out beforehand.

**iteration_\<n\>_accepted.tsv**: Tab-Separated Tab-Separated Values file (TSV) with identical file structure as `dataset.inter` (see above).

This table records the subset of the top_k table that was selected and "consumed" by the user choice model. Its content
is appended to the dataset to be used in the next simulation step.

### Evaluation
To generate the metrics used in the research paper, set your experiment(s) in the `compute_all_metrics.py` script 
and run it as a result, 3 new files will appear in the experiment folder

#### baselines.csv: Comma-separated values file containing the following columns
| Column Order | Column name | Description                                                                                             |
|--------------|-------------|---------------------------------------------------------------------------------------------------------|
| 1            | country     | Country Code of users                                                                                   |
| 2            | us          | Percentage of interactions of users from `country` with music tracks originating from the US            |
| 3            | local       | Percentage of interactions of users from `country` with music tracks originating from their own country |
| 4            | other       | Percentage of interactions of users from `country` neither coming from the US nor their own country     |

#### user_based_metrics.csv Comma-separated values file containing the following columns
This table contains information about the recommendations of each user at each simulation step

| Column name                  | Description                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------|
| user_id                      | ID (row number, starting at 0) of user in demographics table                              |
| model                        | Recbole model used to generate recommendations                                            |
| choice_model                 | Choice model used to simulate user behavior                                               |
| iteration                    | Simulation iteration                                                                      |
| country                      | Country Code of user                                                                      |
| user_count                   | Amount of users originating from this country                                             |
| jsd                          | Artist country JSD divergence - original input file vs. recommendations at this iteration |
| interaction_jsd              | Artist country JSD divergence - original input file vs. input dataset at this iteration   |
| jsd_sumamrized               | like `jsd`, but countries are grouped into US/Local/Other                                 |
| interaction_jsd_sumamrized   | like `interaction_jsd`, but countries are grouped into US/Local/Other                     |
| bin_jsd                      | Popularity bin JSD divergence - original input file vs. recommendations at this iteration |
| interaction_bin_jsd          | Popularity bin JSD divergence - original input file vs. input dataset at this iteration   |
| us_proportion                | Percentage at this iteration of recommendations for tracks originating in US              |
| us_interaction_proportion    | Percentage of input data at this iteration for tracks originating in US                   |
| local_proportion             | Percentage of recommendations for tracks originating in user country                      |
| local_interaction_proportion | Percentage of input data at this iteration for tracks originating in user country         |

#### metrics.csv
The metrics file contains the same information as `user_based_metrics`, but grouped by user country and iteration.
It is useful to compare the RecSys behaviour between different demographic groups.

### Utilities
#### merge_all_metrics.py
Edit the list of experiments in the script to obtain a merged version of all the different metric files. 
The merged metric file was used to generate the figures in the paper.

#### plots.py
Generation of various plots about a dataset. If you want to recreate the figures used in the paper displaying
multiple experiments at once, create a mock experiment folder, insert the common input data and manually copy the merged
metrics files into the experiment folder.

See `python3 plots.py -h` for more details.

#### dataset_statistics.py
Utility script used to get various dataset information.

------
Based on preliminary work by Sebastian Wolff.
