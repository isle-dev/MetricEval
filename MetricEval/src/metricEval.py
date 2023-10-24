
import numpy as np
from scipy.stats import norm
import pandas as pd
from itertools import combinations
from scipy.stats import kendalltau

class MetricEval:

    def normalize2(x):
        # Calculate the mean for elements in x that are <= y for each y in x
        means = np.array([np.mean(x <= y) for y in x])
        # Scale the mean by 0.998 and add 0.001
        scaled_means = 0.001 + 0.998 * means
        # Apply the inverse of the standard normal cumulative distribution function (CDF)
        qnorm_values = norm.ppf(scaled_means)
        # Standardize the resulting values
        standardized_values = (qnorm_values - np.mean(qnorm_values)) / np.std(qnorm_values)
        return standardized_values


    def coeff_alpha_miss(responses):
        # Number of items (columns)
        n = responses.shape[1]

        # Initialize covariance matrix
        C = np.zeros((n, n))

        # Compute the covariance matrix using pairwise deletion for missing values
        for i in range(n):
            for j in range(n):
                x = responses[:, i]
                y = responses[:, j]

                # Pairwise deletion of missing values
                mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[mask]
                y = y[mask]

                C[i, j] = np.cov(x, y)[0, 1]

        # Compute Cronbach's alpha
        tr_C = np.trace(C)
        sum_C = np.sum(C)
        alpha = (1 - tr_C / sum_C) * (n / (n - 1))

        return alpha

    # Function to compute metric stability

    # data1: pandas.DataFrame containing the following columns:
    # - test_id: array containing ids of test examples
    # - model_id: character array containing ids of models
    # - additional columns containing scores on each metric (metric name as column name)

    # data2: pandas.DataFrame containing the same structure as data1 but with data from a second run.

    # The function compares the average metric scores for models between the two data sets (data1 and data2).
    # It calculates the Pearson correlation coefficient between the average metric scores from the first run (data1)
    # and the second run (data2) for each metric.

    # Returns a dictionary containing metric stability estimates (correlation coefficients) for each metric.
    def metric_stability(data1, data2):
        def get_dataframes_for_metric(data1, data2, metric):
            try:
                df1 = data1[['test_id', 'model_id', metric]].pivot(index='test_id', columns='model_id', values=metric)
                df2 = data2[['test_id', 'model_id', metric]].pivot(index='test_id', columns='model_id', values=metric)
                return df1, df2
            except KeyError:
                print(f"Cannot find metric, {metric} in the second run.")
                return None, None

        def get_mean_scores_by_model(df1, df2):
            first_run = [df1[model_id].mean() for model_id in df1.columns]
            second_run = [df2[model_id].mean() if model_id in df2.columns else print(f"Cannot find model, {model_id} in the second run.") for model_id in df1.columns]
            return first_run, second_run

        metric_scores = data1.drop(columns=['model_id', 'test_id'])
        corrcoefs = {}

        for metric in metric_scores.columns:
            df1, df2 = get_dataframes_for_metric(data1, data2, metric)
            if df1 is None or df2 is None:
                continue

            if not df1.index.equals(df2.index):
                print('Mismatch Index')
                continue

            first_run, second_run = get_mean_scores_by_model(df1, df2)
            corrcoefs[metric] = np.corrcoef(first_run, second_run)[0][1]

        return corrcoefs

    # Function to compute metric consistency
    # data: data.frame containing the following columns:
        # test_id: array containing ids to test examples
        # model_id: character array containing ids of models
        # additional columns containing scores on each metric (metric name as column name)
    # N: optional; size of subset of test set examples for computing average metric score.
    # By default the total number of test set examples will be used.
    # Returns metric consistency estimates (alphas) and the standard error of measurement
    # (sems) of each metric in the columns.
    def metric_consistency(data, N=-1):
        metric_scores = data.drop(columns=['test_id', 'model_id'])
        n_metrics = metric_scores.shape[1]
        alphas = {}
        sems = {}

        for metric in metric_scores.columns:
            df = data[['test_id', 'model_id', metric]]
            responses = df.pivot(index='model_id', columns='test_id', values=metric).to_numpy()
            n_models = responses.shape[0]

            if N == -1:
                N = responses.shape[1]

            combs = [np.sort(np.random.choice(responses.shape[1], N, replace=False)) for _ in range(200)]
            combs = np.unique(combs, axis=0)

            alpha_reps = []
            sem_reps = []

            for comb in combs:
                alpha_rep = MetricEval.coeff_alpha_miss(responses[:, comb])
                sem_rep = np.nanstd(np.nanmean(responses[:, comb], axis=1)) * np.sqrt(1 - alpha_rep)
                alpha_reps.append(alpha_rep)
                sem_reps.append(sem_rep)

            alphas[metric] = np.mean(alpha_reps)
            sems[metric] = np.mean(sem_reps)

        return alphas, sems

    # Function for constructing MTMM table.
    # Arguments:
    # data: data.frame containing the following columns:
        # test_id: array containing ids to test examples
        # model_id: character array containing ids of models
        # additional columns containing scores on each metric (metric name as column name)
    # design: table containing 3 columns:
        # trait: trait name
        # method: method name
        # metric: metric name, should correspond to score columns in data matrix.
    # method: correlation type. Default is kendall.
    # Output: An MTMM table. Entries wrapped in *.* provide checks for convergent validity
    # (same trait, different methods). Entries wrapped in (.) provide checsk for
    # discriminant validity (different traits, same method)
    def MTMM(data, design, method='kendall'):
        design = design.sort_values(['trait', 'method']).reset_index(drop=True)

        MTMM_data = data[['model_id', 'test_id'] + list(design['metric'])]
        MTMM_df = MTMM_data.groupby('model_id').mean(numeric_only=True)
        MTMM_cor = MTMM_df.corr(method=method).round(2).astype(str)

        MTMM_cor.values[np.tril_indices_from(MTMM_cor, -1)] = '-'
        alphas, _ = MetricEval.metric_consistency(MTMM_data)
        np.fill_diagonal(MTMM_cor.values, [str(round(alphas.get(metric, np.nan), 2)) for metric in MTMM_cor.columns])

        for i in range(len(design)):
            for j in range(i, len(design)):
                if design.loc[i, 'trait'] == design.loc[j, 'trait'] and design.loc[i, 'method'] != design.loc[j, 'method']:
                    MTMM_cor.iloc[i, j] = f"*{MTMM_cor.iloc[i, j]}*"
                if design.loc[i, 'method'] == design.loc[j, 'method'] and design.loc[i, 'trait'] != design.loc[j, 'trait']:
                    MTMM_cor.iloc[i, j] = f"({MTMM_cor.iloc[i, j]})"

        # Update design DataFrame to set repeated traits to empty strings
        design['trait'] = design['trait'].where(design['trait'] != design['trait'].shift(), '')

        MTMM_print = pd.DataFrame(np.full((len(design) + 2, len(design) + 2), np.nan).astype(str))
        MTMM_print.iloc[2:, 2:] = MTMM_cor.values
        MTMM_print.iloc[:2, 2:] = design[['trait', 'method']].T.values.astype(str)
        MTMM_print.iloc[2:, :2] = design[['trait', 'method']].values.astype(str)
        MTMM_print.iloc[:2, :2] = ''

        return MTMM_print

    # Function to compute concurrent validity

    # data: pandas.DataFrame containing the following columns:
    # - test_id: array containing ids of test examples
    # - model_id: character array containing ids of models
    # - additional columns containing scores on each metric (metric name as column name)

    # criterion: list of criterion variables for which concurrent validity will be computed.

    # The function computes the concurrent validity for each criterion variable by calculating
    # the Kendall's Tau correlation coefficient between the criterion variable and each metric.

    # Returns a transposed pandas DataFrame with criterion variables as row indices and metrics as columns,
    # containing the computed Kendall's Tau correlation coefficients for each metric-criterion pair.
    def concurrent_validity(data, criterion):
        def get_dataframes_for_metric(data, metric):
            try:
                return data[['test_id', 'model_id', metric]].pivot(index='test_id', columns='model_id', values=metric)
            except KeyError:
                print(f"Cannot find metric, {metric} in the data.")
                return None
            
        corrcoefs = {}

        for criteria in criterion:
            df_criteria = get_dataframes_for_metric(data, criteria)
            criteria_responses = [df_criteria[model_id].mean() for model_id in df_criteria.columns]

            corrcoefs[criteria] = {}
            for metric in data.drop(columns=['model_id', 'test_id']).columns:
                df_metric = get_dataframes_for_metric(data, metric)
                metric_responses = [df_metric[model_id].mean() for model_id in df_metric.columns]
                corrcoefs[criteria][metric], _ = kendalltau(criteria_responses, metric_responses)

        return pd.DataFrame(corrcoefs)

    def print_concurrent_validity_table(data):
        print(data.T)


