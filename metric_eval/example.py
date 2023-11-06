import metric_eval
import pandas as pd
import numpy as np

data = pd.read_csv("data/metric_scores_long.csv")
data_2nd_run = pd.read_csv("data/metric_scores_long_2nd_run.csv")

# Example usage
#data = pd.read_csv(uploaded)

# normalize the score columns
data.iloc[:, 3:] = data.iloc[:, 3:].apply(metric_eval.normalize2, axis=0)

# metric stability
rel_cor = metric_eval.metric_stability(data,data_2nd_run)
print(rel_cor)
# metric consistency
alphas, sems = metric_eval.metric_consistency(data)
print(alphas)
print(sems)

# MTMM: metric convergence and discriminant validity
metric_names = data.columns[2:14].tolist()
trait = ['COH', 'CON', 'FLU', 'REL'] * 3
method = ['Expert_1'] * 4 + ['Expert_2'] * 4 + ['Expert_3'] * 4

# Create the MTMM_design DataFrame
MTMM_design = pd.DataFrame({
    'trait': trait,
    'method': method,
    'metric': metric_names
})

# print(MTMM_design)

MTMM_result = metric_eval.MTMM(data, MTMM_design, method = 'pearson')
print(MTMM_result)

# Concurrent Validty Table
criterion = ['Expert.1.COH','Expert.1.CON','Expert.1.FLU','Expert.1.REL']
concurrent_val_table = metric_eval.concurrent_validity(data, criterion)
metric_eval.print_concurrent_validity_table(concurrent_val_table)

# metric consistency by N
# This may take a while to run
Ns = np.arange(10, 110, 10)
metric_names = data.columns[2:].tolist()

metric_consistency_byN = pd.DataFrame(index=[f'N{N}' for N in Ns], columns=metric_names)
for N in Ns:
    alphas, _ = metric_eval.metric_consistency(data, N=N)
    metric_consistency_byN.loc[f'N{N}'] = [alphas.get(metric, np.nan) for metric in metric_names]

print(metric_consistency_byN.round(2))