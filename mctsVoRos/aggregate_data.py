
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid', palette='colorblind', font='serif', font_scale=1.5, rc=None)

files = [pd.read_csv('debug/'+f) for f in os.listdir('debug') if f.endswith('.csv')]
df = pd.concat(files)
del df['Unnamed: 0']
print(df)
df[['reachGoal', 'collision', 'maxSteps']] = df[['reachGoal', 'collision', 'maxSteps']].astype(int)
grouped = df.groupby('algorithm')
final = grouped.mean()

# compute mean and std of nSteps for each algorithm, only for the cases where the goal was reached
filtered_df = df[df['reachGoal'] == 1]
filtered_group = filtered_df.groupby('algorithm')
meanNsteps = filtered_group["nSteps"].mean()
stdNsteps = filtered_group["nSteps"].std()

final['std_nSteps'] = stdNsteps
final['nSteps'] = meanNsteps
print(final)
final.to_csv('final.csv')

trajectories = ['debug/'+f for f in os.listdir('debug') if f.endswith('.pkl') and 'acts' in f]

algorithms = {'VO-TREE': [], 'MCTS': [], 'VO-PLANNER': []}
for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                algorithms[k].append(data)
                
algorithms_metrics = {}
for k in algorithms.keys():
    metrics = {'smoothness':[]}
    print(f'{k} has {len(algorithms[k])} trajectories')
    for data in algorithms[k]:
        data = np.array(data)
        data = data[1:]
        x,y = data[:, 0] * np.cos(data[:, 1]), data[:, 0] * np.sin(data[:, 1])
        smooth = (np.sqrt(np.diff(x)**2 + np.diff(y)**2)/0.1).sum() * 1/(len(x)-1)
        # smooth = smooth / 0.4
        metrics['smoothness'].append(smooth)
    algorithms_metrics[k] = {
        'smoothness': np.mean(metrics['smoothness']),
        'std': np.std(metrics['smoothness'])
    }

df = pd.DataFrame(algorithms_metrics)
df = df.T.sort_values(by='smoothness', ascending=False)

df.to_csv('metrics.csv')
df.plot(kind='bar', y='smoothness', yerr='std')
plt.ylabel('Smoothness')
plt.legend().set_visible(False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('smoothness.png')