import itertools
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_PATH = 'debug/int_data/'
sns.set_theme(context='paper', style='whitegrid', palette='colorblind', font='serif', font_scale=1.5, rc=None)
colors = {
    'VO-PLANNER': sns.color_palette('colorblind')[2],
    'MCTS-VO': sns.color_palette('colorblind')[1],
    'MCTS': sns.color_palette('colorblind')[0]
}

files = [pd.read_csv(BASE_PATH+f) for f in os.listdir(BASE_PATH) if f.endswith('.csv')]
df = pd.concat(files)
df['algorithm'] = df['algorithm'].replace({'VO-TREE': 'MCTS-VO'})
del df['Unnamed: 0']
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
final['undiscountedReturn'] = filtered_group['undiscountedReturn'].mean()
final['std_undiscountedReturn'] = filtered_group['undiscountedReturn'].std()
final['discountedReturn'] = filtered_group['discountedReturn'].mean()
final['std_discountedReturn'] = filtered_group['discountedReturn'].std()
final.to_csv('final.csv')
print(final)

# LINEAR VEL SMOOTHNESS
trajectories = [BASE_PATH+f for f in os.listdir(BASE_PATH) if f.endswith('.pkl') and 'acts' in f]
csvs = [f for f in os.listdir(BASE_PATH) if f.endswith('.csv')]
algorithms = {'MCTS-VO': [], 'MCTS': [], 'VO-PLANNER': []}
algorithms_csv = {'MCTS-VO': [], 'MCTS': [], 'VO-PLANNER': []}
for f in csvs:
    for k in algorithms.keys():
        if k in f:
            data = pd.read_csv(BASE_PATH+f)
            if data['reachGoal'].iloc[0]:
                number = f.split('_')[-1].split('.')[0]
                algorithms_csv[k].append(number)

for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            number = f.split('_')[-1].split('.')[0]
            if number in algorithms_csv[k]:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    algorithms[k].append(data)

algorithms_metrics = {}
algorithms_metrics_complete = {}

for k in algorithms.keys():
    metrics = {'vel_smoothness':[]}
    
    for data in algorithms[k]:
        data = np.array(data)
        avg_vel = data[:, 0].mean()
        x,y = data[:, 0] * np.cos(data[:, 1]), data[:, 0] * np.sin(data[:, 1])
        smooth = (np.sqrt(np.diff(x)**2 + np.diff(y)**2)/0.1).sum() * 1/(len(x)-1)
        # smooth = smooth / avg_vel
        metrics['vel_smoothness'].append(smooth)
    algorithms_metrics[k] = {
        'vel_smoothness': np.mean(metrics['vel_smoothness']),
        'std': np.std(metrics['vel_smoothness'])
    }
    algorithms_metrics_complete[k] = metrics['vel_smoothness']

df = pd.DataFrame(algorithms_metrics)
df = df.T.sort_values(by='vel_smoothness', ascending=False)

df.to_csv('metrics_lin_ve.csv')
# color=[colors[x] for x in df.index]
plt.cla()
plt.clf()
sns.barplot(x=df.index, y='vel_smoothness', yerr=df['std'], data=df, palette=[colors[x] for x in df.index])
plt.ylabel(r'$m_{vsm}$')
plt.legend().set_visible(False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/vel_smoothness.png')

# Box plot of linear velocity smoothness
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms_metrics_complete.items():
    data.extend(v)
    labels.extend([k] * len(v))

plt.cla()
plt.clf()
sns.boxplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.xlabel('')
plt.ylabel(r'$m_{vsm}$')
plt.title('Linear Velocity Smoothness by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/vel_smoothness_boxplot.png')

# ANGULAR VEL SMOOTHNESS
trajectories = [BASE_PATH+f for f in os.listdir(BASE_PATH) if f.endswith('.pkl') and 'actions_executed' in f]
algorithms = {'MCTS-VO': [], 'MCTS': [], 'VO-PLANNER': []}
for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            number = f.split('_')[-1].split('.')[0]
            if number in algorithms_csv[k]:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    algorithms[k].append(data)
                
algorithms_metrics = {}
algorithms_metrics_complete = {}
for k in algorithms.keys():
    metrics = {'h_smoothness':[]}
    
    for data in algorithms[k]:
        data = np.array(data)
        # avg_vel = data[:, 1].mean()
        smooth = (np.abs(np.diff(data[:, 1]))/0.1).sum() * 1/(len(data)-1)
        metrics['h_smoothness'].append(smooth)
    algorithms_metrics[k] = {
        'h_smoothness': np.mean(metrics['h_smoothness']),
        'std': np.std(metrics['h_smoothness'])
    }
    algorithms_metrics_complete[k] = metrics['h_smoothness']

df = pd.DataFrame(algorithms_metrics)
df = df.T.sort_values(by='h_smoothness', ascending=False)

df.to_csv('metrics_ang_ve.csv')
plt.cla()
plt.clf()
sns.barplot(x=df.index, y='h_smoothness', yerr=df['std'], data=df, palette=[colors[x] for x in df.index])
plt.ylabel(r'$m_{hsm}$')
plt.legend().set_visible(False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/ang_smoothness.png')


# Box plot of angular velocity smoothness
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms_metrics_complete.items():
    data.extend(v)
    labels.extend([k] * len(v))

sns.boxplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.xlabel('')
plt.ylabel(r'$m_{hsm}$')
plt.title('Angular Velocity Smoothness by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/ang_smoothness_boxplot.png')

# PLOTS
plt.cla()
plt.clf()
files = [pd.read_csv(BASE_PATH+f) for f in os.listdir(BASE_PATH) if f.endswith('.csv')]
df = pd.concat(files)
df['algorithm'] = df['algorithm'].replace({'VO-TREE': 'MCTS-VO'})
del df['Unnamed: 0']
df[['reachGoal', 'collision', 'maxSteps']] = df[['reachGoal', 'collision', 'maxSteps']].astype(int)
df = df[df['reachGoal'] == 1]

# Bar plot of the undiscounted return
plt.cla()
plt.clf()
grouped = df.groupby('algorithm')
mean_return = grouped['undiscountedReturn'].mean()
std_return = grouped['undiscountedReturn'].std()
sns.barplot(x=mean_return.index, y=mean_return.values, yerr=std_return.values, palette=[colors[x] for x in mean_return.index])
plt.ylabel(r'$\rho_u$')
plt.title('Undiscounted Return by Algorithm')
plt.xticks(rotation=0)
plt.xlabel('')
plt.tight_layout()
plt.savefig('plots/undiscounted_return.png')


# Boxplot of the undiscounted return
plt.cla()
plt.clf()
sns.boxplot(x='algorithm', y='undiscountedReturn', data=df, palette=[colors[x] for x in df['algorithm'].unique()])
plt.xlabel('')
plt.ylabel(r'$\rho_u$')
plt.title('Undiscounted Return by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/undiscounted_return_boxplot.png')

# Bar plot of the number of steps
plt.cla()
plt.clf()
mean_steps = grouped['nSteps'].mean()
std_steps = grouped['nSteps'].std()
sns.barplot(x=mean_steps.index, y=mean_steps.values, yerr=std_steps.values, palette=[colors[x] for x in mean_steps.index])
plt.ylabel(r'$n_s$')
plt.title('Number of Steps by Algorithm')
plt.xticks(rotation=0)
plt.xlabel('')
plt.tight_layout()
plt.savefig('plots/nSteps_bar.png')

# Boxplot of the number of steps
plt.cla()
plt.clf()
sns.boxplot(x='algorithm', y='nSteps', data=df, palette=[colors[x] for x in df['algorithm'].unique()])
plt.xlabel('')
plt.ylabel(r'$n_s$')
plt.title('Number of Steps by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/nSteps_boxplot.png')

# Bar plot of the discounted return
plt.cla()
plt.clf()
mean_return = grouped['discountedReturn'].mean()
std_return = grouped['discountedReturn'].std()
sns.barplot(x=mean_return.index, y=mean_return.values, yerr=std_return.values, palette=[colors[x] for x in mean_return.index])
plt.ylabel(r'$\rho$')
plt.xlabel('')
plt.title('Discounted Return by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/discounted_return.png')


# Boxplot of the discounted return
plt.cla()
plt.clf()
sns.boxplot(x='algorithm', y='discountedReturn', data=df, palette=[colors[x] for x in df['algorithm'].unique()])
plt.xlabel('')
plt.ylabel(r'$\rho$')
plt.title('Discounted Return by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/discounted_return_boxplot.png')



trajectories = [BASE_PATH+f for f in os.listdir(BASE_PATH) if f.endswith('.pkl') and 'sim_num' in f]
algorithms = {'MCTS-VO': [], 'MCTS': []}
for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            number = f.split('_')[-1].split('.')[0]
            if number in algorithms_csv[k]:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    algorithms[k].append(data)
                
# Boxplot of the simulations that reach goal
algorithms2 = {k: list(itertools.chain(*algorithms[k])) for k in algorithms.keys()}
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms2.items():
    data.extend(v)
    labels.extend([k] * len(v))

sns.boxplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.xlabel('')
plt.ylabel(r'$m$')

plt.title('# Simulation by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/simulation_data_boxplot_GOAL.png')

trajectories = [BASE_PATH+f for f in os.listdir(BASE_PATH) if f.endswith('.pkl') and 'sim_num' in f]
algorithms = {'MCTS-VO': [], 'MCTS': []}
for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            number = f.split('_')[-1].split('.')[0]
            if number in algorithms_csv[k]:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    algorithms[k].append(data)

# BarPlot of the simulations that reach goal
algorithms2 = {k: list(itertools.chain(*algorithms[k])) for k in algorithms.keys()}
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms2.items():
    data.extend(v)
    labels.extend([k] * len(v))

sns.barplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.ylabel(r'$m$')

plt.title('# Simulation by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/simulation_data_bar_GOAL.png')

# Boxplot of the simulations all
trajectories = [BASE_PATH+f for f in os.listdir(BASE_PATH) if f.endswith('.pkl') and 'sim_num' in f]
algorithms = {'MCTS-VO': [], 'MCTS': []}
for f in trajectories:
    for k in algorithms.keys():
        if k in f:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                algorithms[k].append(data)
algorithms2 = {k: list(itertools.chain(*algorithms[k])) for k in algorithms.keys()}
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms2.items():
    data.extend(v)
    labels.extend([k] * len(v))

sns.boxplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.xlabel('')
plt.ylabel(r'$m$')

plt.title('# Simulation by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/simulation_data_boxplot_ALL.png')

# BarPlot of the simulations all
algorithms2 = {k: list(itertools.chain(*algorithms[k])) for k in algorithms.keys()}
plt.cla()
plt.clf()
data = []
labels = []
for k, v in algorithms2.items():
    data.extend(v)
    labels.extend([k] * len(v))

sns.barplot(x=labels, y=data, palette=[colors[x] for x in set(labels)])
plt.ylabel(r'$m$')

plt.title('# Simulation by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/simulation_data_bar_ALL.png')

# Bar plot of the collision


files = [pd.read_csv(BASE_PATH+f) for f in os.listdir(BASE_PATH) if f.endswith('.csv')]
df = pd.concat(files)
df['algorithm'] = df['algorithm'].replace({'VO-TREE': 'MCTS-VO'})
del df['Unnamed: 0']
df[['reachGoal', 'collision', 'maxSteps']] = df[['reachGoal', 'collision', 'maxSteps']].astype(int)

plt.cla()
plt.clf()
grouped = df.groupby('algorithm')
mean_collision = grouped['collision'].mean() + grouped['Obscollision'].mean()
sns.barplot(x=mean_collision.index, y=mean_collision.values, palette=[colors[x] for x in mean_collision.index])
plt.xlabel('')
plt.ylabel(r'$\eta_c$')
plt.title('Collision Rate by Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/collision_bar.png')

# Bar plot of reachGoal
plt.cla()
plt.clf()
grouped = df.groupby('algorithm')
mean_goal = grouped['reachGoal'].mean()
sns.barplot(x=mean_goal.index, y=mean_goal.values, palette=[colors[x] for x in mean_goal.index])
plt.ylabel(r'$\eta_g$')
plt.title('Success Rate by Algorithm')
plt.xticks(rotation=0)
plt.xlabel('')
plt.tight_layout()
plt.savefig('plots/reachGoal_bar.png')
