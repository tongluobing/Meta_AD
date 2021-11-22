# %%
import os
import datasets
import numpy as np
import pandas as pd
centers = datasets.load_centers_all()

feature_prefix = 'neurocombat_gmv/{}.csv'
prefix = 'personal_scores/{}.csv'
out_dir = './results/subtype/gmv_ADMCI_4'
#target_label = 2
#%%
center_names = []
person_names = []
MMSEs = []
mean_pss = []
ages = []
genders = []
origin_label = []
# Load AD datas
all_features = None
for center in centers:
    target_label = 1
    persons = center.get_by_label(target_label)
    if persons:
        center_names += [center.name for person in persons]
        person_names += [person.filename for person in persons]
        origin_label += [target_label for person in persons]
        MMSEs += center.get_MMSEs(target_label)[0].tolist()
        ages += center.get_ages(target_label)[0].tolist()
        genders += center.get_males(target_label)[0].tolist()

        features, *_ = center.get_csv_values(persons=persons, prefix=prefix, flatten=True)
        mean_pss += np.mean(features, axis=1).tolist()
        if all_features is None:
            all_features = features
        else:
            all_features = np.vstack((all_features, features))
    target_label = 2
    persons = center.get_by_label(target_label)
    if persons:
        center_names += [center.name for person in persons]
        person_names += [person.filename for person in persons]
        origin_label += [target_label for person in persons]
        MMSEs += center.get_MMSEs(target_label)[0].tolist()
        ages += center.get_ages(target_label)[0].tolist()
        genders += center.get_males(target_label)[0].tolist()

        features, *_ = center.get_csv_values(persons=persons, prefix=prefix, flatten=True)
        mean_pss += np.mean(features, axis=1).tolist()
        if all_features is None:
            all_features = features
        else:
            all_features = np.vstack((all_features, features))

# Load NC original features for ttest
all_features_nc = None
for center in centers:
    persons = center.get_by_label(0)
    if persons:
        features, *_ = center.get_csv_values(persons=persons, prefix=feature_prefix, flatten=True)
        if all_features_nc is None:
            all_features_nc = features
        else:
            all_features_nc = np.vstack((all_features_nc, features))

# Load MCI/AD original features for ttest
all_features_ori = None
for center in centers:
    persons = center.get_by_label(1) + center.get_by_label(2)
    if persons:
        features, *_ = center.get_csv_values(persons=persons, prefix=feature_prefix, flatten=True)
        if all_features_ori is None:
            all_features_ori = features
        else:
            all_features_ori = np.vstack((all_features_ori, features))

data = {'Center_name': center_names,
        'Person_name': person_names,
        'MMSE': MMSEs,
        'Mean_PS': mean_pss,
        'Age':ages,
        'gender':genders,
        'origin_label':origin_label}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)
#%%
from draw_results import plot_correlation_joint
plot_correlation_joint( df['Mean_PS'], df['MMSE'],)
#%%
import pickle
with open(os.path.join(out_dir, 'clustering.pkl'), 'rb') as f:
    clustering = pickle.load(f)
#%%
centers = clustering.cluster_centers_
new_centers = [centers[0], centers[3], centers[2], centers[1]]
#%%
from sklearn.cluster import KMeans
import pickle
method = KMeans(n_clusters=4, init=np.array(new_centers))
clustering = method.fit(all_features)
df['Subtype_label'] = clustering.labels_.tolist()
df.to_csv(os.path.join(out_dir, 'subtype.csv'))

with open(os.path.join(out_dir, 'clustering.pkl'), 'wb') as f:
    pickle.dump(clustering, f)
#%%
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import seaborn as sns
tsne = PCA(n_components=2)
all_features = np.array(all_features)
all_labels = np.array(df['origin_label'])
_, counts = np.unique(all_labels, return_counts=True)
print(counts)
embeded_features = tsne.fit_transform(all_features)

x = embeded_features.T[0]
y = embeded_features.T[1]
hue = all_labels
sns.relplot(x=x, y=y, hue=hue)

# %%
from scipy.stats import ttest_ind
from mask import NiiMask
mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
mask = NiiMask(mask_path)

df = pd.read_csv(os.path.join(out_dir, 'subtype.csv'))
all_labels = df['Subtype_label'].values
ls = np.unique(all_labels)
for l in ls:
    all_features_label = None
    for feature, label in zip(all_features_ori, all_labels):
        if label == l:
            if all_features_label is None:
                all_features_label = feature
            else:
                all_features_label = np.vstack((all_features_label, feature))
    ts, ps = ttest_ind(all_features_label, all_features_nc, axis=0)
    ts = [t if p<0.001/len(ts) else 0 for t, p in zip(ts, ps)]
    ts = dict(zip(range(1, len(ts)+1), ts))
    mask.save_values(ts, os.path.join(out_dir, f'subtype{l}.nii'))
#%%
df = pd.read_csv(os.path.join(out_dir, 'subtype.csv'))
def f(row):
    return row['Center_name'][:4]
df['Dataset'] = df.apply(f, axis=1)
array = df.groupby(["Dataset","Subtype_label"]).size()
print(np.sum(array))
print(array/np.sum(array)*100)
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

color = ['#3caf77', '#4d5aaf', '#ffd900', '#d15d55']
boxen_width = 0.6

palette = sns.color_palette(color)

df = pd.read_csv(os.path.join(out_dir, 'subtype.csv'))

column_names = ['FAQ', 'FDG', 'ABETA', 'TAU', 'PTAU', 'ADAS11', 'ADAS13', 'ADASQ4',
                'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']
with open(os.path.join(out_dir, "t.txt"), "w") as f:
    for column_name in column_names:
        info_df = pd.read_csv('./data/center_info/ADNI/ADNIMERGE_BL.csv', index_col=0)
        ls = np.unique(df['Subtype_label'])
        clinical_features = [[] for _ in ls]
        x = []
        y = []
        for label, row in df.iterrows():
            try:
                series = info_df.loc[row['Person_name']]
                value = series[column_name]
                if isinstance(value, str):
                    continue
                if not np.isnan(value):
                    x.append(row['Subtype_label'])
                    y.append(float(value))
                    clinical_features[row['Subtype_label']].append(float(value))
            except KeyError:
                pass
        print([len(clinical_features[i]) for i in range(len(ls))])
        fig = plt.figure(figsize=(3,4))
        ax = fig.add_axes()
        ax = sns.boxenplot(x=x, y=y, palette=palette, width=boxen_width, saturation=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(os.path.join(out_dir, f'{column_name}.jpg'))
        plt.close()
    
        print(f'-------{column_name}----------', file=f)
        label_pairs = []
        for l in ls:
            for ll in ls:
                if ll > l:
                    label_pairs.append([l, ll])
        for label_pair in label_pairs:
            array_1 = np.array(clinical_features[label_pair[0]])
            array_1 = array_1[~np.isnan(array_1)]
            array_2 = np.array(clinical_features[label_pair[1]])
            array_2 = array_2[~np.isnan(array_2)]
            t, p = ttest_ind(array_1, array_2)
            
            print(label_pair, t, p, array_1.shape, array_2.shape, file=f)
        print(f'----------------------------', file=f)
# plot MMSE
def plot_subtype_stat(df, name, out_dir):
    ls = np.unique(df['Subtype_label'])
    clinical_features = [[] for _ in ls]
    x = []
    y = []
    for label, row in df.iterrows():
        x.append(row['Subtype_label'])
        y.append(row[name])
        clinical_features[row['Subtype_label']].append(row[name])
    fig = plt.figure(figsize=(3,4))
    ax = fig.add_axes()
    ax = sns.boxenplot(x=x, y=y, palette=palette, width=boxen_width, saturation=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(out_dir, f'{name}.jpg'))
    plt.close()
    with open(os.path.join(out_dir, "t.txt"), "a") as f:
        print(f'----------{name}--------------', file=f)
        label_pairs = []
        for l in ls:
            for ll in ls:
                if ll > l:
                    label_pairs.append([l, ll])
        for label_pair in label_pairs:
            array_1 = np.array(clinical_features[label_pair[0]])
            array_1 = array_1[~np.isnan(array_1)]
            array_2 = np.array(clinical_features[label_pair[1]])
            array_2 = array_2[~np.isnan(array_2)]
            t, p = ttest_ind(array_1, array_2)
            print(label_pair, t, p, array_1.shape, array_2.shape, file=f)
        print(f'----------------------------', file=f)
plot_subtype_stat(df, 'MMSE', out_dir)
plot_subtype_stat(df, 'Mean_PS', out_dir)
plot_subtype_stat(df, 'Age', out_dir)
#%%
df.groupby(["Subtype_label", "gender", 'origin_label']).size()
#%%

from scipy.stats import fisher_exact
def plot_subtype_gender_chi2(df, out_dir):
    #color = ['#9dd5fe', '#3cacfd', '#fec69d', '#fd8d3c']
    color = ['#9dd5fe', '#fec69d']
    edgecolor = ['#3cacfd', '#fd8d3c']

    x = np.unique(df['Subtype_label'])
    width = 0.4
    linewidth = 2
    height = df.groupby(["Subtype_label", 'origin_label', "gender"]).size()
    mci_female_count = height[::4]
    mci_male_count = height[1::4]
    ad_female_count = height[2::4]
    ad_male_count = height[3::4]

    print(height)
    plt.bar(x, mci_male_count, width=-width, align='edge',
             label='MCI_Male', color=color[0], edgecolor=edgecolor[0],
             linewidth=linewidth)
    plt.bar(x, mci_female_count, width=-width,
             bottom=mci_male_count, align='edge',
             label='MCI_Female', color=color[1], edgecolor=edgecolor[1],
             linewidth=linewidth)
    plt.bar(x, ad_male_count, width=width, align='edge',
             label='AD_Male', color=color[0], edgecolor=edgecolor[0],
             linewidth=linewidth)
    plt.bar(x, ad_female_count, width=width,
             bottom=ad_male_count,align='edge',
             label='AD_Female', color=color[1], edgecolor=edgecolor[1],
             linewidth=linewidth)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    #plt.savefig(os.path.join(out_dir, f'gender.jpg'))
    plt.close()
    """
    with open(os.path.join(out_dir, "t.txt"), "a") as f:
        print(f'----------GENDER--------------', file=f)
        label_pairs = [[0, 1], [0, 2], [1, 2]]
        for label_pair in label_pairs:
            label1, label2 = label_pair
            table = [[male_count[label1].values[0], female_count[label1].values[0]],
                    [male_count[label2].values[0], female_count[label2].values[0]]]
            _or, p = fisher_exact(table)
            print(label_pair, _or, p, table[0], table[1], file=f)
        print(f'----------------------------', file=f)
    """
plot_subtype_gender_chi2(df, out_dir)
#%%
df = pd.read_csv(os.path.join(out_dir, 'subtype.csv'))
def f(row):
    return row['Center_name'][:4]
df['Dataset'] = df.apply(f, axis=1)
df.groupby(["Subtype_label", 'origin_label', ]).size()
# %%
# plot pie plot
import pandas as pd
import matplotlib.pyplot as plt
radius = 2
size = 0.7

df = pd.read_csv(os.path.join(out_dir, 'subtype.csv'))
def f(row):
    return row['Center_name'][:4]
df['Dataset'] = df.apply(f, axis=1)

array = df.groupby(["Dataset","Subtype_label"]).size()

labels = []
cs = []
for i in array.index:
    a, b = i
    labels.append(f'{a}_{b}')
    cs.append(color[b])

fig, ax = plt.subplots()
ax.pie(array, 
        labels=labels,
        pctdistance=0.82,
        radius=radius, colors=cs,
        wedgeprops=dict(width=size, edgecolor='w'))
ax.pie(df.groupby(["Dataset","Subtype_label",'origin_label']).size(),
        radius=radius-size, colors=['#dddddd', '#aaaaaa'],
        wedgeprops=dict(width=0.2, edgecolor='w'))
plt.show()
#%%

# %%
# divide ROI into Big Categories
cate_names = ['SFG', 'MFG', 'IFG','OrG', 'PrG','PCL',
            'STG', 'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
            'SPL', 'IPL', 'Pcun', 'PoG',
            'INS', 'CG',
            'MVOcC', 'LOcC',
            'Amyg', 'Hipp', 'BG', 'Tha']
cate_id = [[1,14],[15,28],[29,40],[41,52],[53,64],[65,68],
            [69,80],[81,88],[89,102],[103,108],[109,120],[121,124],
            [125,134],[135,146],[147,154],[155,162],
            [163,174],[175,188],
            [189,198],[199,210],
            [211,214],[215,218],[219,230],[231,246]]
#%%
# based on personal score
features = [[],[],[]]
for roi_scores, label in zip(all_features, df['Subtype_label'].values):
    features[label].append(roi_scores)
features = np.array(features)

mean_value = [[],[],[]]
for cate_name, id in zip(cate_names, cate_id):
    start = id[0]-1
    end = id[1]
    i = 0
    for subtype_features in features:
        tmp_values = np.array(subtype_features)[:, start:end]
        mean = np.mean(tmp_values)
        mean_value[i].append(mean)
        i += 1
# %%
# based on tvalue
all_labels = df['Subtype_label'].values
ls = np.unique(all_labels)
mean_value = [[],[],[]]
for l in ls:
    all_features_label = None
    for feature, label in zip(all_features_ori, all_labels):
        if label == l:
            if all_features_label is None:
                all_features_label = feature
            else:
                all_features_label = np.vstack((all_features_label, feature))
    ts, ps = ttest_ind(all_features_label, all_features_nc, axis=0)
    for cate_name, id in zip(cate_names, cate_id):
        start = id[0]-1
        end = id[1]
        i = 0
        mean_value[l].append(np.mean(ts[start:end]))
# %%
from matplotlib import cm,colors
# 根据T值调整color的中心
# -43.99~8.31
"""
colors1 = cm.gnuplot2(np.linspace(0, 1, 138))[42:]
colors2 = np.vstack((cm.BuGn(np.linspace(0, 1, 20)[5:])))

# combine them and build a new colormap
cs = np.vstack((colors1, colors2))
cmap = colors.LinearSegmentedColormap.from_list('my_colormap', cs)
"""
#cmap = 'jet'
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(np.array(mean_value).T, center=0.5,yticklabels=cate_names, cmap=cmap,
            square=True)
plt.show()
# %%
# MCAD 指标
sub_df = df.loc[df['Center_name'].isin(['MCAD\AD_S01', 'MCAD\AD_S02'])]
sub_df = sub_df.loc[sub_df['Subtype_label'].isin([1, 2])]
# %%
sub_df
# %%
from draw_results import plot_correlation_joint
x = df['Mean_PS'].values
y = df['MMSE'].values
plot_correlation_joint(x,y)
