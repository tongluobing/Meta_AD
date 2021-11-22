#%%
import numpy as np
import pandas as pd
import os

def create_stats_csv(centers, csv_prefix, out_dir='./data/stats/',
                     divide_tiv=False):
    labels = [0, 1, 2]
    label_names = ['NC', 'MCI', 'AD']
    out_dir = os.path.join(out_dir, csv_prefix)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for label in labels:
        datas = None
        for center in centers:
            persons_eg = center.get_by_label(label)
            features_eg, _, ids = center.get_csv_values(persons=persons_eg,
                                                            prefix=csv_prefix+'/{}.csv',
                                                            flatten=True)
            tivs, _ = center.get_tivs(label)
            if features_eg is not None:
                if divide_tiv:
                    features_eg = np.divide(features_eg, tivs[:, np.newaxis])
                if datas is None:
                    datas = features_eg
                else:
                    datas = np.vstack((datas, features_eg))

        _mean = np.mean(datas, axis=0)
        _std = np.std(datas, axis=0)
        print(np.shape(_mean)[0]+1)
        index = np.arange(1, np.shape(_mean)[0]+1)
        v_list = []
        v_list.append(_mean)
        v_list.append(_std)
        v_list = np.array(v_list)
        v_dict = dict(zip(index, v_list.T))
        df = pd.DataFrame(data=v_dict, index=['mean', 'std'], dtype=np.float32)
        df.T.to_csv(os.path.join(out_dir, label_names[label]+'.csv'))
# %%
import matplotlib.pyplot as plt
import scipy.stats as stats
def plot_stats(stats_dir, roi, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    files = os.listdir(stats_dir)
    legends = []
    for f in files:
        df = pd.read_csv(os.path.join(stats_dir, f), index_col=0)
        mean = df.loc[roi]['mean']
        std = df.loc[roi]['std']
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        ax.plot(x, stats.norm.pdf(x, mean, std))
        legends.append(f[:-4])
    return ax, legends
# %%
import scipy

def cal_personal_scores(centers, stats_dir, csv_prefix, out_prefix):
    ad_df = pd.read_csv(os.path.join(stats_dir, 'AD.csv'), index_col=0)
    ad_means = ad_df['mean'].values
    ad_stds = ad_df['std'].values

    nc_df = pd.read_csv(os.path.join(stats_dir, 'NC.csv'), index_col=0)
    nc_means = nc_df['mean'].values
    nc_stds = nc_df['std'].values

    for center in centers:
        file_dir = center.file_dir
        for person in center.persons:
            # Load Features
            csv_path = os.path.join(file_dir,
                                csv_prefix.format(person.filename))
            df = pd.read_csv(csv_path, index_col=0)
            features = df.to_numpy().flatten()
            # Calculate personal scores
            personal_scores = []
            roi_counts = np.shape(features)[0]
            for i in range(roi_counts):
                cdf_nc = scipy.stats.norm.cdf(features[i], loc=nc_means[i], scale=nc_stds[i])
                cdf_ad = scipy.stats.norm.cdf(features[i], loc=ad_means[i], scale=ad_stds[i])
                personal_score = cdf_nc/cdf_ad
                # rare case, due to std may cause value > 1
                if personal_score > 1:
                    personal_score = 1
                personal_scores.append(personal_score)

            out_path = os.path.join(file_dir,
                                out_prefix.format(person.filename))

            dic = dict(zip(range(1, roi_counts+1), personal_scores))
            df = pd.DataFrame(dic.items(), columns=['roi', 'value'])
            df.set_index('roi', inplace=True)
            df.to_csv(out_path)