#%%
import pandas as pd

import abeta_pet
import pet_fdg
import draw_results
import meta_roi
from mask import Mask, NiiMask


mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
mask = NiiMask(mask_path)

df = pd.read_csv('./data/gene/expression.csv', index_col=0)
label_pairs = [(2, 0), (2, 1), (1, 0)]

df = df.T
#%%
from correlation import pearson_r
for label_pair in label_pairs:
    new_df = pd.DataFrame(index=df.index, columns=['Abeta_r', 'Abeta_p',
                                      'FDG_r', 'FDG_p',
                                      'gmv_r', 'gmv_p',
                                      'ct_r', 'ct_p',])
    label_eg = label_pair[0]
    label_cg = label_pair[1]

    abeta_t, _ = abeta_pet.ttest_by_label(label_eg, label_cg)
    fdg_t, _ =  pet_fdg.ttest_by_label(label_eg, label_cg)

    roi_models = meta_roi.meta_gmv(label_eg, label_cg, mask, save_nii=False)
    es1 = {k: v.total_effect_size for k,v in sorted(roi_models.items())}

    roi_models = meta_roi.meta_ct(label_eg, label_cg, mask, save_nii=False, save_gii=False)
    es2 = {k: v.total_effect_size for k,v in sorted(roi_models.items())}

    for gene_id, row in df.iterrows():
        v1 = row.to_dict()
        result = pearson_r(gene_id, v1, abeta_t)
        new_df.loc[gene_id]['Abeta_r'] = result.r
        new_df.loc[gene_id]['Abeta_p'] = result.p

        v1 = row.to_dict()
        result = pearson_r(gene_id, v1, fdg_t)
        new_df.loc[gene_id]['FDG_r'] = result.r
        new_df.loc[gene_id]['FDG_p'] = result.p

        v1 = row.to_dict()
        result = pearson_r(gene_id, v1, es1)
        new_df.loc[gene_id]['gmv_r'] = result.r
        new_df.loc[gene_id]['gmv_p'] = result.p

        v1 = row.to_dict()
        result = pearson_r(gene_id, v1, es2)
        new_df.loc[gene_id]['ct_r'] = result.r
        new_df.loc[gene_id]['ct_p'] = result.p

    new_df.to_csv(r'G:\008AD_gene\01_BN_15633\{}_{}.csv'.format(label_eg, label_cg))
# %%
