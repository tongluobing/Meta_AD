#%%
import os
import xml.etree.ElementTree as ET

import pandas as pd
import neuroCombat
import numpy as np

def neurocombat(centers, personal_info_prefix = 'personal_info/{}.csv',
                cat_prefix = 'report/cat_{}.xml',
                csv_prefix = 'roi_gmv/{}.csv',
                out_prefix = 'neurocombat_gmv/{}.csv'):
    all_features = []
    covars = []
    i = 1
    out_pathes = []
    for center in centers:
        file_dir = center.file_dir
        for person in center.persons:
            # Load Age, Gender
            csv_path = os.path.join(file_dir,
                                    personal_info_prefix.format(person.filename))
            df = pd.read_csv(csv_path)
            values = df.to_numpy().flatten()
            age = values[0]
            gender = values[1]
            # Load TIV
            xml_path = os.path.join(file_dir,
                                        cat_prefix.format(person.filename))
            report = ET.parse(xml_path)
            root = report.getroot()
            tiv = float(root.findall('./subjectmeasures/vol_TIV')[0].text)

            covars.append([i, age, gender, tiv])
            # Load Features
            csv_path = os.path.join(file_dir,
                                    csv_prefix.format(person.filename))
            df = pd.read_csv(csv_path, index_col=0)
            all_features.append(df.to_numpy().flatten())
            
            # Set Out path
            out_path = os.path.join(file_dir,
                                    out_prefix.format(person.filename))
            out_pathes.append(out_path)

    covars = pd.DataFrame(covars, columns=['centers', 'age', 'gender', 'TIV'])

    # Perform neuroCombat
    batch_col = 'centers'
    categorical_cols = ['gender']
    all_features = np.array(all_features)
    # all_features: rows (features) and columns (subjects)
    combat = neuroCombat.neuroCombat(dat=all_features.T,
                    covars=covars,
                    batch_col=batch_col,
                    categorical_cols=categorical_cols)

    # Save Results
    for out_path, features in zip(out_pathes, combat['data'].T):
        dic = dict(zip(range(1, features.shape[0]+1), features))
        df = pd.DataFrame(dic.items(), columns=['roi', 'value'])
        df.set_index('roi', inplace=True)
        df.to_csv(out_path)
# %%
