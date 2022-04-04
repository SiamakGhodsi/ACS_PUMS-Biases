from folktables import ACSDataSource, ACSEmployment, ACSIncome
import numpy as np
import os
import csv
import pandas as pd

state_list= {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'US',
              'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'}

horizon = '1-Year'
year = '2019'

data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')

feature_path = "features"
if not os.path.exists(feature_path):
        os.makedirs(feature_path)

for state in state_list:

    str1 = feature_path + "/" + state
    str2 = year + '_' + horizon

    if state=="US":
        if not os.path.exists(str1 + "_features_" + str2):
            dat1 = pd.read_csv('data/2019/1-Year/psam_pusa.csv')
            dat2 = pd.read_csv('data/2019/1-Year/psam_pusb.csv')
            frame = [dat1, dat2]
            raw_data = pd.concat(frame)
            del dat1
            del dat2
            del frame
    else:
        raw_data = data_source.get_data(states=[state], download=False)

    raw_features, raw_labels, raw_group = ACSIncome.df_to_numpy(raw_data)

    if not os.path.exists(str1+"_features_"+str2):
        np.save( str1 + '_features_' + str2, raw_features)
    if not os.path.exists(str1 + "_labels_" + str2):
        np.save(str1 + '_labels_' + str2, raw_labels)
    if not os.path.exists(str1 + "_group_" + str2):
        np.save(str1 + '_group_' + str2, raw_group)

