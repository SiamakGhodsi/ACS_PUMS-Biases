from folktables import ACSDataSource, ACSEmployment
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data_source = ACSDataSource(survey_year='2019', horizon='1-Year', survey='person')

from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year='2019', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)
sd_data = data_source.get_data(states=["SD"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
sd_features, sd_labels, sd_group = ACSIncome.df_to_numpy(sd_data)

# Plug-in your method for tabular datasets
model = make_pipeline(StandardScaler(), LogisticRegression())

# Train on CA data
model.fit(ca_features, ca_labels)

# Test on SD data
s1= model.score(sd_features, sd_labels)

# get fairness
yhat=model.predict(sd_features)
white_tpr = np.mean(yhat[(sd_labels == 1) & (sd_group == 1)])
black_tpr = np.mean(yhat[(sd_labels == 1) & (sd_group == 2)])
non_white_tpr = np.mean(yhat[(sd_labels == 1) & (sd_group != 1)])

print('accuracy: model trained on CA data and tested on SD state:' + str(s1))
print('also TPR of SD for 3 groups: \n\t1-white group: '+ str(white_tpr)+'\n\t2-black_group: '
      + str(black_tpr)+ '\n\t3-all non whites:'+ str(non_white_tpr))


########################### Spatial Distribution shift
X_train_ca, X_test_ca, y_train_ca, y_test_ca, group_train_ca, group_test_ca = train_test_split(
    ca_features, ca_labels, ca_group, test_size=0.2, random_state=0)

model.fit(X_train_ca, y_train_ca)

s2= model.score(X_test_ca, y_test_ca)

yhat = model.predict(X_test_ca)

white_tpr = np.mean(yhat[(y_test_ca == 1) & (group_test_ca == 1)])
black_tpr = np.mean(yhat[(y_test_ca == 1) & (group_test_ca == 2)])
non_white_tpr = np.mean(yhat[(y_test_ca == 1) & (group_test_ca != 1)])

print('\naccuracy: model trained and tested on CA:' + str(s2))
print('also TPR of SD for 3 groups: \n\t1-white group: '+ str(white_tpr)+'\n\t2-black_group: '
      + str(black_tpr)+ '\n\t3-all non whites:'+ str(non_white_tpr))



########################### Time shift


from folktables import ACSDataSource, ACSPublicCoverage
from sklearn.linear_model import LogisticRegression

# Download 2014 data
data_source = ACSDataSource(survey_year=2014, horizon='1-Year', survey='person')
acs_data14 = data_source.get_data(states=["CA"], download=True)
features14, labels14, _ = ACSPublicCoverage.df_to_numpy(acs_data14)

# Train model on 2014 data
# Plug-in your method for tabular datasets
model = LogisticRegression()
model.fit(features14, labels14)

# Evaluate model on 2015-2018 data
accuracies = []
for year in [2015, 2016, 2017, 2018]:
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
    accuracies.append(model.score(features, labels))



