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