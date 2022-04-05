from folktables import ACSDataSource, ACSEmployment, ACSIncome
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from my_useful_funcs import *

#--------------------------------------------------------------------------------------------------
state_list= {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
             'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
             'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
             'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
             'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'}
yhat_list_n2 = {}
accuracy_list_n2 = []
tpr_w_list_n2 = []
tpr_b_list_n2 = []
tpr_nw_list_n2 = []
diff_tpr_wb_n2 = []
diff_tpr_wn_n2 = []

for state_itr in state_list:
    # load data of the state_itr
    new_features, new_group, new_labels = load_features(state_itr)
    model = make_pipeline(StandardScaler(), LogisticRegression())
    # train using the new_state data
    model.fit(new_features, new_labels)

    for state in state_list:
        # load data of the state
        features, group, labels = load_features(state)

        # test the model trained on new_state, on the other states data
        s1 = model.score(features, labels)
        # get fairness
        yhat_sd = model.predict(features)
        white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 1)])
        black_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 2)])
        non_white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group != 1)])

        print(f'accuracy of model trained on {state_itr}'+ f' data and tested on {state} state:\n' + str(s1))
        print(f'also TPR of {state} for 3 groups: \n\t1-white group: ' + str(white_tpr_sd) + '\n\t2-black_group: '
              + str(black_tpr_sd) + '\n\t3-all non whites:' + str(non_white_tpr_sd))

        yhat_list_n2[f'{state_itr}_'+state]= yhat_sd
        accuracy_list_n2.append([state, s1])
        tpr_w_list_n2.append([state, white_tpr_sd])
        tpr_b_list_n2.append([state,black_tpr_sd])
        tpr_nw_list_n2.append([state,non_white_tpr_sd])
        diff_tpr_wb_n2.append([state, abs(white_tpr_sd - black_tpr_sd)])
        diff_tpr_wn_n2.append([state, abs(white_tpr_sd - non_white_tpr_sd)])

## Sort the differences in TPR wb(white_black) and wn(white_non-white) and also accuracy
accuracy_list_n2 = Sort(accuracy_list_n2)
diff_tpr_wb_n2   = Sort(diff_tpr_wb_n2)
diff_tpr_wn_n2   = Sort(diff_tpr_wn_n2)

""" saves 1-"CA predicted labels", 2-"CA accuracy", 3- "CA white-black diff", 4-"CA white-nonwhite", 5-"CA TPR white", 
6-"CA TPR black", 7-"CA TPR non-white",  trained on CA & test on states--> 8-"all predicted labels", 9-"all accuracy"
10- "all TPR diff WB",  11-"all TPR diff WN" ,12"all TPR white", 13-"all TPR black", 14-"all TPR nw" """
save_vars(yhat_list_n2, accuracy_list_n2, diff_tpr_wb_n2, diff_tpr_wn_n2, tpr_w_list_n2, tpr_b_list_n2, tpr_nw_list_n2)

#-------------------------------------------------------------------------------------------------------------
## US_model stats
us_features, us_group, us_labels = load_features(state= 'US')

## train-test split of US data
X_train_us, X_test_us, y_train_us, y_test_us, group_train_us, group_test_us = \
    train_test_split(us_features, us_labels, us_group, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
# train on training set of US
model.fit(X_train_us, y_train_us)
score_us = model.score(X_test_us, y_test_us)
yhat_us = model.predict(X_test_us)

white_tpr_us = np.mean(yhat_us[(y_test_us == 1) & (group_test_us == 1)])
black_tpr_us= np.mean(yhat_us[(y_test_us == 1) & (group_test_us == 2)])
non_white_tpr_us = np.mean(yhat_us[(y_test_us == 1) & (group_test_us != 1)])

US_diff_WB = abs(white_tpr_us - black_tpr_us)
US_diff_WN = abs(white_tpr_us - non_white_tpr_us)

# retrain using the whole US data
model.fit(us_features, us_labels)

i=0
yhat_list = {}
accuracy_list = []
tpr_w_list= []
tpr_b_list= []
tpr_nw_list= []
diff_tpr_wb_us = []
diff_tpr_wn_us = []

for state in state_list:
    # load data of the state
    features, group, labels = load_features(state)

    # test the model trained on US on other states data
    s1 = model.score(features, labels)
    # get fairness
    yhat_sd = model.predict(features)
    white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 1)])
    black_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 2)])
    non_white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group != 1)])

    print('accuracy of model trained on the US-' + f'data and tested on {state} state:\n' + str(s1))
    print(f'also TPR of {state} for 3 groups: \n\t1-white group: ' + str(white_tpr_sd) + '\n\t2-black_group: '
          + str(black_tpr_sd) + '\n\t3-all non whites:' + str(non_white_tpr_sd))

    yhat_list['US_'+state]= yhat_sd
    accuracy_list.append([state, s1])
    tpr_w_list.append([state, white_tpr_sd])
    tpr_b_list.append([state,black_tpr_sd])
    tpr_nw_list.append([state,non_white_tpr_sd])
    diff_tpr_wb_us.append([state, abs(white_tpr_sd - black_tpr_sd)])
    diff_tpr_wn_us.append([state, abs(white_tpr_sd - non_white_tpr_sd)])

## Sort the differences in TPR wb(white_black) and wn(white_non-white) and also accuracy
accuracy_list = Sort(accuracy_list)
diff_tpr_wb_us   = Sort(diff_tpr_wb_us)
diff_tpr_wn_us   = Sort(diff_tpr_wn_us)

#------------------------------------------------------------------------------------------------------------------
# designing boxplots
data = [list(dict(diff_tpr_wb_n2).values()), list(dict(diff_tpr_wb_us).values()),
        list(dict(diff_tpr_wn_n2).values()), list(dict(diff_tpr_wn_us).values())]

title='Comparison of Black and non-White racial groups TPR difference with White groups \n' \
          'over each state when model being trained on every-state/all_the_US data'
xlabel=' '
ylabel='Difference of TPR rates'
differences = ['N2 models W-B','US model W-B', 'N2 models W-nW', 'US model W-nW']
leg1='N-Model trains N2 tests'
leg2= 'Model trained on US'
limit = max(list(itertools.chain(*data)))+ 0.05
name = 'bp_n2'

produce_bp(data, title, xlabel, ylabel, differences, leg1, leg2, name, limit)
