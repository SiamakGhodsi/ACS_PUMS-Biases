from folktables import ACSDataSource, ACSEmployment, ACSIncome
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from my_useful_funcs import *

state_list= {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'US',
              'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'}

ca_features, ca_group, ca_labels = load_features(state= 'CA')

## train-test split of CA data
X_train_ca, X_test_ca, y_train_ca, y_test_ca, group_train_ca, group_test_ca = \
    train_test_split(ca_features, ca_labels, ca_group, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
# train on training set of CA
model.fit(X_train_ca, y_train_ca)
score_ca = model.score(X_test_ca, y_test_ca)
yhat_ca = model.predict(X_test_ca)

white_tpr_ca = np.mean(yhat_ca[(y_test_ca == 1) & (group_test_ca == 1)])
black_tpr_ca = np.mean(yhat_ca[(y_test_ca == 1) & (group_test_ca == 2)])
non_white_tpr_ca = np.mean(yhat_ca[(y_test_ca == 1) & (group_test_ca != 1)])

print('\naccuracy: model trained and tested on Ca:' + str(score_ca))
print('also TPR for the racial groups: \n\t1-white group: ' + str(white_tpr_ca) + '\n\t2-black_group: '
      + str(black_tpr_ca) + '\n\t3-all non whites:' + str(non_white_tpr_ca))

CA_diff_WB = white_tpr_ca - black_tpr_ca
CA_diff_WN = white_tpr_ca - non_white_tpr_ca

# retrain using the whole CA data
model.fit(ca_features, ca_labels)

yhat_list = {}
accuracy_list = []
tpr_w_list= []
tpr_b_list= []
tpr_nw_list= []
diff_tpr_wb = []
diff_tpr_wn = []

for state in state_list:
    # load data of the state
    features, group, labels = load_features(state)

    # test the model trained on CA on other states data
    s1 = model.score(features, labels)
    # get fairness
    yhat_sd = model.predict(features)
    white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 1)])
    black_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 2)])
    non_white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group != 1)])

    print('accuracy: model trained on Ca data and tested on SD state:' + str(s1))
    print('also TPR of SD for 3 groups: \n\t1-white group: ' + str(white_tpr_sd) + '\n\t2-black_group: '
          + str(black_tpr_sd) + '\n\t3-all non whites:' + str(non_white_tpr_sd))

    yhat_list['CA_'+state]= yhat_sd
    #accuracy_list['CA_'+state]=s1
    accuracy_list.append([state, s1])
    tpr_w_list.append([state, white_tpr_sd])
    tpr_b_list.append([state,black_tpr_sd])
    tpr_nw_list.append([state,non_white_tpr_sd])
    diff_tpr_wb.append([state, abs(white_tpr_sd - black_tpr_sd)])
    diff_tpr_wn.append([state, abs(white_tpr_sd - non_white_tpr_sd)])

## Sort the differences in TPR wb(white_black) and wn(white_non-white) and also accuracy
accuracy_list = Sort(accuracy_list)
diff_tpr_wb   = Sort(diff_tpr_wb)
diff_tpr_wn   = Sort(diff_tpr_wn)

""" saves 1-"CA predicted labels", 2-"CA accuracy", 3- "CA white-black diff", 4-"CA white-nonwhite", 5-"CA TPR white", 
6-"CA TPR black", 7-"CA TPR non-white",  trained on CA & test on states--> 8-"all predicted labels", 9-"all accuracy"
10- "all TPR diff WB",  11-"all TPR diff WN" ,12"all TPR white", 13-"all TPR black", 14-"all TPR nw" """
save_vars(yhat_ca, score_ca, CA_diff_WB, CA_diff_WN ,white_tpr_ca, black_tpr_ca, non_white_tpr_ca,
          yhat_list, accuracy_list, diff_tpr_wb, diff_tpr_wn, tpr_w_list, tpr_b_list, tpr_nw_list)

min_state = diff_tpr_wb[0][0]
max_state = diff_tpr_wb[-1][0]
print("\nThe state with the minimum white-black TPR difference from Ca is: "+ str(min_state) +
      " with a TPR difference rate of --> " + str(diff_tpr_wb[0][0]) + "\nand the state with the max difference is: "
      + str(max_state) + " with a TPR difference of of --> " + str(diff_tpr_wb[-1][1]))

min_entries = np.zeros(4)
max_entries = np.zeros(4)
# list of 1-TPR_white, 2-TPR_black, 3-TPR_non-white, 4-Accuracy
min_entries = get_list_val(min_state,tpr_w_list,tpr_b_list,tpr_nw_list,accuracy_list)
max_entries = get_list_val(max_state,tpr_w_list,tpr_b_list,tpr_nw_list,accuracy_list)

bar_plot([white_tpr_ca,black_tpr_ca,non_white_tpr_ca,score_ca], min_entries,['CA',min_state])
bar_plot([white_tpr_ca, black_tpr_ca, non_white_tpr_ca, score_ca], max_entries,['CA',max_state])

entries = get_list_val(min_state, diff_tpr_wb, diff_tpr_wn, diff_tpr_wb, diff_tpr_wn, max_state)
US_entries = get_list_val('US', diff_tpr_wb, diff_tpr_wn, diff_tpr_wb, diff_tpr_wn)
difference_bar([CA_diff_WB,CA_diff_WN], [min_state,entries[0], entries[1]],
               [max_state,entries[2], entries[3]], ['US',US_entries[0],US_entries[1]])

###--------------------------------------------------------------------------------------------------------------------
## US-wide model deployed on each state

us_features, us_group, us_labels = load_features(state= 'US')

## train-test split of the US data
X_train_us, X_test_us, y_train_us, y_test_us, group_train_us, group_test_us = \
    train_test_split(us_features, us_labels, us_group, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
# train on training set of the US
model.fit(X_train_us, y_train_us)
score_us = model.score(X_test_us, y_test_us)
yhat_us = model.predict(X_test_us)

white_tpr_us = np.mean(yhat_us[(y_test_us == 1) & (group_test_us == 1)])
black_tpr_us= np.mean(yhat_us[(y_test_us == 1) & (group_test_us == 2)])
non_white_tpr_us = np.mean(yhat_us[(y_test_us == 1) & (group_test_us != 1)])

print('\naccuracy: model trained and tested on Ca:' + str(score_ca))
print('also TPR for the racial groups: \n\t1-white group: ' + str(white_tpr_ca) + '\n\t2-black_group: '
      + str(black_tpr_ca) + '\n\t3-all non whites:' + str(non_white_tpr_ca))

US_diff_WB = white_tpr_us - black_tpr_us
US_diff_WN = white_tpr_us - non_white_tpr_us

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

    # test the model trained on the US on other states data
    s1 = model.score(features, labels)
    # get fairness
    yhat_sd = model.predict(features)
    white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 1)])
    black_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group == 2)])
    non_white_tpr_sd = np.mean(yhat_sd[(labels == 1) & (group != 1)])

    print('accuracy: model trained on US data and tested on SD state:' + str(s1))
    print('also TPR of SD for 3 groups: \n\t1-white group: ' + str(white_tpr_sd) + '\n\t2-black_group: '
          + str(black_tpr_sd) + '\n\t3-all non whites:' + str(non_white_tpr_sd))

    yhat_list['US_'+state]= yhat_sd
    #accuracy_list['CA_'+state]=s1
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

# again save the variables now for the model learned on the whole US deployed on each state
save_vars(yhat_us, score_us, US_diff_WB, US_diff_WN ,white_tpr_us, black_tpr_us, non_white_tpr_us,
          yhat_list, accuracy_list, diff_tpr_wb_us, diff_tpr_wn_us, tpr_w_list, tpr_b_list, tpr_nw_list)

#------------------------------------------------------------------------------------------------------------------
# producing boxplots
data = [list(dict(diff_tpr_wb).values()), list(dict(diff_tpr_wb_us).values()),
        list(dict(diff_tpr_wn).values()), list(dict(diff_tpr_wn_us).values())]

title='Comparison of Black and non-White racial groups TPR difference with White groups \n' \
      'over each state when model being trained on Ca-state/all_the_US data'
xlabel=' '
ylabel='Difference of TPR rates'
differences = ['Ca model W-B','US model W-B', 'Ca model W-nW', 'US model W-nW']
leg1='Model trained on Ca'
leg2= 'Model trained on US'
limit = 0.27
name= 'bp_Ca'

produce_bp(data, title, xlabel, ylabel, differences, leg1, leg2, name ,limit)
