from folktables import ACSDataSource, ACSEmployment, ACSIncome
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import csv
import pickle
import itertools

## function to sort tuples and list of lists by a specific column
## here our list is has a key and a value column).
def Sort(sub_list):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_list.sort(key=lambda x: x[1])
    return sub_list

results_path = "./results"
if not os.path.exists(results_path):
        os.makedirs(results_path)

####### function to save variables using pickle
def save_vars( *args):
    with open(os.path.join(results_path, 'variables_n2.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([args], f)

def load_features(state):
    horizon = '2019_1-Year.npy'  ## 2019 data as the latest data in the repo is used
    dir = './features'  ## directory to load features from

    str1 = state + '_features_' + horizon
    str2 = state + '_group_' + horizon
    str3 = state + '_labels_' + horizon
    filename1 = os.path.join(dir, str1)
    filename2 = os.path.join(dir, str2)
    filename3 = os.path.join(dir, str3)

    # load data/features, labels and protected group
    features = np.load(filename1)
    group = np.load(filename2)
    labels = np.load(filename3)
    return features, group, labels

def get_list_val(stt,w,b,nw,acc,stt2='CA'):
    w_dict = dict(w)
    b_dict = dict(b)
    nw_dict = dict(nw)
    acc_dict = dict(acc)
    if stt2 == 'CA':
        return [w_dict[stt], b_dict[stt], nw_dict[stt], acc_dict[stt]]
    else:
        return [w_dict[stt], b_dict[stt], nw_dict[stt2], acc_dict[stt2]]

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

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax1.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title='Comparison of Black and non-White racial groups TPR difference with White groups \n'
          'over each state when model being trained on every-state/all_the_US data',
    xlabel=' ',
    ylabel='Difference of TPR rates',
)

# Now fill the boxes with desired colors
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')
    medians[i] = median_y[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, num_boxes + 0.5)
top = max(list(itertools.chain(*data)))+ 0.05
bottom = 0
# to avoid *, also can use list(itertools.chain.from_iterable(data))
ax1.set_ylim(bottom, top)
differences = ['N2 models W-B','US model W-B', 'N2 models W-nW', 'US model W-nW']
#ax1.set_xticklabels(np.repeat(differences, 2), rotation=45, fontsize=8)
ax1.set_xticklabels(differences, rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(num_boxes) + 1
upper_labels = [str(round(s, 3)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k], color=box_colors[k])

# Finally, add a basic legend
fig.text(0.475, 0.08, 'N-Model trains N2 tests',
         backgroundcolor=box_colors[0], color='black', weight='roman',
         size='x-small')
fig.text(0.475, 0.045, 'Model trained on US',
         backgroundcolor=box_colors[1],
         color='white', weight='roman', size='x-small')
fig.text(0.475, 0.010, '*', color='white', backgroundcolor='silver',
         weight='roman', size='medium')
fig.text(0.49, 0.012, ' Average Value', color='black', weight='roman',
         size='x-small')

plt.tight_layout()
plt.savefig(results_path + '//' + 'bpn2.svg', format='svg')