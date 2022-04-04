from folktables import ACSDataSource, ACSEmployment, ACSIncome
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import itertools
import pickle

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
    with open(os.path.join(results_path, 'variables.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
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

def bar_plot(CA_vals,stt_vals, states):
    mode = 'group'
    if mode == 'group':   ## plot group bar charts of Ca once with the min_state, once with max_state
        bar_width = 0.25
        fig, ax = plt.subplots()
        bars = ('white', 'black', 'non-white', 'total_accuracy')
        y_pos = np.arange(len(bars))
        scores = [CA_vals[0], CA_vals[1], CA_vals[2], CA_vals[3]]
        scores2 = [stt_vals[0], stt_vals[1], stt_vals[2], stt_vals[3]]

        # create the bars
        plt.barh(y_pos, scores, height=bar_width, color='green', label='Ca-state')
        plt.barh(y_pos + 0.25, scores2, height=bar_width, color='b', label=states[1]+'-state')

        # Put names in the middle of each group-bar
        ax.set_yticks(y_pos + bar_width / 2)
        ax.set_yticklabels(bars)
        # expanding the label axis to 1
        plt.xlim([0, 1])

        title = "TPRs and accuracy of a model learned on Ca \nand tested on " + states[1] + "-state"
        plt.xlabel('True Positive rates')
        plt.ylabel('Racial groups and accuracy')
        plt.title(title)
        plt.legend(loc='best', ncol=1, shadow=False)

    # the following block is to add label to the bars
    rects = ax.patches
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change how you like.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = "{:.3f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)
    plt.tight_layout()
    plt.savefig(results_path+'//'+states[1]+'.svg', format='svg')

def difference_bar(CA, min_stt, max_stt, US):
    bar_width = 0.15
    fig, ax = plt.subplots()
    bars = ('W-B diff', 'W_nW diff') #, str(max_stt)+' W-B', str(max_stt)+' W-N', 'US'+' W-B', 'US'+' W-N')
    y_pos = np.arange(len(bars))
    scores = [CA[0], CA[1]]
    scores2 = [min_stt[1], min_stt[2]]
    scores3 = [max_stt[1], max_stt[2]]
    scores4=  [US[1], US[2]]

    # create the bars
    plt.barh(y_pos, scores, height=bar_width, color='tab:green', label='Ca-state')
    plt.barh(y_pos+ bar_width, scores2, height=bar_width, color='tab:red', label=min_stt[0] + '-state')
    plt.barh(y_pos+ bar_width*2, scores3, height=bar_width, color='tab:blue', label=max_stt[0] + '-state')
    plt.barh(y_pos+ bar_width*3, scores4, height=bar_width, color='tab:orange', label='the US')

    # Put names in the middle of each group-bar
    ax.set_yticks(y_pos + bar_width*1.5)
    ax.set_yticklabels(bars)
    # expanding the label axis to 1
    plt.xlim([0, 0.5])

    title = "TPR-difference of model learned on Ca-state, tested on\n" + \
            str(min_stt[0])+", "+str(max_stt[0])+ "-states respectively, and the US"
    plt.xlabel('Difference of TPR rates')
    plt.ylabel('Racial groups per state')
    plt.title(title)
    plt.legend(loc='best', ncol=1, shadow=False)

    # the following block is to add label to the bars


    rects = ax.patches
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change how you like.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = "{:.3f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)
    plt.tight_layout()
    plt.savefig(results_path + '//' + 'TPR-differnce.svg', format='svg')
###---------------------------------------------------------------------------------------------------------------------

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
# Producing boxplots

us_features, us_group, us_labels = load_features(state= 'US')

## train-test split of CA data
X_train_us, X_test_us, y_train_us, y_test_us, group_train_us, group_test_us = \
    train_test_split(us_features, us_labels, us_group, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
# train on training set of CA
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

# retrain using the whole CA data
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

    # test the model trained on CA on other states data
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
# designing boxplots
data = [list(dict(diff_tpr_wb).values()), list(dict(diff_tpr_wb_us).values()),
        list(dict(diff_tpr_wn).values()), list(dict(diff_tpr_wn_us).values())]

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
          'over each state when model being trained on Ca-state/all_the_US data',
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
top = 0.27 #max(list(itertools.chain(*data)))+ 0.05
bottom = 0
ax1.set_ylim(bottom, top)
differences = ['Ca model W-B','US model W-B', 'Ca model W-nW', 'US model W-nW']
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
fig.text(0.475, 0.08, 'Model trained on Ca',
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
plt.savefig(results_path + '//' + 'bp.svg', format='svg')