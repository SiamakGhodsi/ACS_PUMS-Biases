import os.path
import joblib
import json
import csv
import pandas as pd

# save using joblib
def write_to_file(obj, filename, path=None, overwrite=False):
    if path is not None:
        filename = os.path.join(path, filename)
    filename = os.path.abspath(filename)
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not overwrite and os.path.exists(filename):
        print("WARNING: file already exists %s; not overwriting." % (filename,))
        pass
        # Check to see whether same as one on disk?
        # When to overwrite?
    else:
        print("Writing to %s" % (filename,))
        joblib.dump(obj, filename)


# Special-case stuff
# ------------------

# -----------------------------------------------------------------

# save JSON
def save_json():
    my_details = {
        'name': 'John Doe',
        'age': 29
    }

    with open('personal.json', 'w') as json_file:
        json.dump(my_details, json_file)


#----------------------------------

# Save CSV
def save_csv(data):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    sales = ['10', '8', '19', '12', '25']

    with open('sales.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(weekdays)
        csv_writer.writerow(sales)

#-------------------------------
# csv reader

## df = pd.read_csv('sales.csv')

"""
how to find data type in python:
1- type(data_name)
2- isinstance(data,str) --> see if a data is string or not
3- df.shape
"""

# save as numpy(.npy) file
""" np.save( str1 + '_features_' + str2, raw_features)
e.g.:
# np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
# np.load('/tmp/123.npy')
-------------------------------
another way to load is also
# with load('foo.npz') as data:
#     a = data['a']
"""

# sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
## use the above line to create regression-type graphs like the ones in "https://tinyurl.com/y5zkavpm" section 9
## where they give an example of how sales is related with advertising over TV, Newspaper and Radio

# ------------------------------------------------------------------------

# path = '~/features/'
# sys.path.insert(0, './features')
# os.path.join('./features', str1)


############################### Plot CA bar-charts and/or CA_SD group bar-chart
mode = 'group'
if mode == 'single':  ## this if only plots the accuracy and TPRs for CA state
    bar_width = 0.330
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])

    # data to be plotted
    bars = ('white', 'black', 'non-white', 'total_accuracy')
    y_pos = np.arange(len(bars))
    scores = [white_tpr_ca, black_tpr_ca, non_white_tpr_ca, score_ca]

    # create the bars
    plt.barh(y_pos, scores, height=bar_width, color='green')

    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    # expanding the label axis to 1
    plt.xlim([0, 1])

    title = "TPRs and accuracy of a model learned and tested on " + state + " state"
    plt.xlabel('True Positive rates')
    plt.ylabel('three racial groups')
    plt.title(title)
    # plt.yticks(('Accuracy', 'Balanced Acc.', 'Eq. Op.', 'ABROCA', 'TPR Prot.', 'TPR Non-Prot.',
    #            'TNR Prot.', 'TNR Non-Prot.'))
elif mode == 'group':
    bar_width = 0.25
    fig, ax = plt.subplots()
    bars = ('white', 'black', 'non-white', 'total_accuracy')
    y_pos = np.arange(len(bars))
    scores = [white_tpr_ca, black_tpr_ca, non_white_tpr_ca, score_ca]

    # create the bars
    plt.barh(y_pos, scores, height=bar_width, color='green', label='CA-state')
    plt.barh(y_pos + 0.25, s1, height=bar_width, color='b', label='SD-state')

    # Put names in the middle of each group-bar
    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(bars)
    # expanding the label axis to 1
    plt.xlim([0, 1])

    title = "TPRs and accuracy of a model learned and tested on " + state + " state"
    plt.xlabel('True Positive rates')
    plt.ylabel('three racial groups')
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
    # ----------------------------------------------------

plt.tight_layout()
plt.savefig(state)
plt.close()


#### load pickle variables
with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    yhat_list, accuracy_list, diff_tpr_wb, diff_tpr_wn, tpr_w_list, tpr_b_list, tpr_nw_list = pickle.load(f)



#########################################################################
"""

# Plug-in your method for tabular datasets
model = make_pipeline(StandardScaler(), LogisticRegression())

# Train on CA data
model.fit(ca_features, ca_labels)

# Test on other state's data
for stt in state_list:
    # get or load data
    sd_data = data_source.get_data(states=stt, download=False)
    # extract numpy features
    sd_features, sd_labels, sd_group = ACSIncome.df_to_numpy(sd_data)
    # get scores from testing model on other states
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


#############################################################################################################3

# vertical bar with plt interface only

        #fig = plt.figure()
        # data to be plotted
        #ax = fig.add_axes([0, 0, 1, 1])
        bars = ('white', 'black', 'non-white', 'total_accuracy')
        y_pos = np.arange(len(bars))
        scores = [white_tpr_ca, black_tpr_ca, non_white_tpr_ca, score_ca]
        # create the bars
        plt.bar(y_pos, width=0.35 ,scores)
        # Create names on the x-axis
        plt.xticks(y_pos, bars)

        #plt.show()
        #plt.xlabel('Cost Increasement (%)')
        title = "TPRs and accuracy of a model learned and tested on" + state + " state"
        plt.xlabel('three racial groups')
        plt.ylabel('True Positive rates')
        plt.title(title )
        #plt.yticks(('Accuracy', 'Balanced Acc.', 'Eq. Op.', 'ABROCA', 'TPR Prot.', 'TPR Non-Prot.',
        #            'TNR Prot.', 'TNR Non-Prot.'))
        plt.savefig(dir + state+ ".png")


Horizontal bar with plt interface only

        fig, ax = plt.subplots()
        # data to be plotted
        #ax = fig.add_axes([0, 0, 1, 1])
        bars = ('white', 'black', 'non-white', 'total_accuracy')
        y_pos = np.arange(len(bars))
        scores = [white_tpr_ca, black_tpr_ca, non_white_tpr_ca, score_ca]
        # create the bars
        plt.barh(y_pos, scores,height=0.40 ,color='green')
        # Create names on the y-axis
        plt.yticks(y_pos, bars)
        # expanding the label axis to 1
        plt.xlim([0, 1])

        #plt.show()
        #plt.xlabel('Cost Increasement (%)')
        title = "TPRs and accuracy of a model learned and tested on " + state + " state"
        plt.xlabel('True Positive rates')
        plt.ylabel('three racial groups')
        plt.title(title )
        #plt.yticks(('Accuracy', 'Balanced Acc.', 'Eq. Op.', 'ABROCA', 'TPR Prot.', 'TPR Non-Prot.',
        #            'TNR Prot.', 'TNR Non-Prot.'))
        plt.savefig(dir + state,)
        plt.close()
"""