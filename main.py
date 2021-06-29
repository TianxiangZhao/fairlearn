import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pandas.core.frame import DataFrame

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from fairlearn.metrics import MetricFrame
import utils
import ipdb
import argparse


parser = argparse.ArgumentParser(description='FairML')
parser.add_argument("--epoch", default=2, type=int)
parser.add_argument("--pretrain_epoch", default=1, type=int)
parser.add_argument("--method", default="base", type=str,choices=['base','corre','groupTPR','learn','remove','learnCorre'])
parser.add_argument("--dataset", default="adult", type=str, choices=['adult','pokec', 'compas','law'])
parser.add_argument("--s", default="sex", type=str) #sex for adult
parser.add_argument("--related",nargs='+', type=str)#choices=['sex','race','age','relationship','marital-status', 'education', 'workclass'] for adult
parser.add_argument("--r_weight",nargs='+', type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weightSum", default=0.3, type=float)# used for learning related weights, weight for corre attr
parser.add_argument("--beta", default=0.5, type=float)#weight for regularization of Lambda

parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model", default='MLP', type=str, choices=['MLP', 'LR', 'SVM'])#weight for regularization of Lambda

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print('beta: {}, weightSum: {}'.format(args.beta, args.weightSum))

#-----------------------
#data analysis
#-----------------------
#load data
if args.dataset == 'adult':
    data = fetch_openml(data_id=1590)
    
    header = list(data.data.columns)
    if args.method =='remove':
        for attr in args.related:
            header.remove(attr)
    data.data = data.data[header]

    X = pd.get_dummies(data.data)
    X = X.sort_index(axis=1)
    y_true = ((data.target == '>50K') * 1).values
    n_classes = y_true.max()+1

    #data_frame = pd.DataFrame(data.data, columns=data.feature_names)
    #sensitive attribute obtain
    data_frame = data.data
    sensitive_attr = data_frame[args.s]
    
    #print(sensitive_attr.value_counts())#1 for male, 0 for female
    #print(data.data.groupby('sensitive_attr')['relationship'].value_counts()/data.data.groupby('sensitive_attr')['relationship'].count())
    
    for relate in args.related:
        coef = utils.cal_correlation(data.data, args.s, relate)
        print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))
        data.data['target'] = data.target
        coef = utils.cal_correlation(data.data, 'target', relate)
        print('coefficient between {} and {} is: {}'.format('target', relate, coef))
    


elif args.dataset == 'pokec':
    args.s = "region"
    predict_attr = "I_am_working_in_field"

    data_name='region_job'
    idx_features_labels = pd.read_csv("./data/pokec/{}.csv".format(data_name))
    idx_features_labels = idx_features_labels.sort_index(axis=1)

    print(idx_features_labels.keys())

    header = list(idx_features_labels.columns)
    header.remove("user_id")

    sensitive_attr = idx_features_labels[args.s]
    data_frame = idx_features_labels[header]

    header.remove(predict_attr)
    if args.method =='remove':
        for attr in args.related:
            header.remove(attr)

    #X = np.array(idx_features_labels[header], dtype=np.float32)
    X = idx_features_labels[header]
    #y_true = idx_features_labels[predict_attr].values
    y_true = idx_features_labels[predict_attr].values
    label_idx = np.where(y_true>=0)[0]
    X = X.iloc[label_idx,:]
    y_true = y_true[label_idx]
    n_classes = y_true.max()+1

    data_frame = data_frame.iloc[label_idx,:]
    sensitive_attr = sensitive_attr.iloc[label_idx]
    
    ipdb.set_trace()
    print(data_frame.groupby(args.s)[predict_attr].value_counts()/data_frame.groupby(args.s)[predict_attr].count())

    
elif args.dataset == 'law':
    args.s = "sex"
    predict_attr = "admit"

    data_name='processed_data'
    idx_features_labels = pd.read_csv("./data/law_school/{}.csv".format(data_name))
    idx_features_labels = idx_features_labels.sort_index(axis=1)

    print(idx_features_labels.keys())

    header = list(idx_features_labels.columns)

    sensitive_attr = idx_features_labels[args.s]
    data_frame = idx_features_labels[header]

    header.remove(predict_attr)
    
    if args.method =='remove':
        for attr in args.related:
            header.remove(attr)

    #X = np.array(idx_features_labels[header], dtype=np.float32)
    X = idx_features_labels[header]
    #y_true = idx_features_labels[predict_attr].values
    y_true = idx_features_labels[predict_attr].values
    label_idx = np.where(y_true>=0)[0]
    X = X.iloc[label_idx,:]
    y_true = y_true[label_idx]
    n_classes = y_true.max()+1

    data_frame = data_frame.iloc[label_idx,:]
    sensitive_attr = sensitive_attr.iloc[label_idx]
    
    X = pd.get_dummies(X)

    
    for relate in args.related:
        coef = utils.cal_correlation(data_frame, args.s, relate)
        print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))

    
    #print(data_frame.groupby(args.s)[predict_attr].value_counts()/data_frame.groupby(args.s)[predict_attr].count())

elif args.dataset == 'compas':
    args.s = 'race'
    predict_attr = "is_recid"

    data_name='Processed_Compas'
    idx_features_labels = pd.read_csv("./data/{}.csv".format(data_name))
    
    print(idx_features_labels.keys())

    header = list(idx_features_labels.columns)

    sensitive_attr = idx_features_labels[args.s]
    data_frame = idx_features_labels[header]

    header.remove(predict_attr)
    

    if args.method =='remove':
        for attr in args.related:
            header.remove(attr)

    #X = np.array(idx_features_labels[header], dtype=np.float32)
    X = idx_features_labels[header]
    #y_true = idx_features_labels[predict_attr].values
    y_true = idx_features_labels[predict_attr].values

    n_classes = y_true.max()+1
    
    
    for relate in args.related:
        coef = utils.cal_correlation(data_frame, args.s, relate)
        print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))

        coef = utils.cal_correlation(data_frame, predict_attr, relate)
        print('coefficient between {} and {} is: {}'.format(predict_attr, relate, coef))

    #ipdb.set_trace()
    #print(data_frame.groupby(args.s)[predict_attr].value_counts()/data_frame.groupby(args.s)[predict_attr].count())

    X = pd.get_dummies(X)
    X = X.sort_index(axis=1)


#-----------------------
#preprocessing
#-----------------------
# split into train/test set
indict = np.arange(sensitive_attr.shape[0])
(X_train, X_test, y_train, y_test, ind_train, ind_test) = train_test_split(X, y_true, indict, test_size=0.5,
                                     stratify=y_true, random_state=7)

# standardize the data
processed_X_train = X_train
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, np.ndarray):
            return torch.from_numpy(df).float()
        return torch.from_numpy(df.values).float()

train_data = PandasDataSet(X_train, y_train, ind_train)
test_data = PandasDataSet(X_test, y_test, ind_test)

train_loader = DataLoader(train_data, batch_size=320, shuffle=True, drop_last=True)

print('# training samples:', len(train_data))
print('# batches:', len(train_loader))

#-----------------------
#model
#-----------------------
class Classifier(nn.Module):

    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, x):
        return self.network(x)

class Classifier_lr(nn.Module):
    def __init__(self, n_features, n_class=2):
        super(Classifier_lr, self).__init__()

        self.linear = nn.Linear(n_features, n_class)


    def forward(self, x):
        
        return self.linear(x)

def loss_SVM(result, truth, model):
    truth[truth==0] = -1
    result = result.squeeze()
    weight = model.linear.weight.squeeze()

    loss = torch.mean(torch.clamp(1 - truth * result, min=0))
    loss += 0.1*torch.mean(torch.mul(weight, weight))

    return loss


n_features = X.shape[1]
#print('feature dimension: {}'.format(n_features))
if args.dataset=='pokec':
    n_hid = 72
else:
    n_hid = 32

if args.model == 'MLP':
    clf = Classifier(n_features=n_features, n_hidden=n_hid,n_class=n_classes)
elif args.model == 'LR':
    clf = Classifier_lr(n_features=n_features,n_class=n_classes)
elif args.model == 'SVM':
    assert n_classes == 2, "classes need to be 2 for SVM classifier"
    clf = Classifier_lr(n_features=n_features,n_class=1)
else:
    raise NotImplementedError("not implemented model: {}".format(args.model))

clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)

#-----------------------
#run
#-----------------------
##Baseline
''
def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y,_ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        loss.backward()
        optimizer.step()
    return clf

##feature-pertubation loss
def Perturb_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        
        for related_attr, related_weight in zip(related_attrs, related_weights):
            #x_new = utils.counter_sample(data.data.iloc[ind.int()], related_attr, scaler)
            x_new = utils.counter_sample(data.data, ind.int(), related_attr, scaler)
            p_y_new = clf(x_new)

            #cor_loss = torch.square(p_y[:,1] - p_y_new[:,1]).mean()
            p_stack = torch.stack((p_y[:,1], p_y_new[:,1]), dim=1)
            p_order = torch.argsort(p_stack,dim=-1)
            cor_loss = torch.square(p_stack[:,p_order[:,1].detach()] - p_stack[:,p_order[:,0]]).mean()

            #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + cor_loss*related_weight

        loss.backward()
        optimizer.step()

    return clf

def CorreErase_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)

        for related_attr, related_weight in zip(related_attrs, related_weights):
            selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
            cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

            #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + cor_loss*related_weight

        loss.backward()
        optimizer.step()

    return clf

##group fairness loss
def Gfair_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        #
        for related_attr, related_weight in zip(related_attrs, related_weights):
            group_TPR = utils.groupTPR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            #group_TPR_loss = (group_TPR - (sum(group_TPR)/len(group_TPR)).detach()).sum()*related_weight
            group_TPR_loss = torch.square(max(group_TPR).detach() - min(group_TPR))

            #group_TNR = utils.groupTNR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            #group_TNR_loss = torch.square(max(group_TNR).detach() - min(group_TNR))
            #print('classification loss: {}, group TPR loss: {}, group TNR loss: {}'.format(loss.item(), group_TPR_loss.item(), group_TNR_loss.item()))
            #print('classification loss: {}, group TPR loss: {}'.format(loss.item(), group_TPR_loss.item()))
            loss = loss + group_TPR_loss*related_weight
        loss.backward()
        optimizer.step()
    return clf

#correlation regularization with learned weights
def CorreLearn_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights, weightSum):
    
    for x, y, ind in data_loader:
        UPDATE_MODEL_ITERS = 1
        UPDATE_WEIGHT_ITERS = 1

        #update model
        for iter in range(UPDATE_MODEL_ITERS):
            clf.zero_grad()
            p_y = clf(x)
            if args.model != 'SVM':
                loss = criterion(p_y, y.long())
            else:
                loss = criterion(p_y, y, clf)

            for related_attr, related_weight in zip(related_attrs, related_weights.tolist()):
                selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
                cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

                #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
                loss = loss + cor_loss*related_weight*weightSum

            loss.backward()
            optimizer.step()

        #update weights
        #ipdb.set_trace()
        for iter in range(UPDATE_WEIGHT_ITERS):
            with torch.no_grad():
                p_y = clf(x)

                cor_losses = []
                for related_attr in related_attrs:
                    selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
                    cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

                    cor_losses.append(cor_loss.item())

                cor_losses = np.array(cor_losses)

                cor_order = np.argsort(cor_losses)

                #compute -v. represent it as v.
                beta = args.beta
                v = cor_losses[cor_order[0]]+ 2*beta
                cor_sum = cor_losses[cor_order[0]]
                l=1
                for i in range(cor_order.shape[0]-1):
                    if cor_losses[cor_order[i+1]] < v:
                        cor_sum = cor_sum + cor_losses[cor_order[i+1]]
                        v = (cor_sum+2*beta)/(i+2)
                        l = l+1
                    else:
                        break
                
                #compute lambda
                for i in range(cor_order.shape[0]):
                    if i <l:
                        related_weights[cor_order[i]] = (v-cor_losses[cor_order[i]])/(2*beta)
                    else:
                        related_weights[cor_order[i]] = 0



                '''
                #older optimization version
                #update
                #related_weights = related_weights - cor_losses*0.001
                #mapping
                #related_weights[related_weights<0] = 0
                #related_weights = related_weights/sum(related_weights)*weightSum
                '''


    return clf, related_weights

#train
related_attrs = args.related
related_weights = args.r_weight

if args.model != 'SVM':
    clf_criterion = nn.CrossEntropyLoss()
else:
    clf_criterion = loss_SVM

for i in range(args.pretrain_epoch):
    clf = clf.train()
    clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)


for epoch in range(args.epoch):
    clf = clf.train()
    if args.method == 'base':
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)
    if args.method == 'conterfactual':#only implemented for dataset ADULT
        clf = Perturb_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'corre':
        clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'groupTPR':
        clf = Gfair_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'learnCorre':
        related_weights = np.array(related_weights)
        clf, related_weights = CorreLearn_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights, args.weightSum)



#test
clf = clf.eval()
with torch.no_grad():
    pre_clf_test = clf(test_data.tensors[0])

if args.model != 'SVM':
    y_pred = pre_clf_test.argmax(dim=1)
else:
    y_pred = (pre_clf_test>0).reshape(-1).int()

print('sensitive attributes: ')
print(set(sensitive_attr.iloc[ind_test]))
#print('labels: ')
#print(set(y_test))
#print('label dist: ')
#unique, counts = np.unique(y_test, return_counts=True)
#print(dict(zip(unique, counts)))

print('sum of weights weightSUM for learning: {}'.format(args.weightSum))
print('learned lambdas: {}'.format(related_weights))

gm = MetricFrame(metrics.accuracy_score, y_test, y_pred, sensitive_features=sensitive_attr.iloc[ind_test])
print('Average accuracy score: {}'.format(gm.overall))
print(gm.by_group)

group_selection_rate = []
group_equal_odds = []
sens_test = sensitive_attr.iloc[ind_test]
for sens_value in set(sens_test):
    y_sense_pred = y_pred[(sens_test==sens_value).values]
    y_sense_test = y_test[(sens_test==sens_value).values]
    sens_sr = []
    sens_eo = []

    for label in set(y_test):
        if label>0:
            sens_sr_label = (y_sense_pred==label).sum()/y_sense_pred.shape[0]
            sens_eo_label = (y_sense_pred[y_sense_test==label]==label).sum()/(y_sense_test==label).sum()

            sens_sr.append(sens_sr_label)
            sens_eo.append(sens_eo_label)

    group_selection_rate.append(sens_sr)
    group_equal_odds.append(sens_eo)

group_selection_rate = np.array(group_selection_rate)
group_equal_odds = np.array(group_equal_odds)


print('group equal odds: ')
print(group_equal_odds)
print('eo_difference: {}'.format(np.mean(np.absolute(group_equal_odds-np.mean(group_equal_odds, axis=0, keepdims=True)))))
if args.dataset=='compas':
    print('target eo_difference: {}'.format((np.absolute(group_equal_odds[0]-group_equal_odds[2]))))

print('group selection rate: ')
print(group_selection_rate)
print('sr_difference: {}'.format(np.mean(np.absolute(group_selection_rate-np.mean(group_selection_rate, axis=0, keepdims=True)))))
if args.dataset=='compas':
    print('target sr_difference: {}'.format((np.absolute(group_selection_rate[0]-group_selection_rate[2]))))


    
'''
gm = MetricFrame(metrics.accuracy_score, y_test, y_pred, sensitive_features=sensitive_attr.iloc[ind_test])
print('Average accuracy score: {}'.format(gm.overall))
print(gm.by_group)

#geographic parity
from fairlearn.metrics import selection_rate
sr = MetricFrame(selection_rate, y_test, y_pred, sensitive_features=sensitive_attr.iloc[ind_test])
print('Average selection_rate: {}'.format(sr.overall))
print(sr.by_group)


#equal odds
gm = MetricFrame(metrics.multilabel_confusion_matrix, y_test, y_pred, sensitive_features=sensitive_attr.iloc[ind_test])
print('confusion matrix for each class:')
print(gm.overall)
print(gm.by_group)
ipdb.set_trace()


#y_pred = pd.Series(pre_clf_test.data.numpy()[:,1], index=y_test.index)
#fig.savefig('images/torch_biased_training.png')
'''


'''
#analyze fair-related metrics
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)
gm = MetricFrame(metrics.accuracy_score, y_true, y_pred, sensitive_features=sensitive_attr)
print(gm.overall)
print(gm.by_group)

from fairlearn.metrics import selection_rate
sr = MetricFrame(selection_rate, y_true, y_pred, sensitive_features=sensitive_attr)
print(sr.overall)
print(sr.by_group)


#mitigating fairness issue
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient

constraint = DemographicParity()
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
mitigator = ExponentiatedGradient(classifier, constraint)
mitigator.fit(X, y_true, sensitive_features=sensitive_attr)
y_pred_mitigated = mitigator.predict(X)

sr_mitigated = MetricFrame(selection_rate, y_true, y_pred_mitigated, sensitive_features=sensitive_attr)
print(sr_mitigated.overall)
print(sr_mitigated.by_group)
'''
