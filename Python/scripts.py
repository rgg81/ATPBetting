#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime,timedelta

from past_features import *
from elo_features import *
from categorical_features import *
from stategy_assessment import *
from utilities import *
import random



################################################################################
######################### Building of the raw dataset ##########################
################################################################################

### Importation of the Excel files - 1 per year (from tennis.co.uk)
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Bet365 and Pinnacle.

import glob

while True:

    data=pd.read_csv("../Generated Data/atp_data.csv")
    data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))


    ######################### The period that interests us #########################

    beg = datetime.datetime(2008,1,1)
    end = data.Date.iloc[-1]
    indices = data[(data.Date>beg)&(data.Date<=end)].index

    data = data.iloc[indices,:].reset_index(drop=True)


    ################################################################################
    #################### Strategy assessment - ROI computing #######################
    ################################################################################

    ## We adopt a sliding window method. We predict the outcome of delta consecutive matches ,
    ## with the N previous matches. A small subset of the training set is devoted to
    ## validation (the consecutive matches right before the testing matches)

    ######################### Confidence computing for each match ############################
    features=pd.read_csv("../Generated Data/atp_data_features.csv")

    start_date=datetime.datetime(2018,1,1) #first day of testing set
    test_beginning_match=data[data.Date==start_date].index[0] #id of the first match of the testing set
    span_matches=len(data)-test_beginning_match+1

    # duration_val_matches_options = [2000, 1500, 2500, 1000, 500]
    # duration_val_matches_options = [2000, 2200, 1900]
    duration_val_matches_options = [2000]
    # duration_train_matches_options = [10400, 8600, 12000, 11000]
    # duration_train_matches_options = [10400, 12400]
    duration_train_matches_options = [10400]
    # duration_test_matches_options = [100, 50, 75, 150]
    duration_test_matches_options = [100]
    # threshold_options = [[(0.05,0.10),(0.04,0.10)],
    #                      [(0.03,0.10),(0.02,0.10)],
    #                      [(0.07,0.10),(0.06,0.10)],
    #                      [(0.01,0.05),(0.02,0.05)],
    #                      [(0.02,0.10),(0.01,0.10)],
    #                      [(0.03,0.10),(0.04,0.10)],
    #                      [(0.03,0.05),(0.04,0.05)],
    #                      [(0.09,0.05),(0.10,0.05)],
    #                      [(0.11,0.05),(0.12,0.05)]]


    # threshold_options = [[(0.03,0.50),(0.02,0.50)],
    #                      [(0.03,0.50),(0.04,0.50)],
    #                      [(0.02,0.50),(0.01,0.50)],
    #                      [(0.05,0.50),(0.06,0.50)]]

    threshold_options = [[(0.06,0.50),(0.07,0.50)]]

    learning_rate_options = [0.3]

    # max_depth_options = [4, 6, 8, 3]
    max_depth_options = [4]
    # early_stopping_rounds_options = [50, 30, 10, 20]
    # early_stopping_rounds_options = [100]
    early_stopping_rounds_options = [500]
    # subsample_options = [0.3, 0.2, 0.4, 0.1, 0.5]
    # subsample_options = [0.3, 0.4, 0.2]
    subsample_options = [0.4]
    # colsample_bytree_options = [0.3, 0.2, 0.4, 0.1, 0.5]
    # colsample_bytree_options = [0.3, 0.4, 0.2]
    colsample_bytree_options = [0.4]
    # mode_options = ['max', 'sum']
    mode_options = ['max']

    mode = random.choice(mode_options)

    duration_val_matches=random.choice(duration_val_matches_options)
    duration_train_matches=random.choice(duration_train_matches_options)
    duration_test_matches=random.choice(duration_test_matches_options)
    list_thresholds = random.choice(threshold_options)

    ## Number of tournaments and players encoded directly in one-hot
    nb_players=50
    nb_tournaments=10

    ## XGB parameters
    learning_rate=[random.choice(learning_rate_options)]
    max_depth=[random.choice(max_depth_options)]
    subsample=[random.choice(subsample_options)]
    gamma=[0.8]
    colsample_bytree=[random.choice(colsample_bytree_options)]
    early_stopping_rounds=[random.choice(early_stopping_rounds_options)]
    alpha=[2]
    num_rounds=[300]
    early_stop=[5]
    params=np.array(np.meshgrid(learning_rate,max_depth,subsample,gamma,colsample_bytree,early_stopping_rounds,alpha,num_rounds,early_stop)).T.reshape(-1,9).astype(np.float)
    xgb_params=params[0]


    ## We predict the confidence in each outcome, "duration_test_matches" matches at each iteration
    key_matches=np.array([test_beginning_match+duration_test_matches*i for i in range(int(span_matches/duration_test_matches)+1)])
    confs=[]
    acc_profit = 0
    acc_total_bests = 0
    bet_value = 100
    total_rounds = 0
    for start in key_matches:
        profit,total_matches,_=vibratingAssessStrategyGlobal(start,duration_train_matches,duration_val_matches,duration_test_matches,xgb_params,nb_players,nb_tournaments,features,data, list_thresholds, mode=mode)
        total_value = bet_value*total_matches
        # profit_iter = profit/100 * total_value
        acc_profit+=profit
        acc_total_bests+=total_matches
        total_rounds+=1
        print('Acc PROFIT:{} profit iter:{} total value:{} total rounds:{}'.format(acc_profit, profit, total_value, total_rounds))

    with open('out.txt', 'a') as f:
        options_selected = [str(acc_profit), str(acc_total_bests), mode, str(duration_val_matches),
                            str(duration_train_matches), str(duration_test_matches), str(list_thresholds), str(max_depth),
                            str(subsample), str(colsample_bytree), str(early_stopping_rounds), str(learning_rate)]
        f.write(', '.join(options_selected) + '\n')


