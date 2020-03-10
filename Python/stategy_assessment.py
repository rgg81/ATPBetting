#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import seaborn as sns
from utilities import *

############################### STRATEGY ASSESSMENT ############################
### the following functions are used to make the predictions and compute the ROI

# list_thresholds =  [0.7, 0.5, 0.3, 0.15, 0.1, 0.05, 0.02]
# list_thresholds =  [(0.05,0.10),(0.04,0.10),(0.03,0.10),(0.02,0.10)]
# list_thresholds =  [0.09]

def find_max_profit_threshold(xtrain, xval, xtest, preds, labels, list_thresholds):
    # labels = dm.get_label()

    if len(labels) == len(xtrain):
        odds = xtrain['odds']
    elif len(labels) == len(xval):
        odds = xval['odds']
    else:
        odds = xtest['odds']

    odds_filtered = ~np.isnan(odds)
    odds = odds[odds_filtered]

    labels_filtered = labels[odds_filtered]
    preds_filtered = preds[odds_filtered]

    preds_sorted = np.copy(preds_filtered)
    preds_sorted.sort()

    profits = []

    for percent in list(list_thresholds):
        min_threshold = preds_sorted[-int(percent*len(preds_sorted))]

        # preds_cod = preds_filtered > threshold

        preds_cod = min_threshold < preds_filtered



        labels__bet = labels_filtered[preds_cod]
        odds_bet = odds[preds_cod]

        labels_bet_good = labels__bet == 1
        odds_bet_good = odds_bet[labels_bet_good]

        number_matches_we_bet_on = len(labels__bet)

        if number_matches_we_bet_on > 0:

            profit=100*(sum(odds_bet_good)-number_matches_we_bet_on)
        else:
            profit=0
        threshold = min_threshold
        profits.append((profit,threshold))
    return max(profits, key=lambda item: item[0])

def find_profit_threshold(features, preds, labels, threshold):
    # labels = dm.get_label()

    odds = features['odds']
    min_threshold = threshold

    odds_filtered = ~np.isnan(odds)
    odds = odds[odds_filtered]

    labels_filtered = labels[odds_filtered]
    preds_filtered = preds[odds_filtered]

    preds_sorted = np.copy(preds_filtered)
    preds_sorted.sort()

    # preds_cod = preds_filtered > threshold
    preds_cod = min_threshold < preds_filtered

    labels__bet = labels_filtered[preds_cod]
    odds_bet = odds[preds_cod]

    labels_bet_good = labels__bet == 1
    odds_bet_good = odds_bet[labels_bet_good]

    number_matches_we_bet_on = len(labels__bet)

    if number_matches_we_bet_on > 0:

        profit=100*(sum(odds_bet_good)-number_matches_we_bet_on)
    else:
        profit=0

    return profit,number_matches_we_bet_on


def xgbModelBinary(xtrain, ytrain, xval, yval, xtest, ytest, p, evals_result, list_thresholds, sample_weights=None):
    """
    XGB model training. 
    Early stopping is performed using xval and yval (validation set).
    Outputs the trained model, and the prediction on the validation set

    """

    def feval(preds, dm):
    # binary classes
        labels = dm.get_label()

        profits,thresholds = find_max_profit_threshold(xtrain,xval,xtest,preds,labels, list_thresholds)

        # from sklearn.metrics import roc_auc_score
        # auc = roc_auc_score(labels, preds)
        return [('my_auc', profits)]



    # if sample_weights==None:
    #     dtrain=xgb.DMatrix(xtrain,label=ytrain)
    # else:
    #     dtrain=xgb.DMatrix(xtrain,label=ytrain,weight=sample_weights)

    even_row = xtrain.index % 2 == 0
    # print(even_row)
    weights = xtrain['odds'].fillna(0)

    # print(weights[0:10])
    weights[~even_row] = 1
    weights[even_row] = 10 * weights[even_row]
    # print(len(weights))
    # print(weights[0:10])
    # print(ytrain[0:10])


    dtrain=xgb.DMatrix(xtrain,label=ytrain,weight=weights)
    # dtrain=xgb.DMatrix(xtrain,label=ytrain)
    dval=xgb.DMatrix(xval,label=yval)

    dtest=xgb.DMatrix(xtest,label=ytest)
    eval_set = [(dtrain,"train"), (dtest, 'test'), (dval, 'validation')]
    # params={"objective":"binary:logistic",'subsample':0.8,
    #         'min_child_weight':p[2],'alpha':p[6],'lambda':p[5],'max_depth':int(p[1]),
    #         'gamma':p[3],'eta':p[0],'colsample_bytree':p[4]}
    # model=xgb.train(params, dtrain, int(p[7]), feval=feval, maximize=True, evals=eval_set,early_stopping_rounds=int(p[8]))

    call_back_custom_early = early_stop(p[5], True, True, min_iteration=p[8] - 1)

    params={"objective":"binary:logistic",'subsample':p[2],'max_depth':int(p[1]), 'seed': random.randint(1,999999),
            'colsample_bytree':p[4], 'eta': p[0]}
    # print('Training total samples train:{} total samples validation:{}'.format(len(xtrain), len(xval)))
    model=xgb.train(params, dtrain, 99999, feval=feval, evals=eval_set, callbacks=[call_back_custom_early], evals_result=evals_result, verbose_eval=True)
    return model


def assessStrategyGlobal(test_beginning_match,
                         duration_train_matches,
                         duration_val_matches,
                         duration_test_matches,
                         xgb_params,
                         nb_players,
                         nb_tournaments,
                         features,
                         data,
                         list_thresholds,
                         model_name="0"):
    """
    Given the id of the first match of the testing set (id=index in the dataframe "data"),
    outputs the confidence dataframe.
    The confidence dataframe tells for each match is our prediction is right, and for
    the outcome we chose, the confidence level.
    The confidence level is simply the probability we predicted divided by the probability
    implied by the bookmaker (=1/odd).
    """
    ########## Training/validation/testing set generation
    
    # Number of matches in our dataset (ie. nb. of outcomes divided by 2)
    nm=int(len(features)/2)
    
    # Id of the first and last match of the testing,validation,training set
    beg_test=test_beginning_match
    end_test=min(test_beginning_match+duration_test_matches-1,nm-1)
    end_val=min(beg_test-1,nm-1)
    beg_val=beg_test-duration_val_matches
    end_train=beg_val-1
    beg_train=beg_val-duration_train_matches
       
    train_indices=range(2*beg_train,2*end_train+2)
    val_indices=range(2*beg_val,2*end_val+2)
    test_indices=range(2*beg_test,2*end_test+2)

    print('train indices:{} {} val:{} {} teste:{} {}'.format(train_indices[0],train_indices[-1],val_indices[0],val_indices[-1],test_indices[0],test_indices[-1]))
    
    if (len(test_indices)==0)|(len(train_indices)==0):
        return 0
    
    # Split in train/validation/test
    xval=features.iloc[val_indices,:].reset_index(drop=True)
    xtest=features.iloc[test_indices,:].reset_index(drop=True)
    xtrain=features.iloc[train_indices,:].reset_index(drop=True)
    ytrain=pd.Series([1,0]*int(len(train_indices)/2))
    yval=pd.Series([1,0]*int(len(val_indices)/2))
    ytest=pd.Series([1,0]*int(len(test_indices)/2))
    
    # We limit the number of players and tournaments one-hot encoded : we'll keep only the 
    # players that won the most matches to avoid overfitting and make the process quicker
    # Biggest players :
    biggest_players=data.iloc[range(beg_train,end_train),:][["Winner","Loser"]]
    biggest_players=pd.concat([biggest_players.Winner,biggest_players.Loser],0)
    biggest_players=list(biggest_players.value_counts().index[:nb_players])
    player_columns=[el for el in xtrain.columns if el[:6]=="player"]
    to_drop_players=[el for el in player_columns if el[7:] not in biggest_players]
    # Biggest Tournaments
    biggest_tournaments=data.iloc[range(beg_train,end_train),:]["Tournament"]
    biggest_tournaments=list(biggest_tournaments.value_counts().index[:nb_tournaments])
    tournament_columns=[el for el in xtrain.columns if el[:10]=="tournament"]
    to_drop_tournaments=[el for el in tournament_columns if el[11:] not in biggest_tournaments]
    # We drop smallest Tournaments and players
    xtrain=xtrain.drop(to_drop_players+to_drop_tournaments,1)
    xval=xval.drop(to_drop_players+to_drop_tournaments,1)
    xtest=xtest.drop(to_drop_players+to_drop_tournaments,1)

    evals_result = {}
    
    ### ML model training

    # print(len(xtrain.columns))
    # for aColumn in xtrain.columns:
    #     print(aColumn)
    # model = None

    model=xgbModelBinary(xtrain, ytrain, xval, yval, xtest, ytest, xgb_params, evals_result, list_thresholds, sample_weights=None)
    
    # The probability given by the model to each outcome of each match :
    pred_val= model.predict(xgb.DMatrix(xval,label=None), ntree_limit=model.best_ntree_limit)
    pred_test= model.predict(xgb.DMatrix(xtest,label=None), ntree_limit=model.best_ntree_limit)

    max_profit,threshold = find_max_profit_threshold(xtrain,xval,xtest,pred_val,yval, list_thresholds)

    max_profit_test,threshold_test = find_max_profit_threshold(xtrain,xval,xtest,pred_test,ytest, list_thresholds)





    profit_test,total_matches_bet = find_profit_threshold(xtest,pred_test,ytest,threshold)
    profit_test_check,total_matches_bet_check = find_profit_threshold(xtest,pred_test,ytest,threshold_test)

    assert profit_test_check == max_profit_test


    # pred_metric = model.evals_result()


    
    return profit_test,total_matches_bet,max_profit

def vibratingAssessStrategyGlobal(km,dur_train,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,list_threshold, total_models=20,total_models_selected=10, mode='max'):
    """
    The ROI is very sensistive to the training set. A few more matches in the training set can
    change it in a non-negligible way. Therefore it is preferable to run assessStrategyGlobal several times
    with slights changes in the training set lenght, and then combine the predictions.
    This is what this function does.
    More precisely we compute the confidence dataset of 7 models with slightly different training sets.
    For each match, each model has an opinion of the winner, and a confidence is its prediction.
    For each match, the final chosen outcome is the outcome chosen by the most models (majority voting)
    And the final confidence is the average of the confidences of the models that chose this outcome.
    """
    profits_matches = []
    bet_value = 100
    for a_model in range(total_models):
        profit_iter,total_matches_bet,max_profit=assessStrategyGlobal(km,dur_train,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,list_threshold, str(a_model))

        # total_value = bet_value*total_matches_bet
        # profit_iter = roi/100 * total_value

        print('VALIDATION STATS MODEL NAME:{} PROFIT:{} MATCHES:{} MAX VAL ROI:{} MODE:{}'
              .format(a_model, profit_iter, total_matches_bet, max_profit, mode))


        profits_matches.append((profit_iter,total_matches_bet,max_profit))

    #
    if mode == 'max':
        return max(profits_matches, key=lambda item:item[2])
    else:
        profits_matches.sort(key=lambda x:x[2], reverse=True)
        profits_matches = profits_matches[:total_models_selected]
        print("Selecting {} best val profits:{}".format(total_models_selected, profits_matches))
        return [sum(x) for x in zip(*profits_matches)]


############################### PROFITS COMPUTING AND VISUALIZATION ############

def profitComputation(percentage,confidence,model_name="0"):
    """
    Input : percentage of matches we want to bet on,confidence dataset
    Output : ROI
    """
    tot_number_matches=len(confidence)
    number_matches_we_bet_on=int(tot_number_matches*(percentage/100))
    matches_selection=confidence.head(number_matches_we_bet_on)
    profit=100*(matches_selection.PSW[matches_selection["win"+model_name]==1].sum()-number_matches_we_bet_on)/number_matches_we_bet_on
    return profit

def plotProfits(confidence,title=""):
    """
    Given a confidence dataset, plots the ROI according to the percentage of matches
    we bet on. 
    """
    profits=[]
    ticks=range(5,101)
    for i in ticks:
        p=profitComputation(i,confidence)
        profits.append(p)
    plt.plot(ticks,profits)
    plt.xticks(range(0,101,5))
    plt.xlabel("% of matches we bet on")
    plt.ylabel("Return on investment (%)")
    plt.suptitle(title)