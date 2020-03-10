#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime,timedelta

from past_features import *
from elo_features import *
from categorical_features import *
from stategy_assessment import *
from utilities import *
import random
import logging
from multiprocessing import Pool, cpu_count
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.ERROR)


################################################################################
######################### Building of the raw dataset ##########################
################################################################################

### Importation of the Excel files - 1 per year (from tennis.co.uk)
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Bet365 and Pinnacle.

import glob


def run_opt(config, start_date_index):
    global data, stats, total_interval_opts, total_iterations, features
    import numpy
    import xgboost as xgb
    numpy.random.seed()

    duration_val_matches=config['duration_val_matches']
    duration_train_matches=config['duration_train_matches']
    list_thresholds = config['list_thresholds']

    ## XGB parameters
    learning_rate=[config['learning_rate']]
    max_depth=[config['max_depth']]
    subsample=[config['subsample']]
    colsample_bytree=[config['colsample_bytree']]
    early_stop=[config['early_stop']]
    min_trees=[config['min_trees']]
    total_models=config['total_models']
    total_models_selected=config['total_models_selected']

    ######################### The period that interests us #########################
    test_beginning_match=start_date_index #id of the first match of the testing set

    print(len(data))
    print(start_date_index)
    print(total_interval_opts)
    print(data[data.Date == data.Date.iloc[start_date_index]])
    end = data[data.Date==data.Date.iloc[start_date_index] + total_interval_opts].index[0]

    print('test_beginning_match:{} end:{} config:{}'.format(data.Date.iloc[test_beginning_match], data.Date.iloc[end], config))

    span_matches=end - test_beginning_match + 1

    # duration_val_matches_options = [2000, 1500, 2500, 1000, 500]
    # duration_val_matches_options = [2000, 2200, 1900]
    ## Number of tournaments and players encoded directly in one-hot
    nb_players=50
    nb_tournaments=10

    alpha = [0.8]
    gamma = [0.9]
    num_rounds=[999999]
    params=np.array(np.meshgrid(learning_rate,max_depth,subsample,gamma,colsample_bytree,early_stop,alpha,num_rounds,min_trees)).T.reshape(-1,9).astype(np.float)
    xgb_params=params[0]
    duration_test_matches=100
    mode=['sum']

    ## We predict the confidence in each outcome, "duration_test_matches" matches at each iteration
    print('span_matches:{} duration_test_matches:{}'.format(span_matches, duration_test_matches))
    key_matches=np.array([test_beginning_match+duration_test_matches*i for i in range(int(span_matches/duration_test_matches)+1)])
    print('key_matches:{}'.format(key_matches))
    confs=[]
    acc_profit = 0
    acc_total_bests = 0
    bet_value = 100
    total_rounds = 0
    end_test = None
    for start in key_matches:
        profit,total_matches,_=vibratingAssessStrategyGlobal(start,duration_train_matches,duration_val_matches,duration_test_matches,xgb_params,nb_players,nb_tournaments,features,data, list_thresholds, total_models=total_models, total_models_selected=total_models_selected, mode=mode)
        total_value = bet_value*total_matches
        # profit_iter = profit/100 * total_value
        acc_profit+=profit
        acc_total_bests+=total_matches
        total_rounds+=1
        print('Acc PROFIT:{} profit iter:{} total value:{} total rounds:{}'.format(acc_profit, profit, total_value, total_rounds))
        nm=int(len(features)/2)
        end_test=min(start+duration_test_matches-1,nm-1)

    key_stats = tuple(config.values())
    print(key_stats)
    current_key_value = stats.get(key_stats, (0,0))
    stats[key_stats] = current_key_value[0] + acc_profit, current_key_value[1] + acc_total_bests

    with open('out.txt', 'a') as f:
        f.write("Profit: {} Plays:{} Config:{}".format(acc_profit, acc_total_bests, config) + '\n')

    return end_test + 1,acc_profit,acc_total_bests


data=pd.read_csv("../Generated Data/atp_data.csv")
data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

beg = datetime.datetime(2008,1,1)
indices = data[(data.Date>beg)&(data.Date<data.Date.iloc[-1])].index
data = data.iloc[indices,:].reset_index(drop=True)
features=pd.read_csv("../Generated Data/atp_data_features.csv")

total_interval_opts = timedelta(days=60)
total_iterations = 150
total_best_iterations = 5

start_simulation_date = datetime.datetime(2018,1,1)
test_beginning_match=data[data.Date==start_simulation_date].index[0]

total_profit = 0
total_games = 0

while data.Date.iloc[test_beginning_match] < data.Date.iloc[-1]:

    duration_val_matches_options = [2000]
    duration_train_matches_options = [10400]
    # threshold_options = [(0.02, 0.01),(0.03, 0.02),(0.04, 0.03)]
    threshold_options = [(0.03, 0.02)]

    # learning_rate_options = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    learning_rate_options = [0.35, 0.3, 0.25]
    # learning_rate_options = [0.3]
    max_depth_options = [4, 5, 6, 7, 8, 9]
    # max_depth_options = [8]
    #max_depth_options = [3, 4, 5, 6]
    # early_stopping_rounds_options = [50, 100, 300]
    early_stopping_rounds_options = [100]
    subsample_options = [0.3, 0.35, 0.25, 0.15, 0.05]
    # subsample_options = [0.25]
    #subsample_options = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    colsample_bytree_options = [0.3, 0.35, 0.25, 0.15, 0.05]
    # colsample_bytree_options = [0.40]
    #colsample_bytree_options = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    mode_options = ['sum']
    total_models_options = [20]
    total_models_selected_options = [20, 15, 10, 15, 5]
    # total_models_selected_options = [15]
    # total_models_options = [5, 7, 10]
    # min_trees_options = [5, 10, 20, 30, 40]
    # min_trees_options = [5, 15, 30, 50]
    min_trees_options = [20, 10, 30]
    stats = {}

    def gen_random_config():

        mode = random.choice(mode_options)

        duration_val_matches=random.choice(duration_val_matches_options)
        duration_train_matches=random.choice(duration_train_matches_options)
        list_thresholds = random.choice(threshold_options)

        ## XGB parameters
        learning_rate=random.choice(learning_rate_options)
        max_depth=random.choice(max_depth_options)
        subsample=random.choice(subsample_options)
        gamma=[0.8]
        colsample_bytree=random.choice(colsample_bytree_options)
        early_stop=random.choice(early_stopping_rounds_options)
        alpha=[2]
        total_models = random.choice(total_models_options)
        total_models_selected = random.choice(total_models_selected_options)
        min_trees = random.choice(min_trees_options)

        config = {
            "duration_train_matches": duration_train_matches,
            "duration_val_matches": duration_val_matches,
            "list_thresholds": list_thresholds,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stop": early_stop,
            "total_models": total_models,
            "total_models_selected": total_models_selected,
            "min_trees": min_trees
        }
        return config


    end_match = None

    params_config = [(gen_random_config(),test_beginning_match) for x in range(total_iterations)]

    # with Pool(cpu_count()-1) as pool:
    with Pool(1) as pool:
        result = pool.starmap(run_opt, params_config)

    for index_best_iteration in range(total_best_iterations):
        list_stats = list(stats.items())
        list_stats.sort(key=lambda x: x[1][0], reverse=True)
        print("Running best iteration with bests:{}".format(list_stats[:10]))
        params_best = [(best_configs[0],test_beginning_match) for best_configs in list_stats[:10]]

        with Pool(1) as pool:
            result = pool.map(run_opt, params_best)

    test_beginning_match = result[-1][0]
    list_stats = list(stats.items())
    list_stats.sort(key=lambda x: x[1][0], reverse=True)
    params_best = [(best_configs[0],test_beginning_match) for best_configs in list_stats[:10]]

    with Pool(1) as pool:
        result = pool.map(run_opt, params_best)

    for index, a_param_best in enumerate(params_best):
        profit_iteration = result[index][1]
        games_iteration = result[index][2]
        total_profit += profit_iteration
        total_games += games_iteration
        print("a param:{} a profit:{} games:{} profit acc:{} games acc:{}".format(a_param_best, profit_iteration, games_iteration, total_profit, total_games))

    print("Reseting stats")
    stats = {}

    print("end date:{} profit acc:{} games acc:{}".format(data.Date.iloc[test_beginning_match], total_profit, total_games))

