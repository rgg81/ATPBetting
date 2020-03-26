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
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.ERROR)


################################################################################
######################### Building of the raw dataset ##########################
################################################################################

### Importation of the Excel files - 1 per year (from tennis.co.uk)
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Bet365 and Pinnacle.

import glob


def run_opt(config, start_date_index,index_iteration, optimize=True):
    global data, stats, total_interval_opts, total_iterations, features, data_copy

    threshold_prob_bet = config['threshold_prob_bet']
    data = data_copy[(data_copy['PSW']>threshold_prob_bet)&(data_copy['PSL']>threshold_prob_bet)]
    list_ids = list(data['matchid'].values)

    features = features_copy.loc[features_copy['matchid0'].isin(list_ids)]

    print(f"Features odds filtered:{features[-50:]}")

    data.reset_index(inplace=True)
    features.reset_index(inplace=True)

    print(f"Colum names:{list(features.columns)}")
    print(f"Sample len features:{len(features)} len data:{len(data)}")

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
    factor_weight=1

    features_select = config["features_select"]

    if start_date_index is None:
        start_date_index=data[data.Date>=start_simulation_date].index[0]

    test_beginning_match=start_date_index #id of the first match of the testing set

    print(len(data))
    print(start_date_index)
    print(total_interval_opts)
    print(data[data.Date == data.Date.iloc[start_date_index]])
    end = data[data.Date >= data.Date.iloc[start_date_index] + total_interval_opts].index[0]

    print('test_beginning_match:{} end:{} config:{}'.format(data.Date.iloc[test_beginning_match], data.Date.iloc[end], config))

    span_matches=end - test_beginning_match + 1

    # duration_val_matches_options = [2000, 1500, 2500, 1000, 500]
    # duration_val_matches_options = [2000, 2200, 1900]
    ## Number of tournaments and players encoded directly in one-hot
    nb_players=1
    nb_tournaments=1

    alpha = [0.8]
    gamma = [0.9]
    num_rounds=[999999]
    params=np.array(np.meshgrid(learning_rate,max_depth,subsample,gamma,colsample_bytree,early_stop,alpha,num_rounds,min_trees,factor_weight)).T.reshape(-1,10).astype(np.float)
    xgb_params=params[0]
    duration_test_matches=100
    mode=['sum']

    ## We predict the confidence in each outcome, "duration_test_matches" matches at each iteration
    print('span_matches:{} duration_test_matches:{}'.format(span_matches, duration_test_matches), flush=True)
    key_matches=np.array([test_beginning_match+duration_test_matches*i for i in range(int(span_matches/duration_test_matches)+1)])
    print('key_matches:{}'.format(key_matches), flush=True)
    confs=[]
    acc_profit = 0
    acc_total_bests = 0
    bet_value = 100
    total_rounds = 0
    end_test = None
    for start in key_matches:
        profit,total_matches=vibratingAssessStrategyGlobal(start,duration_train_matches,duration_val_matches,duration_test_matches,xgb_params,nb_players,nb_tournaments,features,data, list_thresholds,features_select, total_models=total_models, total_models_selected=total_models_selected, mode=mode)
        total_value = bet_value*total_matches
        # profit_iter = profit/100 * total_value
        acc_profit+=profit
        acc_total_bests+=total_matches
        total_rounds+=1
        print('Acc PROFIT:{} profit iter:{} total value:{} total rounds:{}'.format(acc_profit, profit, total_value, total_rounds), flush=True)
        nm=int(len(features)/2)
        end_test=min(start+duration_test_matches-1,nm-1)

    key_stats = tuple(config.values())
    print(key_stats)
    current_key_value = stats.get(key_stats, (0,0))
    stats[key_stats] = current_key_value[0] + acc_profit, current_key_value[1] + acc_total_bests

    with open('out.txt', 'a') as f:
        f.write("Profit: {} Plays:{} Config:{} iteration:{}".format(acc_profit, acc_total_bests, config, index_iteration) + '\n')
    print("stats runopt:{}".format(stats))
    if optimize:
        state['optimize_iteration'] = index_iteration
    state['stats'] = stats
    save_state()
    return end_test + 1,acc_profit,acc_total_bests


data=pd.read_csv("../Generated Data/atp_data.csv")
data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
data_copy=data.copy(deep=True)
# beg = datetime.datetime(2004,1,1)

# indices = data[(data.Date>beg) & (data['PSW']>threshold_prob_bet) & (data['PSL']>threshold_prob_bet)].index
# indices = data.index
# data = data.iloc[indices,:].reset_index(drop=True)
features=pd.read_csv("../Generated Data/atp_data_features.csv")
features_copy=features.copy(deep=True)

# total_interval_opts = timedelta(days=360)
total_interval_opts = timedelta(days=120)
total_iterations = 150
total_best_iterations = 5
top_pick = 10
total_repeat = 2

# start_simulation_date = datetime.datetime(2018,1,1)
start_simulation_date = datetime.datetime(2017,8,1)
# test_beginning_match=data[data.Date>=start_simulation_date].index[0]
test_beginning_match=None

total_profit = 0
total_games = 0
state = {}

stats = {}
optimize_iteration = 0
repeat_best_iteration = 0


def save_state():
    with open('state.pickle', 'wb') as f:
        pickle.dump(state, f)


try:
    with open('state.pickle', 'rb') as f:
        state = pickle.load(f)
        stats = state['stats']
        if 'optimize_iteration' in state:
            optimize_iteration = state['optimize_iteration']
        if 'repeat_best_iteration' in state:
            repeat_best_iteration = state['repeat_best_iteration']
        if 'test_beginning_match' in state:
            test_beginning_match = state['test_beginning_match']
        if 'total_profit' in state:
            total_profit = state['total_profit']
        if 'total_games' in state:
            total_games = state['total_games']
        print(f'recovered state with success:{state}')
except:
    print('not able to recover state basic')


while test_beginning_match is None or data.Date.iloc[test_beginning_match] < data.Date.iloc[-1]:

    # duration_val_matches_options = [2000]
    duration_val_matches_options = [1000]
    # duration_train_matches_options = [10400]
    duration_train_matches_options = [16000]
    # threshold_options = [(0.02, 0.01),(0.03, 0.02),(0.04, 0.03)]
    # threshold_options = [(0.10, 0.08),(0.15, 0.10),(0.20, 0.15)]
    # threshold_options = [(0.15, 0.10),(0.10, 0.08),(0.08, 0.06),(0.06, 0.04),(0.04, 0.02)]
    threshold_options = [(0.50, 0.40, 0.30),(0.60, 0.50, 0.40),(0.40, 0.30, 0.20)]
    # threshold_options = [(0.40, 0.25, 0.10), (0.25, 0.15, 0.10), (0.55, 0.35, 0.20), (0.75, 0.55, 0.40), (0.15, 0.10, 0.05)]

    # features_select_options = [tuple(["all"]),
    #                            tuple(["generalft, playerft"]),
    #                            tuple(["elo","cat_feature"]),
    #                            tuple(["playerft","duoft"]),
    #                            tuple(["cat_feature","playerft"]),
    #                            tuple(["elo","playerft"]),
    #                            tuple(["duoft","generalft"])
    #                            ]
    features_select_options = [tuple(["elo","playerft","generalft"])]

    # threshold_prob_bet_options = [1.4, 1.5, 1.6, 1.7]
    threshold_prob_bet_options = [1.79, 1.8]

    # learning_rate_options = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    # learning_rate_options = [0.35, 0.3, 0.40, 0.45]
    learning_rate_options = [0.35, 0.30, 0.40]
    # learning_rate_options = [0.30]
    # max_depth_options = [8, 9, 10]
    # max_depth_options = [4, 6, 8, 10]
    # max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    # max_depth_options = [6,7,8,9, 10, 11, 12]
    # max_depth_options = [7, 8, 9, 10]
    max_depth_options = [4, 6, 7, 8, 10]
    # early_stopping_rounds_options = [50, 100, 300]
    # early_stopping_rounds_options = [10, 15, 20]
    early_stopping_rounds_options = [10]
    # subsample_options = [0.25, 0.3, 0.35, 0.40, 0.45]
    # subsample_options = [0.35, 0.40, 0.25, 0.40, 0.45, 0.50, 0.55]
    # subsample_options = [0.55, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.65, 0.70]
    # subsample_options = [0.25, 0.01, 0.20, 0.15, 0.10, 0.05]
    subsample_options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    # colsample_bytree_options = [0.25, 0.3, 0.35, 0.40, 0.45]
    colsample_bytree_options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    # colsample_bytree_options = [0.45, 0.50, 0.55, 0.60, 0.40, 0.35, 0.30, 0.25, 0.20, 0.65, 0.70]
    # colsample_bytree_options = [0.45]
    mode_options = ['sum']
    # total_models_options = [50, 40, 60]
    # total_models_options = [70]
    total_models_options = [30]
    # total_models_selected_options = [10, 9, 8, 7]
    # total_models_selected_options = [20, 25, 30]
    # total_models_selected_options = [40, 30, 50]
    total_models_selected_options = [20]
    # total_models_options = [5, 7, 10]
    # min_trees_options = [5, 10, 20, 30, 40]
    # min_trees_options = [5, 15, 30, 50]
    # min_trees_options = [10, 20, 30]
    # min_trees_options = [20, 30, 40, 50]
    # min_trees_options = [1, 5, 10, 20, 30]
    min_trees_options = [2]

    factor_weight_options = [1, 3, 5, 10]


    def from_values_to_config(duration_train_matches,duration_val_matches,list_thresholds,learning_rate,max_depth,subsample,colsample_bytree,early_stop,total_models,total_models_selected,min_trees,factor_weight,threshold_prob_bet,features_select):
        return {
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
            "min_trees": min_trees,
            # "factor_weight": factor_weight,
            "threshold_prob_bet": threshold_prob_bet,
            "features_select":features_select
        }


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
        factor_weight = random.choice(factor_weight_options)
        threshold_prob_bet = random.choice(threshold_prob_bet_options)
        features_select = random.choice(features_select_options)

        config = from_values_to_config(duration_train_matches,duration_val_matches,list_thresholds,learning_rate,max_depth,subsample,colsample_bytree,early_stop,total_models,total_models_selected,min_trees,factor_weight,threshold_prob_bet,features_select)
        return config


    end_match = None
    params_config = [(gen_random_config(),test_beginning_match,index_iteration) for index_iteration in range(optimize_iteration, total_iterations)]
    # with Pool(cpu_count()-1) as pool:
    # with Pool(1) as pool:
    #     result = pool.starmap(run_opt, params_config)

    result = [run_opt(config, start_date_index, index_iteration) for config, start_date_index, index_iteration in params_config]

    for index_best_iteration in range(repeat_best_iteration, total_best_iterations):
        print("stats:{}".format(stats))
        list_stats = list(stats.items())
        list_stats.sort(key=lambda x: x[1][0]/x[1][1], reverse=True)
        best_config_list = [from_values_to_config(*x[0]) for x in list_stats[:top_pick]]
        print("Running best iteration with bests:{} {}".format(list_stats[:top_pick],index_best_iteration), flush=True)
        params_best = [(best_configs,test_beginning_match,index_best) for index_best,best_configs in enumerate(best_config_list)]
        result = [run_opt(config, start_date_index, index_iteration, optimize=False) for config, start_date_index, index_iteration in
                  params_best]
        state['repeat_best_iteration'] = index_best_iteration
        save_state()

    test_beginning_match = result[-1][0]
    list_stats = list(stats.items())
    list_stats.sort(key=lambda x: x[1][0]/x[1][1], reverse=True)
    best_config_list = [from_values_to_config(*x[0]) for x in list_stats[:top_pick]]
    print("Picking best 10 for test:{}".format(list_stats[:top_pick]), flush=True)
    params_best = [(best_configs,test_beginning_match,index_best_test) for index_best_test,best_configs in enumerate(best_config_list)]

    # with Pool(1) as pool:
    #     result = pool.map(run_opt, params_best)

    for index_total_repeat in range(total_repeat):
        result = [run_opt(config, start_date_index, index_iteration, optimize=False) for config, start_date_index, index_iteration in
                  params_best]

        for index, a_param_best in enumerate(params_best):
            profit_iteration = result[index][1]
            games_iteration = result[index][2]
            total_profit += profit_iteration
            total_games += games_iteration
            with open('log_profits.txt', 'a') as f:
                print("a param:{} a profit:{} games:{} profit acc:{} games acc:{} {} {}".format(a_param_best, profit_iteration,
                                                                                             games_iteration, total_profit,
                                                                                             total_games,
                                                                                             data.Date.iloc[test_beginning_match], index_total_repeat), file=f, flush=True)

    print("Reseting stats")
    stats = {}
    optimize_iteration = 0
    repeat_best_iteration = 0
    state['optimize_iteration'] = optimize_iteration
    state['repeat_best_iteration'] = repeat_best_iteration
    state['stats'] = stats
    state['test_beginning_match'] = test_beginning_match
    state['total_profit'] = total_profit
    state['total_games'] = total_games
    save_state()

    print("end date:{} profit acc:{} games acc:{}".format(data.Date.iloc[test_beginning_match], total_profit, total_games))

