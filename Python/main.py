#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime,timedelta

from past_features import *
from elo_features import *
from categorical_features import *
from stategy_assessment import *
from utilities import *



################################################################################
######################### Building of the raw dataset ##########################
################################################################################

### Importation of the Excel files - 1 per year (from tennis.co.uk)
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Bet365 and Pinnacle.

import glob

genders = ['men', 'women']

match_id_start = 0

for gender_dir in genders:
# gender_dir = "women"
    filenames=list(glob.glob(f"../Data/{gender_dir}/20*.xls*"))
    l = [pd.read_excel(filename,encoding='latin-1') for filename in filenames]
    no_b365=[i for i,d in enumerate(l) if "B365W" not in l[i].columns]
    no_pi=[i for i,d in enumerate(l) if "PSW" not in l[i].columns]
    for i in no_pi:
        l[i]["PSW"]=np.nan
        l[i]["PSL"]=np.nan
    for i in no_b365:
        l[i]["B365W"]=np.nan
        l[i]["B365L"]=np.nan
    l=[d[list(d.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]] for d in [l[0]]+l[2:]]
    data=pd.concat(l,0)

    ### Data cleaning
    data=data.sort_values("Date")
    data=data[data['Date'].notnull()]
    # print(data['Date'][-200:])

    data["WRank"]=data["WRank"].replace(np.nan,0)
    data["WRank"]=data["WRank"].replace("NR",2000)
    data["LRank"]=data["LRank"].replace(np.nan,0)
    data["LRank"]=data["LRank"].replace("NR",2000)
    data["WRank"]=data["WRank"].astype(int)
    data["LRank"]=data["LRank"].astype(int)
    data["Wsets"]=data["Wsets"].astype(float)
    data["Lsets"]=data["Lsets"].replace("`1",1)
    data["Lsets"]=data["Lsets"].astype(float)
    data["Winner"]= data["Winner"].apply(lambda x: x.strip())
    data["Loser"] = data["Loser"].apply(lambda x: x.strip())

    data=data.reset_index()
    ### Elo rankings data
    # Computing of the elo ranking of each player at the beginning of each match.
    elo_rankings = compute_elo_rankings(data)
    data = pd.concat([data,elo_rankings],1)
    data['matchid'] = data.index + match_id_start
    # data.index.names = ['matchid']
    data.to_csv(f"../Generated Data/{gender_dir}/atp_data.csv",index=False)

    ################################################################################
    ######################## Building training set #################################
    ################################################################################
    ### We'll add some features to the dataset

    data=pd.read_csv(f"../Generated Data/{gender_dir}/atp_data.csv")
    elo_rankings = data[["elo_winner","elo_loser","proba_elo"]]
    elo_1 = elo_rankings
    elo_2 = elo_1[["elo_loser","elo_winner","proba_elo"]]
    elo_2.columns = ["elo_winner","elo_loser","proba_elo"]
    elo_2.proba_elo = 1-elo_2.proba_elo
    elo_2.index = range(1,2*len(elo_1),2)
    elo_1.index = range(0,2*len(elo_1),2)
    features_elo_ranking = pd.concat([elo_1,elo_2]).sort_index(kind='merge')

    print(data[-200:])
    data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

    # data = data.iloc[indices,:].reset_index(drop=True)

    ######################### The period that interests us #########################

    # threshold_prob_bet = 1.5

    # beg = datetime.datetime(2004,1,1)
    # end = data.Date.iloc[-1]
    indices = data.index

    ################### Building of some features based on the past ################

    features_player  = features_past_generation(features_player_creation,180,"playerft5",data,indices)

    label_player  =    features_past_generation(label_creation,1,"label",data,indices)
    # features_duo     = features_past_generation(features_duo_creation,720,"duoft",data,indices)
    features_general = features_past_generation(features_general_creation,60,"generalft",data,indices)
    # dump(player_features,"player_features")
    # dump(duo_features,"duo_features")
    # dump(general_features,"general_features")
    # dump(recent_features,"recent_features")
    # features_player=load("player_features")
    # features_duo=load("duo_features")
    # features_general=load("general_features")
    # features_recent=load("recent_features")

    ########################### Selection of our period ############################

    # data = data.iloc[indices,:].reset_index(drop=True)
    odds = data[["PSW","PSL"]]

    ########################## Encoding of categorical features ####################

    features_categorical = data[["Surface","Round"]]
    features_categorical_encoded = categorical_features_encoding(features_categorical)
    players_encoded = features_players_encoding(data)
    tournaments_encoded = features_tournaments_encoding(data)
    features_onehot = pd.concat([features_categorical_encoded],1)


    ############################### Duplication of rows ############################
    ## For the moment we have one row per match.
    ## We "duplicate" each row to have one row for each outcome of each match.
    ## Of course it isn't a simple duplication of  each row, we need to "invert" some features

    # Elo data


    # Categorical features
    features_onehot = pd.DataFrame(np.repeat(features_onehot.values,2, axis=0),columns=features_onehot.columns)


    # Date features
    features_date = pd.DataFrame(np.repeat(data.Date.values,2, axis=0),columns=["Date"])
    features_matchid  =  pd.DataFrame(np.repeat(data['matchid'].values,2, axis=0),columns=["matchid0"])

    # odds feature
    features_odds = pd.Series(odds.values.flatten(),name="odds")
    features_odds = pd.DataFrame(features_odds)

    # Date feature
    # features_date = pd.Series(data['Date'],name="Date").repeat(2)
    # features_date = pd.DataFrame(features_date)
    # features_date = features_date.reindex()
    # print(f"features date:{features_date.shape}")

    ### Building of the final dataset
    # You can remove some features to see the effect on the ROI
    features = pd.concat([features_matchid,
                          features_odds,
                          features_date,
                        features_elo_ranking,
                      features_onehot,
                      features_player,
    #                  features_duo,
                      features_general,
                      label_player],1)

    # features = pd.concat([features_matchid,features_date,features_odds],1)

    features.to_csv(f"../Generated Data/{gender_dir}/atp_data_features.csv",index=False)
    match_id_start = data.iloc[-1]['matchid'] + 1

all_data_columns = ["matchid", "Winner", "Loser", "PSW", "PSL", "Date", "Tournament"]
all_data=pd.DataFrame(columns=all_data_columns)
all_data_features=pd.DataFrame(columns=features.columns)
for gender_dir in genders:
    data=pd.read_csv(f"../Generated Data/{gender_dir}/atp_data.csv")
    features=pd.read_csv(f"../Generated Data/{gender_dir}/atp_data_features.csv")
    all_data=all_data.append(data[all_data_columns], ignore_index = True)
    all_data_features=all_data_features.append(features, ignore_index = True)

print(all_data[-100:])
print(all_data_features[-100:])
print(all_data.columns)
all_data=all_data.sort_values(["Date","matchid"])
all_data_features=all_data_features.sort_values(["Date", "matchid0"])

all_data_features.to_csv(f"../Generated Data/atp_data_features.csv",index=False)
all_data.to_csv(f"../Generated Data/atp_data.csv",index=False)
