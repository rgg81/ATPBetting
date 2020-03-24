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
filenames=list(glob.glob("../Data/20*.xls*"))
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

data.reset_index(inplace=True)
data.index.names = ['matchid']

data.to_csv("../Generated Data/atp_data.csv",index=True)

### Elo rankings data
# Computing of the elo ranking of each player at the beginning of each match.
elo_rankings = compute_elo_rankings(data)
data = pd.concat([data,elo_rankings],1)

################################################################################
######################## Building training set #################################
################################################################################
### We'll add some features to the dataset

data=pd.read_csv("../Generated Data/atp_data.csv")
data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

data.reset_index(inplace=True)

######################### The period that interests us #########################

# threshold_prob_bet = 1.5

# beg = datetime.datetime(2004,1,1)
# end = data.Date.iloc[-1]
indices = data.index

################### Building of some features based on the past ################

features_player  = features_past_generation(features_player_creation,180,"playerft5",data,indices)
features_matchid  =    features_past_generation(index_match_creation,1,"matchid",data,indices)
label_player  =    features_past_generation(label_creation,1,"label",data,indices)
features_duo     = features_past_generation(features_duo_creation,720,"duoft",data,indices)
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

data = data.iloc[indices,:].reset_index(drop=True)
odds = data[["PSW","PSL"]]

########################## Encoding of categorical features ####################

features_categorical = data[["Surface","Round"]]
features_categorical_encoded = categorical_features_encoding(features_categorical)
# players_encoded = features_players_encoding(data)
# tournaments_encoded = features_tournaments_encoding(data)
features_onehot = pd.concat([features_categorical_encoded],1)


############################### Duplication of rows ############################
## For the moment we have one row per match. 
## We "duplicate" each row to have one row for each outcome of each match. 
## Of course it isn't a simple duplication of  each row, we need to "invert" some features

# Elo data
elo_rankings = data[["elo_winner","elo_loser","proba_elo"]]
elo_1 = elo_rankings
elo_2 = elo_1[["elo_loser","elo_winner","proba_elo"]]
elo_2.columns = ["elo_winner","elo_loser","proba_elo"]
elo_2.proba_elo = 1-elo_2.proba_elo
elo_2.index = range(1,2*len(elo_1),2)
elo_1.index = range(0,2*len(elo_1),2)
features_elo_ranking = pd.concat([elo_1,elo_2]).sort_index(kind='merge')

# Categorical features
features_onehot = pd.DataFrame(np.repeat(features_onehot.values,2, axis=0),columns=features_onehot.columns)

# odds feature
features_odds = pd.Series(odds.values.flatten(),name="odds")
features_odds = pd.DataFrame(features_odds)

### Building of the final dataset
# You can remove some features to see the effect on the ROI
features = pd.concat([features_matchid, features_odds,
                  features_elo_ranking,
                  features_onehot,
                  features_player,
                  features_duo,
                  features_general,
                  label_player],1)

features.to_csv("../Generated Data/atp_data_features.csv",index=False)