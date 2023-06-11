########====================================================IMPORTS====================================================########
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
    
import json

from Utils.constants import *
import Utils.utils_file as utils_file

import warnings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

########====================================================CLASS====================================================########

class Final_Model_Class:
    def __init__(self, target, model_type) -> None:
        self.Utils = utils_file.Utils_Class(target=target, model_type=model_type,cache_model=True, cache_scraping=True)

#%%================Utils================###
    def get_team_players_cache(self, name):
        tempDf = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==name)
                                            & (self.Utils.team_data_table['Year'].astype(int)==self.Utils.LAST_YEAR)
                                            & (self.Utils.team_data_table['Semester'].astype(int)==self.Utils.LAST_SEMESTER)]
        
        namesList = tempDf[['TOP','JNG','MID','ADC','SUP']].iloc[0]
        
        return namesList

    def get_feature_team_mean(self, namesList, feature):
        values=[]
        noDataList=[]
        for name in namesList:
            filteredTempDf = self.Utils.player_data_table[(self.Utils.player_data_table['Player']==name)
                                                        & (self.Utils.player_data_table['Year']==self.Utils.LAST_YEAR)
                                                        & (self.Utils.player_data_table['Semester']==self.Utils.LAST_SEMESTER)]
            
            valueToAppend = filteredTempDf[feature.replace('Team_Red_','').replace('Team_Blue_','')]
            if len(valueToAppend)>0:
                values.append(valueToAppend.iloc[0])
            else: 
                noDataList.append(name)
        
        return np.mean(values), noDataList
    
    def get_train_data(self):
        final_train_df = self.Utils.TARGET_DF[self.Utils.TARGET_DF['regionAbrev'].isin(self.Utils.regions_train_data[self.region])].copy()
        final_train_df = final_train_df.sort_values(by='Date',ascending=True)
        self.last_game_date = final_train_df['Date'].max()

        feature_cols_filtered = [x for x in list(final_train_df.columns) 
                            if x.replace('Team_Blue_','').replace('Team_Red_','') in self.Utils.regions_feature_cols[self.region]]
        
        xdata = final_train_df[feature_cols_filtered]
        ydata = final_train_df[self.Utils.TARGET]

        return xdata, ydata, feature_cols_filtered

    def get_input_features(self, feature_cols_filtered):
        features_dict = {}
        for feature in feature_cols_filtered:
            side = feature.split('_')[1]
            if side == 'Blue':
                features_dict[feature],noDataNames = self.get_feature_team_mean(self.team_blue_names, feature)
                
            elif side == 'Red':
                features_dict[feature],noDataNames = self.get_feature_team_mean(self.team_red_names, feature)
        print(f'no data names on: {noDataNames}')
        print(f'last train data: {self.last_game_date}')

        return features_dict

#%%================Print================###

    def print_region_teams(self, region:str):
        self.region = region
        dfTemp = self.Utils.TARGET_DF[self.Utils.TARGET_DF['regionAbrev']==region]
        teamsSet = sorted(set(list(dfTemp['Blue'].unique())+list(dfTemp['Red'].unique())))
        teamsDict = dict(zip(range(len(teamsSet)),teamsSet))
        self.teams_dict = teamsDict
        print(teamsDict)

    def print_team_players(self):
        playerNames = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==self.teams_dict[self.blue_team])
                                                & (self.Utils.team_data_table['Year'].astype(int)==self.Utils.LAST_YEAR)
                                                & (self.Utils.team_data_table['Semester'].astype(int)==self.Utils.LAST_SEMESTER)][['TOP','JNG','MID','ADC','SUP']]

        try: print(f'team blue players = {list(playerNames.values[0])}')
        except: print('no team found')

        playerNames = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==self.teams_dict[self.red_team])
                                                & (self.Utils.team_data_table['Year'].astype(int)==self.Utils.LAST_YEAR)
                                                & (self.Utils.team_data_table['Semester'].astype(int)==self.Utils.LAST_SEMESTER)][['TOP','JNG','MID','ADC','SUP']]

        try:print(f'team red players = {list(playerNames.values[0])}')
        except: print('no team found')

 #%%================Model================###
   
    def make_prediction(self, manual_insert=False, team_blue_list=None, team_red_list=None):
        self.team_blue_list = team_blue_list
        self.team_red_list = team_red_list
        self.manual_insert = manual_insert

        for i in range(5):
            print('=================')

            #set player names
            if self.manual_insert:
                self.team_blue_names = self.team_blue_list
                self.team_red_names = self.team_red_list
            else:
                self.team_blue_names = self.get_team_players_cache(self.teams_dict[self.blue_team])
                self.team_red_names = self.get_team_players_cache(self.teams_dict[self.red_team])

            #generate train data
            xdata, ydata, feature_cols_filtered = self.get_train_data()

            #generate input df
            features_dict = self.get_input_features(feature_cols_filtered)

            inputDf = pd.DataFrame(features_dict.values(),index=features_dict.keys()).transpose()
            for col in inputDf:
                inputDf[col].fillna(0, inplace=True)

            #model and prediction
            modelNum = (self.Utils.regions_stats[self.Utils.regions_stats['region']==self.region])['model'].iloc[0]
            model = self.Utils.BASE_MODELS[modelNum]
            model.fit(xdata,ydata)
            prediction = self.teams_dict[self.blue_team] if model.predict(inputDf)==0 else self.teams_dict[self.red_team]
            print(prediction)