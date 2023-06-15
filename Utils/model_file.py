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
    def __init__(self, target, model_type, region=None) -> None:
        self.Utils = utils_file.Utils_Class(target=target, model_type=model_type, cache_model=True, cache_scraping=True)
        self.last_game_date = self.Utils.TARGET_DF['Date'].max()
        self.region = region

#%%================Utils================###
    def get_team_players_cache(self, name):
        tempDf = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==name)
                                            & (self.Utils.team_data_table['Year'].astype(int)==CURRENT_YEAR)
                                            & (self.Utils.team_data_table['Semester'].astype(int)==CURRENT_SEMESTER)]
        
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

        feature_cols_filtered = [x for x in list(final_train_df.columns) 
                            if x.replace('Team_Blue_','').replace('Team_Red_','') in self.Utils.regions_feature_cols[self.region]]
        
        xdata = final_train_df[feature_cols_filtered]
        ydata = final_train_df[self.Utils.TARGET]

        return xdata, ydata, feature_cols_filtered

    def get_input_features(self, feature_cols_filtered):
        features_dict = {}
        for feature in feature_cols_filtered:

            if 'Blue' in feature:
                features_dict[feature],noDataNames = self.get_feature_team_mean(self.team_blue_names, feature)
                
            elif 'Red' in feature:
                features_dict[feature],noDataNames = self.get_feature_team_mean(self.team_red_names, feature)
        print(f'no data names on: {noDataNames}')

        return features_dict

#%%================Print================###

    def print_region_teams(self, region=None):
        if self.region == None:
            self.region = region
        elif region != None:
            region = self.region
        else: 
            raise Exception("Specify a region!")

        dfTemp = self.Utils.TARGET_DF[self.Utils.TARGET_DF['regionAbrev']==region]
        teamsSet = sorted(set(list(dfTemp['Blue'].unique())+list(dfTemp['Red'].unique())))
        teamsDict = dict(zip(range(len(teamsSet)),teamsSet))
        self.teams_dict = teamsDict
        print(teamsDict)

    def print_team_players(self):
        playerNames = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==self.teams_dict[self.blue_team])
                                                & (self.Utils.team_data_table['Year'].astype(int)==CURRENT_YEAR)
                                                & (self.Utils.team_data_table['Semester'].astype(int)==CURRENT_SEMESTER)][['TOP','JNG','MID','ADC','SUP']]

        try: print(f'team_blue_list = {list(playerNames.values[0])}')
        except: print('no team found')

        playerNames = self.Utils.team_data_table[(self.Utils.team_data_table['Name']==self.teams_dict[self.red_team])
                                                & (self.Utils.team_data_table['Year'].astype(int)==CURRENT_YEAR)
                                                & (self.Utils.team_data_table['Semester'].astype(int)==CURRENT_SEMESTER)][['TOP','JNG','MID','ADC','SUP']]

        try:print(f'team_red_list = {list(playerNames.values[0])}')
        except: print('no team found')

 #%%================Model================###
   
    def make_prediction(self, manual_insert=False, team_blue_list=None, team_red_list=None):
        print(f'last train data: {self.last_game_date}')

        self.team_blue_list = team_blue_list
        self.team_red_list = team_red_list
        self.manual_insert = manual_insert

        for i in range(1):
            print('=================')

            if self.manual_insert:
                self.team_blue_names = self.team_blue_list
                self.team_red_names = self.team_red_list
            else:
                self.team_blue_names = self.get_team_players_cache(self.teams_dict[self.blue_team])
                self.team_red_names = self.get_team_players_cache(self.teams_dict[self.red_team])

            xdata, ydata, feature_cols_filtered = self.get_train_data()
            features_dict = self.get_input_features(feature_cols_filtered)

            inputDf = pd.DataFrame(features_dict.values(),index=features_dict.keys()).transpose()
            for col in inputDf:
                inputDf[col].fillna(0, inplace=True)

            modelNum = (self.Utils.regions_stats[self.Utils.regions_stats['region']==self.region])['model'].iloc[0]
            model = self.Utils.BASE_MODELS[modelNum]
            model.fit(xdata,ydata)
            if self.Utils.MODEL_TYPE == 'binary':
                prediction = model.predict(inputDf)[0]
            elif self.Utils.MODEL_TYPE == 'logistic':
                prediction = model.predict_proba(inputDf)[:,1][0]

            #winner_name = self.teams_dict[self.blue_team] if round(prediction)==0 else self.teams_dict[self.red_team]
            #winner_name = 'Blue' if round(prediction)==0 else 'Red'
            print(prediction)
            #print(winner_name)