##=================================IMPORTS=================================##
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##=================================CLASSIFIERS=================================##
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

##=================================LOGISTICS=================================##
from sklearn.linear_model import LogisticRegression

##=================================REGRESSORS=================================##
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, OneClassSVM
from xgboost import XGBRegressor

##=================================NATIVES=================================##
import json
from utils.constants import *
import warnings

##=================================OPTIONS=================================##
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

##=================================CLASS=================================##

class Utils_Class:

    def __init__(self, target, year, semester, split_type, default_model, fill=True) -> None:
        self.TARGET = target
        self.FILL = fill
        self.DEFAULT_MODEL = default_model
        self.INFO_COLS = ['Date', 'tournamentId', self.TARGET, 'regionAbrev']

        self.CURRENT_YEAR = year
        self.CURRENT_SEMESTER = semester
        self.CURRENT_SEMESTER_YEAR = str(self.CURRENT_YEAR)+str(self.CURRENT_SEMESTER)
        self.LAST_SEMESTER = abs(self.CURRENT_SEMESTER-1)
        self.LAST_YEAR = self.CURRENT_YEAR-1 if self.LAST_SEMESTER==1 else self.CURRENT_YEAR
        self.LAST_SEMESTER_YEAR = str(self.LAST_YEAR)+str(self.LAST_SEMESTER)

        self.SPLIT_TYPE = split_type

        self.player_match_list, self.team_match_list = self.load_files(cache=False)
        self.TARGET_DF = self.team_match_list

    def define_models(self, model_type):
        if model_type=='binary':
            self.BASE_MODELS = [
                                RandomForestClassifier(), #0
                                KNeighborsClassifier(algorithm = 'brute'), #1
                                LinearSVC(C=0.0001), #2
                                BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10), #3
                                AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6), #4
                                DecisionTreeClassifier(), #5
                                LogisticRegression(), #6
                                LogisticRegression(solver='newton-cg') #7
                                ]
        elif model_type=='logistic':
            self.BASE_MODELS = [
                                LogisticRegression(solver='newton-cg') #0
                                ]
        elif model_type=='regression':
            self.BASE_MODELS = [
                                GradientBoostingRegressor(loss='absolute_error') #0
                                ,ElasticNet() #1
                                ,BayesianRidge() #2
                                ,LinearRegression() #3
                                ,SVR() #4
                                ,KernelRidge() #6
                                ,XGBRegressor() #7
                                ,RandomForestRegressor() #8
                                ]

    def load_files(self, cache=True):
        self.team_data_table = pd.read_pickle("data/raw_data/teamDataTable.pkl")
        self.player_data_table = pd.read_pickle("data/raw_data/playerDataTable.pkl")

        self.match_list = pd.read_pickle("data/raw_data/matchList.pkl")
        self.match_list_fill = pd.read_pickle("data/raw_data/matchListFill.pkl")

        if cache:
            team_match_list = pd.read_pickle("data/raw_data/teamMatchList.pkl")
            player_match_list = pd.read_pickle("data/raw_data/playerMatchList.pkl")
            self.regions_stats = pd.read_pickle("data/raw_data/regionsStats.pkl")

            with open(f'data/raw_data/regionsFeatureCols.json', 'r') as fp:
                self.regions_feature_cols = json.load(fp)
            with open(f'data/raw_data/regionsTrainData.json', 'r') as fp:
                self.regions_train_data = json.load(fp)
        else:
            if self.FILL:
                df_to_split = self.match_list_fill
            else:
                df_to_split = self.match_list

            player_match_list, team_match_list = self.split_match_list(df_to_split)
        
        return player_match_list, team_match_list

    def split_match_list(self, df):
        matchListDateFilter = (df[df['Date'] >= pd.to_datetime('2019-7-01',format='%Y-%m-%d')]
                                            .reset_index(drop=True).copy())
        
        playerMatchList = matchListDateFilter.copy()
        teamMatchList = matchListDateFilter.copy()

        for color in ['Blue','Red']:
            for feature in PLAYER_SIMPLE_FEATURE_COLS:
                    teamMatchList[f'Team_{color}_{feature}'] = (matchListDateFilter[[f"{position}_{color}_{feature}" 
                                                                                    for position in ROLES]]
                                                                                    .mean(skipna=True,axis=1).copy())
                    
                    teamMatchList.drop([f"{position}_{color}_{feature}" for position in ROLES],axis=1,inplace=True)
                    
            teamMatchList.drop([f"{position}_{color}" for position in ROLES],axis=1,inplace=True)

        return playerMatchList, teamMatchList
    
    def train_test_split_region(self, df_temp, tournament_id, verbose=True):
        
        if self.SPLIT_TYPE==0:
            testData = df_temp[df_temp['tournamentId']==tournament_id].copy()
            xtest= testData.drop(['Date',self.TARGET],axis=1).copy()
            xtest= xtest.drop(OFF_COLS,axis=1,errors='ignore')
            ytest = testData[self.TARGET]

            trainData = df_temp[df_temp['tournamentId']!=tournament_id].copy()
            xtrain = trainData.drop(['Date',self.TARGET],axis=1).copy()
            xtrain = xtrain.drop(OFF_COLS,axis=1,errors='ignore')
            ytrain = trainData[self.TARGET]
            
        elif self.SPLIT_TYPE==1:
            xCols = self.TARGET_DF.drop(['Date', self.TARGET]+OFF_COLS,axis=1,errors='ignore')
            yCols = self.TARGET_DF[self.TARGET]
            xtrain, xtest, ytrain, ytest = train_test_split(xCols, yCols, test_size=0.20, shuffle=False)
        
        if False:
            ytrain_mean, ytrain_std = np.mean(ytrain), np.std(ytrain)
            cut_off = ytrain_std * cut_off_var
            lower, upper = ytrain_mean - cut_off, ytrain_mean + cut_off
            outlierMask = ytrain.apply(lambda x: False if x < lower or x > upper else True)
            if verbose:
                print(f'train len: {len(xtrain)}')
            lentemp = len(xtrain)
            #xtrain, ytrain = xtrain[outlierMask], ytrain[outlierMask]
            if verbose:
                print(f'train len no outliers: {len(xtrain)}')
                print(f'percent of len removed: {round(abs(len(xtrain)/lentemp*100-100),2)}%')
                print(f'test len: {len(xtest)}\n')
        
        return xtrain, ytrain, xtest, ytest
    
    def generate_metric(self, model_number, region_feature_cols, region_data_list, tournament_id, reps):
        
        df_temp = self.TARGET_DF[self.TARGET_DF['regionAbrev'].isin(region_data_list)].copy()
        temp_cols = [x for x in list(df_temp.columns) if x.replace('Team_Red_','').replace('Team_Blue_','') in region_feature_cols]
        df_temp = df_temp[temp_cols+self.INFO_COLS]
        df_temp = df_temp.sort_values(by='Date',ascending=True).copy()
        
        xtrain, ytrain, xtest, ytest = self.train_test_split_region(df_temp, tournament_id, verbose=False)

        errors=0
        for i in range(reps):
            region_model = self.BASE_MODELS[model_number]
            region_model.fit(xtrain, ytrain)
            pred = region_model.predict(xtest)
            errors = accuracy_score(ytest, pred)+errors
        errors_final=errors/reps
        
        metric=round(abs(errors_final-1),3)
        
        return metric, pred

    def region_lists(self, min_entries=30):

        regions = self.TARGET_DF['regionAbrev'].unique()
        regions_to_feed = [x for x in self.TARGET_DF['regionAbrev'].unique()]
        regions_filter = ([x for x in regions if self.CURRENT_YEAR in (self.TARGET_DF[self.TARGET_DF['regionAbrev']==x])['realYear'].unique()
                                                and self.CURRENT_YEAR-1 in (self.TARGET_DF[self.TARGET_DF['regionAbrev']==x])['realYear'].unique()])
        regions_to_predict = []
        for region in regions_filter:
            regions_filter_size = len(self.TARGET_DF[(self.TARGET_DF['realYear']==self.CURRENT_YEAR) 
                                                     & (self.TARGET_DF['regionAbrev']==region)])
            if regions_filter_size>=min_entries:
                regions_to_predict.append(region)
        
        return regions_to_feed, regions_to_predict