##=================================IMPORTS=================================##
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##=================================CLASSIFIERS=================================##
import sklearn.metrics as skm
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
import warnings

from Utils.constants import *
##=================================OPTIONS=================================##
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

##=================================CLASS=================================##

class Utils_Class:

    def __init__(self, year=CURRENT_YEAR, semester=CURRENT_SEMESTER, target=None, split_type=0
                 , default_model=None, model_type=None, fill=True, cache_model=False, cache_scraping=False) -> None:
        
        self.TARGET = target
        self.FILL = fill
        self.DEFAULT_MODEL = default_model
        self.INFO_COLS = ['Date', 'tournamentId', self.TARGET, 'regionAbrev', 'realSemesterYear', 'semesterYear']
        self.MODEL_TYPE = model_type

        self.YEAR = year
        self.SEMESTER = semester
        self.SEMESTER_YEAR = str(self.YEAR)+str(self.SEMESTER)
        self.SEASON_NUM = SY_TO_SN[self.SEMESTER_YEAR]

        self.LAST_SEMESTER = abs(self.SEMESTER-1)
        self.LAST_YEAR = self.YEAR-1 if self.LAST_SEMESTER==1 else self.YEAR
        self.LAST_SEMESTER_YEAR = str(self.LAST_YEAR)+str(self.LAST_SEMESTER)

        self.SPLIT_TYPE = split_type

        self.cache_model = cache_model
        self.cache_scraping = cache_scraping

        self.load_files()
        self.define_models(model_type)

        if self.cache_scraping:
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
                                LogisticRegression(solver='newton-cg', penalty=None) #0
                                ,LogisticRegression(solver='newton-cg') #1
                                ,LogisticRegression() #2
                                ,LogisticRegression(penalty=None) #3
                                ,LogisticRegression(solver='liblinear') #4
                                ,LogisticRegression(solver='liblinear', penalty='l1') #5
                                ,LogisticRegression(solver='newton-cholesky') #6
                                ,LogisticRegression(solver='newton-cholesky', penalty=None) #7
                                ,LogisticRegression(solver='sag') #8
                                ,LogisticRegression(solver='sag', penalty=None) #9
                                ,LogisticRegression(solver='saga') #9
                                #,LogisticRegression(solver='saga', penalty='elasticnet') #10
                                ,LogisticRegression(solver='saga', penalty='l1') #11
                                ,LogisticRegression(solver='saga', penalty=None) #12
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

    def load_files(self):

        tournaments_2023 = open("Data/raw_data/tournaments_2023.txt", "r").read().split('\n')
        tournaments_2022 = open("Data/raw_data/tournaments_2022.txt", "r").read().split('\n')
        tournaments_2021 = open("Data/raw_data/tournaments_2021.txt", "r").read().split('\n')
        tournaments_2020 = open("Data/raw_data/tournaments_2020.txt", "r").read().split('\n')
        tournaments_2019 = open("Data/raw_data/tournaments_2019.txt", "r").read().split('\n')
        tournaments_list = [tournaments_2023,tournaments_2022,tournaments_2021,tournaments_2020,tournaments_2019]
        self.all_tournaments = []
        for tournament in tournaments_list:
            self.all_tournaments.extend(tournament)

        with open(f'Data/cache/regions_cache.json', 'r') as fp:
            self.regions_cache = json.load(fp)

        if self.MODEL_TYPE!=None:
            try:
                self.regions_feature_cols = self.regions_cache[str([CURRENT_YEAR_SEMESTER, 'feature_cols', self.MODEL_TYPE])]
                self.regions_train_data = self.regions_cache[str([CURRENT_YEAR_SEMESTER, 'train_data', self.MODEL_TYPE])]
            except:
                print(f'No data found for season {CURRENT_YEAR_SEMESTER} and model {self.MODEL_TYPE}')
            
        if self.cache_scraping:
            self.team_data_table = pd.read_pickle("Data/treated_data/team_data_table.pkl")
            self.player_data_table = pd.read_pickle("Data/treated_data/player_data_table.pkl")

            self.match_list = pd.read_pickle("Data/treated_data/match_list.pkl")
            self.match_list_fill = pd.read_pickle("Data/treated_data/match_list_fill.pkl")

            if self.FILL:
                df_to_split = self.match_list_fill
            else:
                df_to_split = self.match_list

            self.player_match_list, self.team_match_list = self.split_match_list(df_to_split)

            if self.cache_model:
                try:
                    self.regions_stats = pd.read_pickle(F"Data/cache/regions_stats_{CURRENT_YEAR_SEMESTER}_{self.MODEL_TYPE}.pkl")
                except:
                    print('Cannot read regions_stats')

    def split_match_list(self, df):
        # matchListDateFilter = (df[df['Date'] >= pd.to_datetime('2019-7-01',format='%Y-%m-%d')]
        #                                     .reset_index(drop=True).copy())
        matchListDateFilter = df.copy()
        
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
    
    def train_test_split_region(self, df_temp, region, verbose=True):
        if self.SPLIT_TYPE==0:
            testData = df_temp[(df_temp['regionAbrev']==region)
                               & (df_temp['realSemesterYear'].isin([SN_TO_SY[self.SEASON_NUM-4], SN_TO_SY[self.SEASON_NUM-3]
                                                                    , SN_TO_SY[self.SEASON_NUM-2], SN_TO_SY[self.SEASON_NUM-1]]))].copy()

            xtest= testData.drop(['Date',self.TARGET],axis=1).copy()
            xtest= xtest.drop(OFF_COLS,axis=1,errors='ignore')
            ytest = testData[self.TARGET]

            trainData = df_temp[((df_temp['regionAbrev']!=region)
                               & (df_temp['realSemesterYear'].astype(int) < int(SN_TO_SY[self.SEASON_NUM-1])))
                               |
                               ((df_temp['regionAbrev']==region)
                               & (df_temp['realSemesterYear'].astype(int) < int(SN_TO_SY[self.SEASON_NUM-4])))].copy()
            
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
    
    def make_pred(self, model_number, reps, xtrain, ytrain, xtest, ytest):

        errors=0
        if self.MODEL_TYPE == 'logistic':
            threshold = self.logistic_threshold

        region_model = self.BASE_MODELS[model_number]
        
        if self.MODEL_TYPE=='binary':
            for i in range(reps):
                region_model.fit(xtrain, ytrain)
                pred = region_model.predict(xtest)
                errors = accuracy_score(ytest, pred)+errors
            errors_final=errors/reps
            metric=round(abs(errors_final-1),3)

        elif self.MODEL_TYPE=='logistic':
            for i in range(reps):
                region_model.fit(xtrain, ytrain)
                pred = region_model.predict_proba(xtest)
                pred = pred[:,1]
                prediction_df = pd.DataFrame([pred, ytest]).transpose()
                oldlen = len(prediction_df)
                prediction_df = prediction_df[(prediction_df[0]>abs(threshold-1))
                                                | (prediction_df[0]<threshold)].reset_index(drop=True)
                prediction_df = prediction_df.round()
                newlen = len(prediction_df)
                try:
                    errors = skm.mean_absolute_error(prediction_df[0], prediction_df[1])+errors
                except:
                    errors = 1+errors
                    print('no data left after filtering')
#%%
            # df_temp = pd.DataFrame(dict({'predicted':pred,'true':ytest}))
            # perc_val_df = df_temp

            # perc_val_df['round'] = perc_val_df['predicted'].round().astype(int)
            # perc_val_df['result'] = (perc_val_df['true'] == perc_val_df['round']).replace({True:1,False:0})
            # perc_val_df['predicted'] = perc_val_df['predicted'].apply(lambda x: 1-x if x<0.5 else x)
            # perc_val_result_df = pd.DataFrame()
            # for x in np.arange(0.5,0.9,0.1):
            #     threshold2 = round(x,2)
            #     test_df = perc_val_df[(perc_val_df['predicted']>=threshold2)
            #                                 & (perc_val_df['predicted']<=threshold2+0.1)]
            #     if len(test_df.result.unique())>1:
            #         result = round(test_df.result.value_counts()[1]/len(test_df),2)
            #         perc_val_result_df = perc_val_result_df.append(pd.Series([round(threshold2,2), result, len(test_df)]), ignore_index=True)

            # perc_val_result_df.columns = ['threshold','result','len']
            # perc_val_result_df['diff'] = perc_val_result_df['result'] - perc_val_result_df['threshold']
            # len_bad_diffs = (perc_val_result_df[perc_val_result_df['diff']<-0.05])['diff']
            #print(f'bad diff percentage: {len(len_bad_diffs)/len(perc_val_result_df)}')
            #print(f'bad diff mean: {round(np.mean(len_bad_diffs),2)}')
#%%

            self.len_lost = round((oldlen-newlen)/oldlen,2)
            errors_final=errors/reps
            metric=round(abs(errors_final),3)

        elif self.MODEL_TYPE=='regression':
            for i in range(reps):
                pass

        return metric, pred
    
    def generate_metric(self, model_number, region_feature_cols, region_data_list, region, reps):
        
        df_temp = self.TARGET_DF[self.TARGET_DF['regionAbrev'].isin(region_data_list)].copy()
        #df_temp = df_temp[df_temp['is_playoffs']==False]
        temp_cols = [x for x in list(df_temp.columns) if x.replace('Team_Red_','').replace('Team_Blue_','') in region_feature_cols]
        df_temp = df_temp[temp_cols+self.INFO_COLS]
        
        xtrain, ytrain, xtest, ytest = self.train_test_split_region(df_temp, region, verbose=False)
        self.train_len = len(xtrain)

        if len(ytrain.unique())>1:
            metric, pred = self.make_pred(model_number, reps, xtrain, ytrain, xtest, ytest)
        else:
            metric = 1
            pred = []
            print('only one class found')
            print(f'ytrain len: {len(ytrain)}')
            print(f'list of data used: {region_data_list}')
        
        return metric, pred, ytest

    def region_lists(self):

       regions_df = self.TARGET_DF.copy()

       regions_to_feed = regions_df[regions_df['realSemesterYear'].astype(int) < int(self.LAST_SEMESTER_YEAR)].regionAbrev.unique()
       regions_to_feed = [x for x in regions_to_feed]
       
       regions_to_predict = regions_df[regions_df['realSemesterYear'].astype(int) == int(self.LAST_SEMESTER_YEAR)].regionAbrev.unique()
       regions_to_predict = [x for x in regions_to_predict]

       
       return regions_to_feed, regions_to_predict
    
    def save_model_cache(self, regions_stats, regions_feature_cols, regions_train_data):

        regions_cache = self.regions_cache
        regions_cache[str([CURRENT_YEAR_SEMESTER, 'feature_cols', self.MODEL_TYPE])] = regions_feature_cols
        regions_cache[str([CURRENT_YEAR_SEMESTER, 'train_data', self.MODEL_TYPE])] = regions_train_data

        with open(f'Data/cache/regions_cache.json', 'w') as fp:
            json.dump(regions_cache, fp)

        regions_stats.to_pickle(f"Data/cache/regions_stats_{CURRENT_YEAR_SEMESTER}_{self.MODEL_TYPE}.pkl")

        print('Cache saved!')

    def bet_ratio_vars(self, result=None, ratio=None, chance=None):
        if result==None:
            result = (chance * (ratio-1)) - (1-chance)

            print(f'Expected result: {result}')

        elif ratio==None:
            ratio = ((result + (1 - chance)) / chance) + 1
            print(f'Expected ratio: {ratio}')
        
        elif chance==None:
            chance_range = np.arange(0, 1, 0.05)
            true_range = [(chance * (ratio-1)) - (1-chance) > 0 for chance in chance_range]
            range_df = pd.DataFrame([chance_range,true_range])
            range_df = range_df[range_df.columns[range_df.iloc[1]==True]]

            min_chance = round(range_df.iloc[0].min(),3)
            result_min = round((min_chance * (ratio-1)) - (1-min_chance),3)

            print(f'Min chance: {min_chance}')
            print(f'Result on min chance: {result_min}')
            #print(f'Range df:\n{range_df}')