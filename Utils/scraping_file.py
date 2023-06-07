########====================================================IMPORTS====================================================########
from Utils.constants import *
from Utils.utils_file import Utils_Class

import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

headers = requests.utils.default_headers()
headers.update({"User-Agent": "Chrome/51.0.2704.103"})

########====================================================CLASS====================================================########

class Scraping_Class:

    def __init__(self, year=CURRENT_YEAR, semester=CURRENT_SEMESTER, Utils=Utils_Class(), cache=True) -> None:
        self.current_year = year
        self.current_semester = 'Summer' if semester==1 else 'Spring'
        self.all_tournaments = Utils.all_tournaments
        self.Utils = Utils

        if cache:
            self.player_data_table = self.Utils.player_data_table
            self.team_data_table = self.Utils.team_data_table
            self.match_list = self.Utils.match_list
    
#%%================Scraping Utils================###
    def team_tournament_find(self, code,split):
        page = requests.get(f'https://gol.gg/teams/team-matchlist/{code}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def player_tournament_find(self, code,season,split):
        page = requests.get(f'https://gol.gg/players/player-matchlist/{code}/season-{season}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def get_table(self, url,mult=False):
        page = requests.get(url,headers=headers)
        table = pd.read_html(page.text)
        table = [x for x in table if len(x)>1]
        if mult==False:
            if len(table)>0: 
                return table[0]
            else:
                return []
        else: 
            return table
        
    def get_match_stats(self, code):
        page = requests.get(f'https://gol.gg/game/stats/{code}/page-game/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        redScore = int(bs.find_all('span',class_='score-box red_line')[0].text)
        blueScore = int(bs.find_all('span',class_='score-box blue_line')[0].text)
        allScore = redScore+blueScore
        
        return [blueScore, redScore, allScore]

    def score_select(self, score):
        blueScore = int(score[0])
        redScore = int(score[-1])
        
        finalScore = blueScore-redScore
        
        if finalScore < 0:
            return 1
        elif finalScore > 0:
            return 0
        else:
            return 2

    def find_region_tournament(self, tournament):
        tournamentSplit = tournament.split('%20')
        
        if tournamentSplit[0] not in ['Spring','Summer']:
            region = tournamentSplit[0]
            
        elif tournamentSplit[1] not in ['Spring','Summer']:
            region = tournamentSplit[1]
        
        if region == 'LCK-LPL-LMS-VCS':
            region = 'Asia'
            
        if any(y in tournamentSplit for y in ['Finals','Playoffs']) in tournamentSplit:
            region = region+'_Playoffs'

        if any(y in tournamentSplit for y in ['Proving','Div','Academy','Hitpoint','Elite']):
            region = region+'_Tier2'
            
        return region

#%%================Transforming Utils================###
    def fill_feature_team(self, df, index, role, feature, side):
        value = df[role+'_'+feature][index]
        
        if np.isnan(value):
            roleSide = [x for x in ROLE_SIDE_COLS if side in x]
            columnsToMean = [col+'_'+feature for col in roleSide]
            listToMean = df[columnsToMean].loc[index].dropna()
            newValue = np.mean(listToMean)
            
            return newValue
        else:
            return value
    
    def fill_feature_player(self, df, index, role, feature, side):
        value = df[role+'_'+side+'_'+feature][index]
        name = df[role+'_'+side][index]
        year = df['Year'][index]
        semester = df['Semester'][index]
        
        if np.isnan(value):
            playerDfFilter = (self.player_data_table[self.player_data_table['Player']==name])[['Player','Semester','Year',feature]]
            if semester==0:
                playerDfFilter = playerDfFilter[(playerDfFilter['Year']<=year) 
                                                & ((playerDfFilter['Year']>=year-1)
                                                & (playerDfFilter['Semester']==1))]
            else:
                playerDfFilter = playerDfFilter[(playerDfFilter['Year']>=year) 
                                                & ((playerDfFilter['Year']<=year+1)
                                                & (playerDfFilter['Semester']==0))]
            newValue = np.mean(playerDfFilter[feature].tail(3))
            if not np.isnan(newValue):
                self.cont.append(0)
            print(f'nan vals removed: {len(self.cont)} index number: {round(index/len(df)*100,2)}%',end='\r')
                
            return newValue
        else:
            return value
        
    def transforming_player(self, df):
        for col in df:
            df[col] = df[col].apply(lambda x: str(x).replace('%',''))
            df[col] = df[col].apply(lambda x: np.nan if ('-' in x and len(x)<2) else x )

        df[PLAYER_FLOAT_COLS] = df[PLAYER_FLOAT_COLS].astype(float)
        df[PLAYER_INT_COLS] = df[PLAYER_INT_COLS].fillna(0).astype(int)
        df.columns = [x.replace(' ','_') for x in df.columns]

        self.player_data_table = df
        return df
    
    def transforming_team(self, df):
        for col in df:
            df[col] = df[col].apply(lambda x: str(x).replace('%',''))
            df[col] = df[col].apply(lambda x: np.nan if ('-' in x and len(x)<2) else x )

        df[TEAM_FLOAT_COLS] = df[TEAM_FLOAT_COLS].astype(float)
        df[TEAM_INT_COLS] = df[TEAM_INT_COLS].fillna(0).astype(int)

        df['Game duration'] = pd.to_datetime(df['Game duration'], format='%H:%M:%S').dt.time
        df.columns = [x.replace(' ','_') for x in df.columns]

        df[ROLES] = 'sNaN'
        for i in range(len(df)):
            
            code = df['teamCode'][i]
            split = df['Split'][i]
            page = requests.get(f'https://gol.gg/teams/team-stats/{code}/split-{split}/tournament-ALL/',headers=headers)
            tables = pd.read_html(page.text)[-1]
            
            allNames = tables['Player'][1:6]
            for name,val in zip(ROLES,allNames):
                df[name][i] = val

        self.team_data_table = df
        return df
    
    def transforming_match(self, df):
        df['StatsTemp'] = df['matchCode'].apply(lambda x: self.get_match_stats(x))
        df['blueKills'] = df['StatsTemp'].apply(lambda x: x[0])
        df['redKills'] = df['StatsTemp'].apply(lambda x: x[1])
        df['totalKills'] = df['StatsTemp'].apply(lambda x: x[2])

        matchListToDrop=['StatsTemp','matchCodePre']
        for col in matchListToDrop:
            df.drop(col, axis=1, errors='ignore', inplace=True)

        df['Score'] = df['Score'].apply(lambda x: self.score_select(x))

        self.match_list = df
        return df
    
    def season_data_swap(self, df):
        
        df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x,format='%Y-%m-%d'))
        df['Semester'] = df['Date'].apply(lambda x: 0 if x.month <= 6 else 1)
        df['Split'] = df['Date'].apply(lambda x: 'Spring' if x.month <= 6 else 'Summer')
        df['Year'] = df['Date'].apply(lambda x: x.year)

        df['realSemester'] = df['Semester']
        df['realYear'] = df['Year']
        df['realSemesterYear'] = (df['realYear'].astype(str) + df['realSemester'].astype(str))

        df['Semester'] = df['Semester'].apply(lambda x: 0 if x==1 else 1)
        df['Year'] = df['Year'] - df['Semester']
        df['semesterYear'] = (df['Year'].astype(str) + df['Semester'].astype(str))

        df['regionAbrev'] = df['Tournament'].apply(lambda x: self.find_region_tournament(x))
        df['tournamentId'] = (df['regionAbrev'].astype(str) + df['realSemesterYear'].astype(str))

        for role in ROLE_SIDE_COLS:

            tempCols = [role+'_'+col for col in PLAYER_SIMPLE_FEATURE_COLS]
            df.drop(tempCols, inplace=True, errors='ignore', axis=1)
            
            playerDataCols = ['Player','Year','Semester']
            playerDataTableMerge = self.player_data_table[playerDataCols+PLAYER_SIMPLE_FEATURE_COLS]
            df = pd.merge(df,
                            playerDataTableMerge,
                            how='left',
                            left_on=[role,'Year','Semester'],
                            right_on=playerDataCols)

            renameCols = PLAYER_SIMPLE_FEATURE_COLS.copy()
            renameCols.append('Player')
            renameCols = [role+'_'+x for x in renameCols]
            renameDict = dict(zip(PLAYER_SIMPLE_FEATURE_COLS+['Player'], renameCols))
            df = df.rename(columns=renameDict)
            df.drop([role+'_Player'],axis=1,inplace=True)

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True,inplace=True)

        self.match_list = df
        return df
    
    def fill_nan_values_player(self, df):
        self.cont = []

        print(f'df size: {len(df)}')
        match_list_fill = df.reset_index().copy()
        match_list_fill.reset_index(drop=True,inplace=True)
        match_list_fill.columns = [x.replace(' ','_') for x in match_list_fill.columns]
        oldNanSum = sum(match_list_fill.isna().sum())

        self.player_data_table.columns = [x.replace(' ','_') for x in self.player_data_table.columns]

        for feature in PLAYER_SIMPLE_FEATURE_COLS:
            for role_side in ROLE_SIDE_COLS:
                print('\n-',role_side+'_'+feature)
                side = role_side.split('_')[1]
                role = role_side.split('_')[0]
                role_side_feature = role_side+'_'+feature
                
                print(f'\nold nan cont: {match_list_fill[role_side_feature].isna().sum()}')
                match_list_fill[role_side_feature] = match_list_fill['index'].apply(lambda indexl: self.fill_feature_player(match_list_fill
                                                                                                                            ,indexl
                                                                                                                            ,role
                                                                                                                            ,feature
                                                                                                                            ,side))
                print(f'\nnew nan cont: {match_list_fill[role_side_feature].isna().sum()}')
                newNanSum = sum(match_list_fill.isna().sum())
                print(f'\nold nan sum: {oldNanSum}, new nan sum: {newNanSum}, diff: {oldNanSum-newNanSum}')
                
        match_list_fill.drop('index',inplace=True,axis=1)

        for col in match_list_fill.columns:
                        match_list_fill[col] = match_list_fill[col].fillna(0)

        self.match_list_fill = match_list_fill
        return match_list_fill
    
#%%================Table Scraping================###

    def make_player_data_table(self):
        player_data_table = pd.DataFrame()
        for year,season in zip(SEASONS_YEAR,SEASONS):
            player_data_table_temp = pd.DataFrame()
            
            for semester,split in zip(SEASONS_SEMESTER,SEASONS_SPLIT):
                if not (year==self.current_year and split==self.current_semester):

                    playersLink = f'https://gol.gg/players/list/season-{season}/split-{split}/tournament-ALL/'
                    page = requests.get(playersLink,headers=headers)
                    bs = BeautifulSoup(page.content, 'lxml')
                    linhas = bs.select("""a[href*='player-stats']""")
                    playersCode = [x['href'].split('/')[2] for x in linhas]

                    player_data_table_temp2 = self.get_table(playersLink)
                    player_data_table_temp2['playerCode'] = playersCode
                    player_data_table_temp2['Semester'] = semester
                    player_data_table_temp2['Split'] = split

                    player_data_table_temp = pd.concat([player_data_table_temp,player_data_table_temp2])
                    player_data_table_temp.reset_index(drop=True,inplace=True)
                    
            player_data_table_temp['Year'] = year
            player_data_table = pd.concat([player_data_table,player_data_table_temp])
            player_data_table.reset_index(drop=True,inplace=True)
        
        self.player_data_table = player_data_table
        return player_data_table
    
    def make_matches_table(self):
        match_list = pd.DataFrame()
        for tournament in self.Utils.all_tournaments:
            page = requests.get(f'https://gol.gg/tournament/tournament-matchlist/{tournament}/',headers=headers)
            bs = BeautifulSoup(page.content, 'lxml')
            linhas = bs.select("""a[href*='game/stats/']""")
            gameCodesPre = [x['href'].split('/')[3] for x in linhas]

            match_list_temp = self.get_table(f'https://gol.gg/tournament/tournament-matchlist/{tournament}/')
            if len(match_list_temp)>0:
                match_list_temp = match_list_temp[match_list_temp['Score'].str.contains('FF') == False]
                match_list_temp['matchCodePre'] = gameCodesPre
                match_list_temp['Tournament'] = tournament
                match_list_temp.dropna(inplace=True)
                match_list = pd.concat([match_list,match_list_temp])
            
        match_list = (match_list.drop(['Game','Unnamed: 4','Patch'],axis=1)
                            .rename(columns={'Unnamed: 1':'Blue','Unnamed: 3':'Red'})
                            .reset_index(drop=True))
        
        ##=======Match Details=======##
        gameCodes=[]
        match_list[ROLE_SIDE_COLS] = 'sNaN'
        for i in range(len(match_list)):
            code = match_list['matchCodePre'][i]
            
            page = requests.get(f'https://gol.gg/game/stats/{code}/page-summary/',headers=headers)
            bs = BeautifulSoup(page.content, 'lxml')
            linhas = bs.select("""a[href*='page-game']""")
            gameCodesSummary = [x['href'].split('/')[3] for x in linhas]
            gameCodes.append(gameCodesSummary)
            
            tables = pd.read_html(page.text)
            blueNames = (tables[0])['Player']
            redNames = (tables[1])['Player']
            allNames = list(blueNames)
            allNames.extend(list(redNames))
            
            for name,val in zip(ROLE_SIDE_COLS,allNames):
                match_list[name][i] = val

        match_list['matchCode'] = gameCodes
        match_list = match_list.explode('matchCode')
        match_list.reset_index(drop=True,inplace=True)
        
        self.match_list = match_list
        return match_list
    
    def make_team_data_table(self):
        team_data_table = pd.DataFrame()
        for year,season in zip(SEASONS_YEAR,SEASONS):

            team_data_table_temp = pd.DataFrame()
            for semester,split in zip(SEASONS_SEMESTER,SEASONS_SPLIT):
                if not(year=='2023' and split=='Summer'):

                    teamsLink = f'https://gol.gg/teams/list/season-{season}/split-{split}/tournament-ALL/'
                    page = requests.get(teamsLink,headers=headers)
                    bs = BeautifulSoup(page.content, 'lxml')
                    linhas = bs.select("""a[href*='team-stats']""")
                    teamsCode = [x['href'].split('/')[2] for x in linhas]

                    teamDataTableTemp2 = self.get_table(teamsLink)
                    teamDataTableTemp2['teamCode'] = teamsCode
                    teamDataTableTemp2['Semester'] = semester
                    teamDataTableTemp2['Split'] = split

                    team_data_table_temp = pd.concat([team_data_table_temp,teamDataTableTemp2])
                    team_data_table_temp.reset_index(drop=True,inplace=True)
            
            team_data_table_temp['Year'] = year
            team_data_table = pd.concat([team_data_table,team_data_table_temp])
            team_data_table.reset_index(drop=True,inplace=True)

        self.team_data_table = team_data_table
        return team_data_table