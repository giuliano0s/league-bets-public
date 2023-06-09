########====================================================IMPORTS====================================================########
from Utils.constants import *
import Utils.utils_file as utils_file

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

    def __init__(self, year=CURRENT_YEAR, semester=CURRENT_SEMESTER, cache=True) -> None:
        self.current_year = year
        self.current_semester = 'Summer' if semester==1 else 'Spring'
        self.Utils = utils_file.Utils_Class(cache_scraping=cache)
        self.all_tournaments = self.Utils.all_tournaments
        
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
            newValue = np.nanmean(playerDfFilter[feature].tail(3))
            if not np.isnan(newValue):
                self.cont.append(0)
            print(f'nan vals removed: {len(self.cont)}',end='\r')
                
            return newValue
        else:
            return value
        
    def transforming_player(self, df):
        for col in df:
            df[col] = df[col].apply(lambda x: str(x).replace('%',''))
            df[col] = df[col].apply(lambda x: np.nan if ('-' in x and len(x)<2) else x )
        df.columns = [x.replace(' ','_') for x in df.columns]

        df[PLAYER_FLOAT_COLS] = df[PLAYER_FLOAT_COLS].astype(float)
        df[PLAYER_INT_COLS+['Year','Semester']] = df[PLAYER_INT_COLS+['Year','Semester']].astype(float)

        self.player_data_table = df
        return df
    
    def transforming_team(self, df, updating=False):
        for col in df:
            df[col] = df[col].apply(lambda x: str(x).replace('%',''))
            df[col] = df[col].apply(lambda x: np.nan if ('-' in x and len(x)<2) else x )
        df.columns = [x.replace(' ','_') for x in df.columns]

        df[TEAM_FLOAT_COLS] = df[TEAM_FLOAT_COLS].astype(float)
        df[TEAM_INT_COLS] = df[TEAM_INT_COLS].astype(float)

        df['Game_duration'] = pd.to_datetime(df['Game_duration'], format='%M:%S').dt.time

        df[ROLES] = 'sNaN'
        for i in range(len(df)):
            
            code = df['teamCode'][i]
            split = df['Split'][i]
            page = requests.get(f'https://gol.gg/teams/team-stats/{code}/split-{split}/tournament-ALL/',headers=headers)
            tables = pd.read_html(page.text)[-1]
            
            allNames = tables['Player'][1:6]
            for name,val in zip(ROLES,allNames):
                df[name][i] = val

        if not updating:
            self.team_data_table = df
        return df
    
    def transforming_match(self, df, updating=False):
        df['StatsTemp'] = df['matchCode'].apply(lambda x: self.get_match_stats(x))
        df['blueKills'] = df['StatsTemp'].apply(lambda x: x[0])
        df['redKills'] = df['StatsTemp'].apply(lambda x: x[1])
        df['totalKills'] = df['StatsTemp'].apply(lambda x: x[2])

        matchListToDrop=['StatsTemp','matchCodePre']
        for col in matchListToDrop:
            df.drop(col, axis=1, errors='ignore', inplace=True)

        df['Score'] = df['Score'].apply(lambda x: self.score_select(x))

        if not updating: 
            self.match_list = df
        return df
    
    def season_data_swap(self, df, updating=False):
        
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
            
            player_data_cols = ['Player','Year','Semester']
            player_data_table_merge = self.player_data_table[player_data_cols+PLAYER_SIMPLE_FEATURE_COLS]
            df = pd.merge(df,
                            player_data_table_merge,
                            how='left',
                            left_on=[role,'Year','Semester'],
                            right_on=player_data_cols)

            renameCols = PLAYER_SIMPLE_FEATURE_COLS.copy()
            renameCols.append('Player')
            renameCols = [role+'_'+x for x in renameCols]
            renameDict = dict(zip(PLAYER_SIMPLE_FEATURE_COLS+['Player'], renameCols))
            df = df.rename(columns=renameDict)
            df.drop([role+'_Player'],axis=1,inplace=True)

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True,inplace=True)

        if not updating:
            self.match_list = df
        return df
    
    def fill_nan_values_player(self, df, updating=False):
        print('Filling NaN values:')
        self.cont = []

        match_list_fill = df.reset_index().copy()
        match_list_fill.reset_index(drop=True,inplace=True)
        match_list_fill.columns = [x.replace(' ','_') for x in match_list_fill.columns]

        self.player_data_table.columns = [x.replace(' ','_') for x in self.player_data_table.columns]

        old_nan_sum = match_list_fill.isna().sum().sum()
        for feature in PLAYER_SIMPLE_FEATURE_COLS:
            for role_side in ROLE_SIDE_COLS:

                side = role_side.split('_')[1]
                role = role_side.split('_')[0]
                role_side_feature = role_side+'_'+feature
                
                
                match_list_fill[role_side_feature] = match_list_fill['index'].apply(lambda indexl: self.fill_feature_player(match_list_fill
                                                                                                                            ,indexl
                                                                                                                            ,role
                                                                                                                            ,feature
                                                                                                                            ,side))

        new_nan_sum = match_list_fill.isna().sum().sum()
        print(f'\nold nan sum: {old_nan_sum}, new nan sum: {new_nan_sum}, diff: {old_nan_sum-new_nan_sum}')
                
        match_list_fill.drop('index',inplace=True,axis=1)

        for col in match_list_fill.columns:
                        match_list_fill[col] = match_list_fill[col].fillna(0)

        if not updating:
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
    
    def make_matches_table(self, updating=False):

        if updating:
            tournaments_to_scan = [x for x in self.Utils.all_tournaments if str(CURRENT_YEAR) in x]
        else:
            tournaments_to_scan = self.Utils.all_tournaments

        match_list = pd.DataFrame()
        for tournament in tournaments_to_scan:
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
        
        if not updating:
            self.match_list = match_list
        return match_list
    
    def make_team_data_table(self, updating=False):
        
        if updating:
            seasons_year_to_scan = [CURRENT_YEAR]
            seasons_to_scan = [YEAR_TO_SEASON[CURRENT_YEAR]]
            seasons_semester_to_scan = [CURRENT_SEMESTER]
            seasons_split_to_scan = [SEMESTER_TO_SPLIT[CURRENT_SEMESTER]]
        else:
            seasons_year_to_scan = SEASONS_YEAR
            seasons_to_scan = SEASONS
            seasons_semester_to_scan = SEASONS_SEMESTER
            seasons_split_to_scan = SEASONS_SPLIT

        team_data_table = pd.DataFrame()
        for year,season in zip(seasons_year_to_scan, seasons_to_scan):

            team_data_table_temp = pd.DataFrame()
            for semester,split in zip(seasons_semester_to_scan, seasons_split_to_scan):
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

        if not updating:
            self.team_data_table = team_data_table

        return team_data_table

#%%================Update================###

    def update(self, player_data=True, team_data=True, match_data=True):
        print(f'Updating semester {CURRENT_SEMESTER} of year {CURRENT_YEAR}\n')
        print('\n=============================\n')
        if player_data:
            self.make_player_data_table()
            self.transforming_player(self.player_data_table)

            self.player_data_table.to_pickle("Data/raw_data/player_data_table.pkl")
            print('player_data_table updated!\n')

        if team_data:
            team_data_table_update = self.make_team_data_table(updating=True)
            team_data_table_update = self.transforming_team(team_data_table_update, updating=True)
            team_data_table_update = pd.concat([self.team_data_table, team_data_table_update])
            subset_temp = [x for x in TEAM_INFO_COLS if x not in ROLES]
            team_data_table_update.drop_duplicates(subset=subset_temp,inplace=True, keep='last')
            self.team_data_table = team_data_table_update

            self.team_data_table.to_pickle("Data/raw_data/team_data_table.pkl")
            print('team_data_table updated!\n')

        if match_data:
            match_list_update = self.make_matches_table(updating=True)
            match_list_update = self.transforming_match(match_list_update, updating=True)
            match_list_update = pd.concat([self.match_list, match_list_update])
            match_list_update.drop_duplicates(subset=['matchCode'],inplace=True, keep='last')
            match_list_update = self.season_data_swap(match_list_update, updating=True)
            self.match_list = match_list_update

            match_list_update = self.fill_nan_values_player(match_list_update, updating=True)
            self.match_list_fill = match_list_update

            self.match_list.to_pickle("Data/raw_data/match_list.pkl")
            self.match_list_fill.to_pickle("Data/raw_data/match_list_fill.pkl")
            print('match_data updated!')


