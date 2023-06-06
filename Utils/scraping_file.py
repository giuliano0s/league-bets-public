########====================================================IMPORTS====================================================########
from Utils.constants import *
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

headers = requests.utils.default_headers()
headers.update({"User-Agent": "Chrome/51.0.2704.103"})

########====================================================CLASS====================================================########

class Scraping_Class:

    def __init__(self, year, semester) -> None:
        self.current_year = year
        self.current_semester = 'Summer' if semester==1 else 'Spring'
    
#%%================Scraping Utils================###
    def teamTournamentFind(self, code,split):
        page = requests.get(f'https://gol.gg/teams/team-matchlist/{code}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def playerTournamentFind(self, code,season,split):
        page = requests.get(f'https://gol.gg/players/player-matchlist/{code}/season-{season}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def getTable(self, url,mult=False):
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
        
    def getMatchStats(self, code):
        page = requests.get(f'https://gol.gg/game/stats/{code}/page-game/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        redScore = int(bs.find_all('span',class_='score-box red_line')[0].text)
        blueScore = int(bs.find_all('span',class_='score-box blue_line')[0].text)
        allScore = redScore+blueScore
        
        return [blueScore, redScore, allScore]

    def scoreSelect(self, score):
        blueScore = int(score[0])
        redScore = int(score[-1])
        
        finalScore = blueScore-redScore
        
        if finalScore < 0:
            return 1
        elif finalScore > 0:
            return 0
        else:
            return 2

    def findRegionTournament(self, tournament):
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

                    player_data_table_temp2 = self.getTable(playersLink)
                    player_data_table_temp2['playerCode'] = playersCode
                    player_data_table_temp2['Semester'] = semester
                    player_data_table_temp2['Split'] = split

                    player_data_table_temp = pd.concat([player_data_table_temp,player_data_table_temp2])
                    player_data_table_temp.reset_index(drop=True,inplace=True)
                    
            player_data_table_temp['Year'] = year
            player_data_table = pd.concat([player_data_table,player_data_table_temp])
            player_data_table.reset_index(drop=True,inplace=True)

        return player_data_table
    
    def make_matches_table(self):
        matchList = pd.DataFrame()
        for tournament in all_tournaments:
            page = requests.get(f'https://gol.gg/tournament/tournament-matchlist/{tournament}/',headers=headers)
            bs = BeautifulSoup(page.content, 'lxml')
            linhas = bs.select("""a[href*='game/stats/']""")
            gameCodesPre = [x['href'].split('/')[3] for x in linhas]

            matchListTemp = Scraping.getTable(f'https://gol.gg/tournament/tournament-matchlist/{tournament}/')
            if len(matchListTemp)>0:
                matchListTemp = matchListTemp[matchListTemp['Score'].str.contains('FF') == False]
                matchListTemp['matchCodePre'] = gameCodesPre
                matchListTemp['Tournament'] = tournament
                matchListTemp.dropna(inplace=True)
                matchList = pd.concat([matchList,matchListTemp])
            
        matchList = (matchList.drop(['Game','Unnamed: 4','Patch'],axis=1)
                            .rename(columns={'Unnamed: 1':'Blue','Unnamed: 3':'Red'})
                            .reset_index(drop=True))