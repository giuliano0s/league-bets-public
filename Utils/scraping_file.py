########====================================================IMPORTS====================================================########
#import urllib.request
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

    def __init__(self) -> None:
        pass
    
    def teamTournamentFind(code,split):
        page = requests.get(f'https://gol.gg/teams/team-matchlist/{code}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def playerTournamentFind(code,season,split):
        page = requests.get(f'https://gol.gg/players/player-matchlist/{code}/season-{season}/split-{split}/tournament-ALL/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        linhas = bs.select("""a[href*='tournament/tournament-stats/']""")
        tournamentNames = [x['href'].split('/')[-2] for x in linhas]
        ret = max(set(tournamentNames), key = tournamentNames.count)

        return ret

    def getTable(url,mult=False):
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
        
    def getMatchStats(code):
        page = requests.get(f'https://gol.gg/game/stats/{code}/page-game/',headers=headers)
        bs = BeautifulSoup(page.content, 'lxml')
        redScore = int(bs.find_all('span',class_='score-box red_line')[0].text)
        blueScore = int(bs.find_all('span',class_='score-box blue_line')[0].text)
        allScore = redScore+blueScore
        
        return [blueScore, redScore, allScore]

    def scoreSelect(score):
        blueScore = int(score[0])
        redScore = int(score[-1])
        
        finalScore = blueScore-redScore
        
        if finalScore < 0:
            return 1
        elif finalScore > 0:
            return 0
        else:
            return 2

    def findRegionTournament(tournament):
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