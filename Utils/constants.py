########====================================================Date====================================================########
SEASONS = ['S9','S10','S11','S12','S13']
SEASONS_YEAR = [2019,2020,2021,2022,2023]
SEASONS_SEMESTER = [0,1]
SEASONS_SPLIT = ['Spring','Summer']
DATE_COLS = ['Semester','Year','Split']
EXTRA_DATE_COLS = ['realSemester', 'realYear','realSemesterYear']
ALL_DATE_COLS = DATE_COLS + EXTRA_DATE_COLS

########====================================================Targets====================================================########
TARGETS = ['Score','totalKills']

########====================================================Nominal====================================================########

ROLES = ['TOP','JNG','MID','ADC','SUP']

ROLE_SIDE_COLS = [pos + '_' + side
                  for pos in ROLES
                  for side in ['Blue','Red']]

OFF_COLS = ['Blue', 'Red', 'Tournament', 'matchCode', 'blueKills', 'redKills'
            , 'tournamentId', 'regionAbrev'] + ALL_DATE_COLS

########====================================================Formatting====================================================########
TEAM_INFO_COLS = ['Name','Region','teamCode'] + ROLES + DATE_COLS
TEAM_INT_COLS = ['Games','GPM','GDM','DPM']
TEAM_FLOAT_COLS = ['K:D','Kills_/_game','Deaths_/_game', 'Towers_killed', 'Towers_lost', 'FB%',
                 'FT%', 'DRAPG', 'DRA%', 'HERPG', 'HER%', 'DRA@15', 'TD@15', 'GD@15', 'PPG',
                 'NASHPG', 'NASH%', 'CSM']

PLAYER_INFO_COLS = ['Player','Country','playerCode'] + DATE_COLS 
PLAYER_INT_COLS = ['Games','GPM','DPM','GD@15','CSD@15','XPD@15','Penta_Kills','Solo_Kills']
PLAYER_FLOAT_COLS = ['Win_rate','KDA','Avg_kills', 'Avg_deaths','Avg_assists','CSM','KP%','DMG%','VSPM',
                     'Avg_WPM','Avg_WCPM','Avg_VWPM','FB_%','FB_Victim']

########====================================================Features====================================================########
PLAYER_SIMPLE_FEATURE_COLS = PLAYER_INT_COLS + PLAYER_FLOAT_COLS

TEAM_SIMPLE_FEATURE_COLS = TEAM_INT_COLS + TEAM_FLOAT_COLS

########====================================================Complete features Team====================================================########
FINAL_BLUE_TEAM_FEATURES = ['Team_Blue_' + x for x in TEAM_SIMPLE_FEATURE_COLS]
                    
FINAL_RED_TEAM_FEATURES = ['Team_Red_' + x for x in TEAM_SIMPLE_FEATURE_COLS]

FINAL_ALL_TEAM_FEATURES = FINAL_BLUE_TEAM_FEATURES + FINAL_RED_TEAM_FEATURES

########====================================================Complete features Player====================================================########
FINAL_BLUE_PLAYER_FEATURES = [pos+'_Blue_'+feature 
                              for pos in ROLES
                              for feature in PLAYER_SIMPLE_FEATURE_COLS]

FINAL_RED_PLAYER_FEATURES = [pos+'_Red_'+feature 
                              for pos in ROLES
                              for feature in PLAYER_SIMPLE_FEATURE_COLS]

FINAL_ALL_PLAYER_FEATURES = FINAL_BLUE_PLAYER_FEATURES + FINAL_RED_PLAYER_FEATURES