import os

########%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_System_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%########
ROOT_DIR = os.path.dirname(os.path.abspath('requirements.txt'))

########%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Date_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%########
SEASONS = ['S9','S10','S11','S12','S13']
SEASONS_YEAR = [2019,2020,2021,2022,2023]
SEASONS_SEMESTER = [0,1]
SEMESTER_YEARS = [str(x)+str(y) for x in SEASONS_YEAR for y in SEASONS_SEMESTER]
SEASONS_NUM = [*range(len(SEMESTER_YEARS))]
SEASONS_SPLIT = ['Spring','Summer']

DATE_COLS = ['Semester','Year','Split']
EXTRA_DATE_COLS = ['realSemester', 'realYear','realSemesterYear', 'semesterYear']
ALL_DATE_COLS = DATE_COLS + EXTRA_DATE_COLS

CURRENT_YEAR = 2023
CURRENT_SEMESTER = 1
CURRENT_YEAR_SEMESTER = str(CURRENT_YEAR) + str(CURRENT_SEMESTER)

SEASON_TO_YEAR = dict(zip(SEASONS,SEASONS_YEAR))
YEAR_TO_SEASON = dict(zip(SEASONS_YEAR, SEASONS))

SEMESTER_TO_SPLIT = dict(zip(SEASONS_SEMESTER, SEASONS_SPLIT))
SPLIT_TO_SEMESTER = dict(zip(SEASONS_SPLIT, SEASONS_SEMESTER))

SY_TO_SN = dict(zip(SEMESTER_YEARS, SEASONS_NUM))
SN_TO_SY = dict(zip(SEASONS_NUM, SEMESTER_YEARS))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Targets_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
TARGETS = ['Score','totalKills']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Nominal_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
ROLES = ['TOP','JNG','MID','ADC','SUP']

ROLE_SIDE_COLS = [pos + '_' + side
                  for pos in ROLES
                  for side in ['Blue','Red']]

OFF_COLS = ['Blue', 'Red', 'Tournament', 'matchCode', 'blueKills', 'redKills'
            , 'tournamentId', 'regionAbrev'] + ALL_DATE_COLS


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Formatting_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
TEAM_INFO_COLS = ['Name','Region','teamCode'] + ROLES + DATE_COLS
TEAM_INT_COLS = ['Games','GPM','GDM','DPM']
TEAM_FLOAT_COLS = ['K:D','Kills_/_game','Deaths_/_game', 'Towers_killed', 'Towers_lost', 'FB%',
                 'FT%', 'DRAPG', 'DRA%', 'HERPG', 'HER%', 'DRA@15', 'TD@15', 'GD@15', 'PPG',
                 'NASHPG', 'NASH%', 'CSM']

PLAYER_INFO_COLS = ['Player','Country','playerCode'] + DATE_COLS 
PLAYER_INT_COLS = ['Games','GPM','DPM','GD@15','CSD@15','XPD@15','Penta_Kills','Solo_Kills']
PLAYER_FLOAT_COLS = ['Win_rate','KDA','Avg_kills', 'Avg_deaths','Avg_assists','CSM','KP%','DMG%','VSPM',
                     'Avg_WPM','Avg_WCPM','Avg_VWPM','FB_%','FB_Victim']


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Features_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
PLAYER_SIMPLE_FEATURE_COLS = PLAYER_INT_COLS + PLAYER_FLOAT_COLS

TEAM_SIMPLE_FEATURE_COLS = TEAM_INT_COLS + TEAM_FLOAT_COLS


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Complete features Team_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
FINAL_BLUE_TEAM_FEATURES = ['Team_Blue_' + x for x in TEAM_SIMPLE_FEATURE_COLS]
                    
FINAL_RED_TEAM_FEATURES = ['Team_Red_' + x for x in TEAM_SIMPLE_FEATURE_COLS]

FINAL_ALL_TEAM_FEATURES = FINAL_BLUE_TEAM_FEATURES + FINAL_RED_TEAM_FEATURES


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_Complete features Player_%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
FINAL_BLUE_PLAYER_FEATURES = [pos+'_Blue_'+feature 
                              for pos in ROLES
                              for feature in PLAYER_SIMPLE_FEATURE_COLS]

FINAL_RED_PLAYER_FEATURES = [pos+'_Red_'+feature 
                              for pos in ROLES
                              for feature in PLAYER_SIMPLE_FEATURE_COLS]

FINAL_ALL_PLAYER_FEATURES = FINAL_BLUE_PLAYER_FEATURES + FINAL_RED_PLAYER_FEATURES