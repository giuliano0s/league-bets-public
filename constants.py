########====================================================Seasons====================================================########
seasons = ['S9','S10','S11','S12','S13']
seasonsYear = [2019,2020,2021,2022,2023]
seasonsSemester = [0,1]
seasonsSplit = ['Spring','Summer']

########====================================================Targets====================================================########
targets = ['Score','totalKills','handicap']

########====================================================Nominal====================================================########
nameCols = (['TOP_Blue','JNG_Blue','MID_Blue','ADC_Blue','SUP_Blue',
           'TOP_Red','JNG_Red','MID_Red','ADC_Red','SUP_Red'])

positions = ['TOP','JNG','MID','ADC','SUP']

offCols = ['Blue', 'Red', 'Score', 'Tournament', 'TOP_Blue', 'JNG_Blue', 'tournament_id',
           'MID_Blue', 'ADC_Blue', 'SUP_Blue', 'TOP_Red', 'JNG_Red', 'MID_Red',
           'ADC_Red', 'SUP_Red', 'matchCode', 'blueKills', 'redKills', 'TournamentRegion',
           'Semester', 'Split', 'Year', 'realSemester', 'realYear','realSemesterYear']

########====================================================Formatting====================================================########
teamStrCols = ['Name','Season','Region','TOP','JNG','MID','ADC','SUP','teamCode']#,'Tournament'
teamIntCols = ['Games','GPM','GDM','DPM']
teamFloatCols = ['K:D','Kills_/_game','Deaths_/_game', 'Towers_killed', 'Towers_lost', 'FB%',
                 'FT%', 'DRAPG', 'DRA%', 'HERPG', 'HER%', 'DRA@15', 'TD@15', 'GD@15', 'PPG',
                 'NASHPG', 'NASH%', 'CSM']

playerStrCols = ['Player','Country','playerCode','Split']
playerIntCols = ['Games','GPM','DPM','GD@15','CSD@15','XPD@15','Penta Kills','Solo Kills','Semester','Year']
playerFloatCols = ['Win_rate','KDA','Avg_kills', 'Avg_deaths','Avg_assists','CSM','KP%','DMG%','VSPM',
                 'Avg_WPM','Avg_WCPM','Avg_VWPM','FB_%','FB_Victim']

########====================================================Math operations====================================================########

sumFeatures = ['Games','Penta_Kills','Solo_Kills']
meanFeatures = ['Win_rate','KDA','Avg_kills', 'Avg_deaths','Avg_assists','CSM','KP%',
                'DMG%','VSPM','Avg_WPM','Avg_WCPM','Avg_VWPM','FB_%','FB_Victim','GPM',
                'DPM','GD@15','CSD@15','XPD@15']

########====================================================Features Player====================================================########
playerFeatureCols = playerIntCols + playerFloatCols

########====================================================Features Team====================================================########
teamFeatureCols = teamIntCols + teamFloatCols

########====================================================Complete features Team====================================================########
finalBlueTeamFeatures = ['Team_Blue_Win_rate', 'Team_Blue_KDA', 'Team_Blue_Avg_kills', 'Team_Blue_Avg_deaths',
                    'Team_Blue_Avg_assists', 'Team_Blue_CSM', 'Team_Blue_KP%', 'Team_Blue_DMG%', 'Team_Blue_VSPM',
                    'Team_Blue_Avg_WPM', 'Team_Blue_Avg_WCPM', 'Team_Blue_Avg_VWPM', 'Team_Blue_FB_%',
                    'Team_Blue_FB_Victim', 'Team_Blue_GPM', 'Team_Blue_DPM', 'Team_Blue_GD@15', 'Team_Blue_CSD@15',
                    'Team_Blue_XPD@15', 'Team_Blue_Games', 'Team_Blue_Penta_Kills', 'Team_Blue_Solo_Kills']
                    
finalRedTeamFeatures =  ['Team_Red_Win_rate', 'Team_Red_KDA', 'Team_Red_Avg_kills', 'Team_Red_Avg_deaths',
                    'Team_Red_Avg_assists', 'Team_Red_CSM', 'Team_Red_KP%', 'Team_Red_DMG%', 'Team_Red_VSPM',
                    'Team_Red_Avg_WPM', 'Team_Red_Avg_WCPM', 'Team_Red_Avg_VWPM', 'Team_Red_FB_%', 'Team_Red_FB_Victim',
                    'Team_Red_GPM', 'Team_Red_DPM', 'Team_Red_GD@15', 'Team_Red_CSD@15', 'Team_Red_XPD@15',
                    'Team_Red_Games', 'Team_Red_Penta_Kills', 'Team_Red_Solo_Kills']

finalAllTeamFeatures = finalBlueTeamFeatures + finalRedTeamFeatures

########====================================================Complete features Player====================================================########
finalBluePlayerFeatures = ([
'TOP_Blue_Win_rate','TOP_Blue_KDA','TOP_Blue_Avg_kills','TOP_Blue_Avg_deaths','TOP_Blue_Avg_assists','TOP_Blue_CSM','TOP_Blue_KP%','TOP_Blue_DMG%','TOP_Blue_VSPM','TOP_Blue_Avg_WPM','TOP_Blue_Avg_WCPM','TOP_Blue_Avg_VWPM','TOP_Blue_FB_%','TOP_Blue_FB_Victim','TOP_Blue_Games','TOP_Blue_GPM','TOP_Blue_DPM','TOP_Blue_GD@15','TOP_Blue_CSD@15','TOP_Blue_XPD@15','TOP_Blue_Penta_Kills','TOP_Blue_Solo_Kills','JNG_Blue_Win_rate','JNG_Blue_KDA','JNG_Blue_Avg_kills','JNG_Blue_Avg_deaths','JNG_Blue_Avg_assists','JNG_Blue_CSM','JNG_Blue_KP%','JNG_Blue_DMG%','JNG_Blue_VSPM','JNG_Blue_Avg_WPM','JNG_Blue_Avg_WCPM','JNG_Blue_Avg_VWPM','JNG_Blue_FB_%','JNG_Blue_FB_Victim','JNG_Blue_Games','JNG_Blue_GPM','JNG_Blue_DPM','JNG_Blue_GD@15','JNG_Blue_CSD@15','JNG_Blue_XPD@15','JNG_Blue_Penta_Kills','JNG_Blue_Solo_Kills','MID_Blue_Win_rate','MID_Blue_KDA','MID_Blue_Avg_kills','MID_Blue_Avg_deaths','MID_Blue_Avg_assists','MID_Blue_CSM','MID_Blue_KP%','MID_Blue_DMG%','MID_Blue_VSPM','MID_Blue_Avg_WPM','MID_Blue_Avg_WCPM','MID_Blue_Avg_VWPM','MID_Blue_FB_%','MID_Blue_FB_Victim','MID_Blue_Games','MID_Blue_GPM','MID_Blue_DPM','MID_Blue_GD@15','MID_Blue_CSD@15','MID_Blue_XPD@15','MID_Blue_Penta_Kills','MID_Blue_Solo_Kills','ADC_Blue_Win_rate','ADC_Blue_KDA','ADC_Blue_Avg_kills','ADC_Blue_Avg_deaths','ADC_Blue_Avg_assists','ADC_Blue_CSM','ADC_Blue_KP%','ADC_Blue_DMG%','ADC_Blue_VSPM','ADC_Blue_Avg_WPM','ADC_Blue_Avg_WCPM','ADC_Blue_Avg_VWPM','ADC_Blue_FB_%','ADC_Blue_FB_Victim','ADC_Blue_Games','ADC_Blue_GPM','ADC_Blue_DPM','ADC_Blue_GD@15','ADC_Blue_CSD@15','ADC_Blue_XPD@15','ADC_Blue_Penta_Kills','ADC_Blue_Solo_Kills','SUP_Blue_Win_rate','SUP_Blue_KDA','SUP_Blue_Avg_kills','SUP_Blue_Avg_deaths','SUP_Blue_Avg_assists','SUP_Blue_CSM','SUP_Blue_KP%','SUP_Blue_DMG%','SUP_Blue_VSPM','SUP_Blue_Avg_WPM','SUP_Blue_Avg_WCPM','SUP_Blue_Avg_VWPM','SUP_Blue_FB_%','SUP_Blue_FB_Victim','SUP_Blue_Games','SUP_Blue_GPM','SUP_Blue_DPM','SUP_Blue_GD@15','SUP_Blue_CSD@15','SUP_Blue_XPD@15','SUP_Blue_Penta_Kills','SUP_Blue_Solo_Kills'])

finalRedPlayerFeatures = ([ 'TOP_Red_Win_rate','TOP_Red_KDA','TOP_Red_Avg_kills','TOP_Red_Avg_deaths','TOP_Red_Avg_assists','TOP_Red_CSM','TOP_Red_KP%','TOP_Red_DMG%','TOP_Red_VSPM','TOP_Red_Avg_WPM','TOP_Red_Avg_WCPM','TOP_Red_Avg_VWPM','TOP_Red_FB_%','TOP_Red_FB_Victim','TOP_Red_Games','TOP_Red_GPM','TOP_Red_DPM','TOP_Red_GD@15','TOP_Red_CSD@15','TOP_Red_XPD@15','TOP_Red_Penta_Kills','TOP_Red_Solo_Kills','JNG_Red_Win_rate','JNG_Red_KDA','JNG_Red_Avg_kills','JNG_Red_Avg_deaths','JNG_Red_Avg_assists','JNG_Red_CSM','JNG_Red_KP%','JNG_Red_DMG%','JNG_Red_VSPM','JNG_Red_Avg_WPM','JNG_Red_Avg_WCPM','JNG_Red_Avg_VWPM','JNG_Red_FB_%','JNG_Red_FB_Victim','JNG_Red_Games','JNG_Red_GPM','JNG_Red_DPM','JNG_Red_GD@15','JNG_Red_CSD@15','JNG_Red_XPD@15','JNG_Red_Penta_Kills','JNG_Red_Solo_Kills','MID_Red_Win_rate','MID_Red_KDA','MID_Red_Avg_kills','MID_Red_Avg_deaths','MID_Red_Avg_assists','MID_Red_CSM','MID_Red_KP%','MID_Red_DMG%','MID_Red_VSPM','MID_Red_Avg_WPM','MID_Red_Avg_WCPM','MID_Red_Avg_VWPM','MID_Red_FB_%','MID_Red_FB_Victim','MID_Red_Games','MID_Red_GPM','MID_Red_DPM','MID_Red_GD@15','MID_Red_CSD@15','MID_Red_XPD@15','MID_Red_Penta_Kills','MID_Red_Solo_Kills','ADC_Red_Win_rate','ADC_Red_KDA','ADC_Red_Avg_kills','ADC_Red_Avg_deaths','ADC_Red_Avg_assists','ADC_Red_CSM','ADC_Red_KP%','ADC_Red_DMG%','ADC_Red_VSPM','ADC_Red_Avg_WPM','ADC_Red_Avg_WCPM','ADC_Red_Avg_VWPM','ADC_Red_FB_%','ADC_Red_FB_Victim','ADC_Red_Games','ADC_Red_GPM','ADC_Red_DPM','ADC_Red_GD@15','ADC_Red_CSD@15','ADC_Red_XPD@15','ADC_Red_Penta_Kills','ADC_Red_Solo_Kills','SUP_Red_Win_rate','SUP_Red_KDA','SUP_Red_Avg_kills','SUP_Red_Avg_deaths','SUP_Red_Avg_assists','SUP_Red_CSM','SUP_Red_KP%','SUP_Red_DMG%','SUP_Red_VSPM','SUP_Red_Avg_WPM','SUP_Red_Avg_WCPM','SUP_Red_Avg_VWPM','SUP_Red_FB_%','SUP_Red_FB_Victim','SUP_Red_Games','SUP_Red_GPM','SUP_Red_DPM','SUP_Red_GD@15','SUP_Red_CSD@15','SUP_Red_XPD@15','SUP_Red_Penta_Kills','SUP_Red_Solo_Kills'])

finalAllPlayerFeatures = finalBluePlayerFeatures + finalRedPlayerFeatures