from IPython.display import HTML
import random
from constants import *
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold, cross_val_score
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score
from sklearn.cluster import AffinityPropagation as AP

########====================================================Interface====================================================########
def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)
    
########====================================================Scrapping====================================================########
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

########====================================================Data manipulation====================================================########

def train_test_split2(dfToSplitFunc,df2, tournamentId, currentTarget, dropTypeF, verbose=True):
    testData = df2[df2['tournament_id']==tournamentId].copy()
    testData = filterNan(testData,1)
    xtest= testData.drop(['Date',currentTarget],axis=1).copy()
    xtest= xtest.drop(offCols,axis=1,errors='ignore')
    ytest = testData[currentTarget]
    
    trainData = dfToSplitFunc[dfToSplitFunc['tournament_id']!=tournamentId].copy()
    trainData = filterNan(trainData,dropTypeF)
    xtrain = trainData.drop(['Date',currentTarget],axis=1).copy()
    xtrain = xtrain.drop(offCols,axis=1,errors='ignore')
    ytrain = trainData[currentTarget]
    
    if list(ytrain).count(0)/len(ytrain)==1:
        print(len(ytrain))
        print(dropTypeF)
        print(dfToSplitFunc[currentTarget])
        print(df2[currentTarget])
        print('==========================================================')
    print('============')
    
    ytrain_mean, ytrain_std = np.mean(ytrain), np.std(ytrain)
    cut_off = ytrain_std * 1.1
    lower, upper = ytrain_mean - cut_off, ytrain_mean + cut_off
    
    outlierMask = ytrain.apply(lambda x: False if x < lower or x > upper else True)
    
    if verbose:
        print(f'train len: {len(xtrain)}')
    lentemp = len(xtrain)
    xtrain, ytrain = xtrain[outlierMask], ytrain[outlierMask]
    if verbose:
        print(f'train len no outliers: {len(xtrain)}')
        print(f'percent of len removed: {round(abs(len(xtrain)/lentemp*100-100),2)}%')
        print(f'test len: {len(xtest)}\n')
    
    return xtrain, ytrain, xtest, ytest

########====================================================Output generation====================================================########
def printFinalResults(df, accName):
    print('===============================\n')
    meanAcc = df[accName].mean()
    print(f'mean accuracy: {round(meanAcc,3)}')
    dfAvgSize = df['size'].mean()
    print(f'avg df len: {dfAvgSize}\n')
    
def plotOverview(col,region,usePred):
    reg=region

    regName=regions['region'][reg].replace('%20',' ')
    tresholdC = regions['ceiling'][reg]
    tresholdF = regions['floor'][reg]
    lsdf = labelStatsTemp.replace(reg,-1).sort_values(by='Region',ascending=True).copy()
    lsdf = lsdf[lsdf['Region'].isin(regionBestCluster[str(reg)])]
    
    featuresUse =  list(regionsFeatureImp[str(region)].dropna().index.values)
    
    testdf = lsdf[lsdf['Region']==-1].copy()
    testdf = testdf[lsdfCols]
    xtest = testdf.drop('label',axis=1).copy()
    ytest = testdf['label'].copy()

    traindf = lsdf[lsdf['Region']!=-1].copy()
    traindf = traindf[lsdfCols]
    xtrain =  traindf.drop('label',axis=1).copy()
    ytrain =  traindf['label'].copy()

    if len(traindf)<1:
        lsdfShort = lsdf.copy()
        lsdfShort = lsdfShort[lsdfCols]
        features = lsdfShort.drop(['label'],axis=1)
        label = lsdfShort['label']

        xtrain, xtest, ytrain, ytest = train_test_split(features, label, train_size=0.75, random_state=42)
        
    base_model = XGBRegressor()
    base_model.fit(xtrain, ytrain)
    
    errors,predPlot = evaluate1(base_model,xtest,ytest)
    
    win,perc = evaluate2(predPlot,ytest.reset_index(drop=True),regions[col][reg],col,regions['botBase'][reg],regions['topBase'][reg],usePred)
    
    meanPred = np.mean(predPlot)
    
    return predPlot,ytest,win,perc,meanPred

def evaluate1(model, test_features, test_labels):
    
    predictions = model.predict(test_features)
    errors = predictions - test_labels
    resultTest = pd.DataFrame()
    
    return errors,predictions

def evaluate2(predictions,test_labels,testing,col,botBase,topBase,pred):
    
    tp=[]
    
    threshold=testing
    meanPred = (np.mean(predictions))
    if pred!=-1:
        meanPred=pred
        threshold=0
        
#     print(meanPred+threshold)
#     print(predictions)    
    for x in range(len(predictions)):
            
        if (col=='ceiling') & ((predictions[x])>meanPred+threshold):
            if (botBase<=test_labels[x]):
                tp.append(1)
            else:
                tp.append(0)
                
        if (col=='floor') & ((predictions[x])<meanPred+threshold):
            if(topBase>=test_labels[x]): 
                tp.append(1)
            else: tp.append(0)
            
    tp2 = tp.count(1)/len(tp)*100 if len(tp)>0 else 0
    perc = len(tp)/len(predictions)
    
    return tp2,perc