from IPython.display import HTML
import random
import constants
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold, cross_val_score
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score
from sklearn.cluster import AffinityPropagation as AP

#%%====================================================Output generation====================================================########
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