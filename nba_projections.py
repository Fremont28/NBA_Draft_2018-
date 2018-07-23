#7/12/18 
#read-in csv
nba_current=pd.read_csv("nba_metrics.csv")
nba_current=nba_current.dropna() 
#merge dataframes
nba_current1=nba_current[["Player","VORP"]]
nba_finals=pd.merge(nba_current1,sprite1,on="Player")
nba_finals.rename(columns={'VORP_x':'VORP'},inplace=True)
nba_finals['POS.x'].unique() 
nba_finals.info() 

#VORP quantiles
nba_finals['VORP'].quantile(0.9)
nba_finals['VORP'].quantile(0.7)
nba_finals['VORP'].quantile(0.4)
nba_finals['VORP'].quantile(0.3)

#convert non-null to float 
nba_finals['VORP']=nba_finals['VORP'].astype(float)

def vorp_score(vorp):
    if vorp>=10.56:
        return 1
    if (vorp>=3.63 and vorp<10.56):
        return 2
    if (vorp>=0.6 and vorp<3.63):
        return 3
    if (vorp>=0 and vorp<0.6):
        return 4
    else:
        return 5

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
nba_finals['vorp_rank']=nba_finals['VORP'].apply(vorp_score)
nba_finals['vorp_rank'].head(4)
nba_finals.info() 

#random forest classifier 
X=nba_finals.iloc[:,4:33].values 
y=nba_finals['vorp_rank'].values 
from sklearn.metrics import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=344)

random_forest=RandomForestClassifier(n_estimators=50) 
random_forest.fit(X_train,y_train)
predict_vorp=random_forest.predict(X_test)
accuracy=accuracy_score(predict_vorp,y_test)
accuracy #35.1% 

#sgd classifier 
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(n_iter=3,loss="hinge")
sgd.fit(X_train,y_train)
predict_vorp1=sgd.predict(X_test)
accuracy1=sgd.score(X_train,y_train) 

#logistic regression classifier 
from sklearn.linear_model import LogisticRegression

lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)
predict_vorp2=lr_model.predict(X_test)
accuracy_lr=accuracy_score(predict_vorp2,y_test)
accuracy_lr #43.2% 

#k-fold (i.e. 4 folds trained on 3 folds and tested on the last fold)
from sklearn.model_selection import cross_val_score 

random_forest=RandomForestClassifier(n_estimators=90) 
rf_scores=cross_val_score(random_forest,X_train,y_train,cv=8,scoring='accuracy')
rf_scores.mean() #33.2% 
rf_scores.std() #13.0% 

#feature importance
random_forest.feature_importances_ 

#2018 draft combine metrics
draft_metrics=pd.read_csv("2018_metrics.csv")
draft_metrics1=pd.read_csv("2018_metrics1.csv")
combine_metrics=pd.merge(draft_metrics,draft_metrics1,on="Player")
metrics=pd.read_csv("2018_metrics.csv")
metrics1=pd.read_csv("2018_draft.csv")
metrics1=metrics1.dropna() #49,24 

#per game function 
def per_game(stat,gp):
    return stat/gp 

metrics1['MP']=metrics1.apply(lambda x:per_game(x['MP'],x['G']),axis=1)
metrics1['FG']=metrics1.apply(lambda x:per_game(x['FG'],x['G']),axis=1)
metrics1['FGA']=metrics1.apply(lambda x:per_game(x['FGA'],x['G']),axis=1)
metrics1['X3P']=metrics1.apply(lambda x:per_game(x['3P'],x['G']),axis=1)
metrics1['X3PA']=metrics1.apply(lambda x:per_game(x['3PA'],x['G']),axis=1)
metrics1['FT']=metrics1.apply(lambda x:per_game(x['FT'],x['G']),axis=1)
metrics1['FTA']=metrics1.apply(lambda x:per_game(x['FTA'],x['G']),axis=1)
metrics1['TRB']=metrics1.apply(lambda x:per_game(x['TRB'],x['G']),axis=1)
metrics1['AST']=metrics1.apply(lambda x:per_game(x['AST'],x['G']),axis=1)
metrics1['STL']=metrics1.apply(lambda x:per_game(x['STL'],x['G']),axis=1)
metrics1['BLK']=metrics1.apply(lambda x:per_game(x['BLK'],x['G']),axis=1)
metrics1['TOV']=metrics1.apply(lambda x:per_game(x['TOV'],x['G']),axis=1)
metrics1['PF']=metrics1.apply(lambda x:per_game(x['PF'],x['G']),axis=1)
metrics1['FG']=metrics1.apply(lambda x:per_game(x['FG'],x['G']),axis=1)
metrics1['PTS']=metrics1.apply(lambda x:per_game(x['PTS'],x['G']),axis=1)

#join dataframes
final_metrics=pd.merge(combine_metrics,metrics1,on="Player") #40,40 
final_metrics.info()
final_metrics['Height.wo.shoes   ']

#rename columns 
final_metrics=final_metrics.rename(columns={'Lane.Agiility   _x':'Lane.Agiility'})
final_metrics=final_metrics.rename(columns={'three.quarter.sprint  _x':'three.quarter.sprint'})
final_metrics=final_metrics.rename(columns={'standing.vertical   _y':'standing.vertical'})
final_metrics=final_metrics.rename(columns={'Hand.Length ':'Hand.Length'})
final_metrics=final_metrics.rename(columns={'Height.wo.shoes   ':'Height.wo.shoes'})
final_metrics=final_metrics.rename(columns={'Wingspan ':'Wingspan'})
final_metrics=final_metrics.rename(columns={'Hand.Width ':'Hand.Width'})


new_players=final_metrics[['Player','Body.Fat..','Hand.Length','Hand.Width','Height.wo.shoes',
'Standing.Reach.Height.w.shoes','Wingspan','Lane.Agiility','three.quarter.sprint',
'standing.vertical','G','MP','FG','FGA','X3P','X3PA','FT','FTA','TRB','AST','STL',
'BLK','TOV','PF','PTS']]

old_players=sprite1[['Player','Body.Fat..','Hand.Length','Hand.Width','Height.wo.shoes',
'Standing.Reach.Height.w.shoes','Wingspan','Lane.Agiility','three.quarter.sprint',
'standing.vertical','G','MP','FG','FGA','X3P','X3PA','FT','FTA','TRB','AST','STL',
'BLK','TOV','PF','PTS','VORP']] #245,26 

#1. VORP classifier 
#random forest model 
X=old_players.iloc[:,1:25].values 
y=old_players['VORP'].values 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=344)

random_forest=RandomForestClassifier(n_estimators=500) #trees in the forest 
random_forest.fit(X,y) 
predict_vorp=random_forest.predict(X_test)
accuracy=accuracy_score(predict_vorp,y_test)
accuracy #46.9%

#predict on the 2018 nba draft class 
X_new=new_players.iloc[:,1:25].values 
current_predict=random_forest.predict(X_new)

player_results=pd.DataFrame(current_predict,columns=['VORP'])
player_sub=new_players['Player']
predictions=pd.concat([player_sub,player_results],axis=1)
predictions

##2. random forest regressor 
from sklearn.preprocessing import StandardScaler

#scale the data 
scaler=StandardScaler().fit(X_train)
X_train_scale=scaler.transform(X_train)
X_test_scale=scaler.transform(X_test)

#projecting points over the first three seasons
X=old_players.iloc[:,1:24]
y=old_players['PTS']
X1=new_players.iloc[:,1:24]

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=344)
rf_reg.fit(X,y)
rf_predict=rf_reg.predict(X1)
feature_importance=rf_reg.feature_importances_ 
rf_importance = pd.DataFrame(feature_importance, index=X.columns, 
                          columns=["Importance"])
rf_importance.sort_values

player_results1=pd.DataFrame(rf_predict,columns=['PTS'])
player_sub1=new_players['Player']
predictions1=pd.concat([player_sub,player_results1],axis=1)
predictions1['PTS'].max() 
predictions1['PTS'].sort_values()

#standardize the data
scaler=StandardScaler().fit(X)
X_old_scale=scaler.transform(X)
X_new_scale=scaler.transform(X1)
y=old_players['VORP'].values 

random_forest_scale=RandomForestClassifier(n_estimators=500) 
random_forest_scale.fit(X_old_scale,y) 

#predict on the 2018 nba draft class
current_predict1=random_forest_scale.predict(X_new_scale)
player_results2=pd.DataFrame(current_predict1,columns=['VORP'])
player_sub2=new_players['Player']
predictions2=pd.concat([player_sub2,player_results2],axis=1)
predictions2

#predict points 
rf_reg.fit(X_old_scale,y)
current_predict2=rf_reg.predict(X_new_scale)

player_results3=pd.DataFrame(current_predict2,columns=['PTS'])
player_sub3=new_players['Player']
predictions3=pd.concat([player_sub3,player_results3],axis=1)
predictions3

new_players
old_players 
vorp_new=pd.read_csv("nba_current_vorp.csv")
vorp_new1=vorp_new[["Player","VORP"]]

#merge old_players with updated vorp (over first 5 seasons) 
old_players1=pd.merge(old_players,vorp_new1,on="Player") #237,27 (VORP_y)

#random forest regression 
X=old_players1.iloc[:,1:25]
y=old_players1['VORP_y']
X1=new_players.iloc[:,1:25]

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=344)
rf_reg.fit(X,y)
rf_predict=rf_reg.predict(X1)
feature_importance=rf_reg.feature_importances_ 
rf_importance = pd.DataFrame(feature_importance, index=X.columns, 
                          columns=["Importance"])
rf_importance.sort_values
rf_importance.to_csv("plum.csv")

#predict 2018 VORP
predict_vorp=rf_reg.predict(X1)
player_results_vorp=pd.DataFrame(predict_vorp,columns=['VORP'])
player_sub=new_players['Player']
predictions=pd.concat([player_sub,player_results_vorp],axis=1)
predictions.sort_values('VORP',ascending=False) 
predictions.to_csv("pred_vorpy.csv")

scaler=StandardScaler().fit(X)
X_old_scale=scaler.transform(X)
X_new_scale=scaler.transform(X1)
y=old_players1['VORP_y'].values 

rf_reg1=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=344)
rf_reg1.fit(X_old_scale,y)
rf_predict1=rf_reg.predict(X_new_scale)

player_results_vorp1=pd.DataFrame(rf_predict1,columns=['VORP'])
player_sub=new_players['Player']
predictions1=pd.concat([player_sub,player_results_vorp1],axis=1)
predictions1 
pred1=predictions1.sort_values('VORP',ascending=False)
pred1.to_csv("pred1.csv")

new_players.iloc[3]
player_search = [col for col in new_players.Player if 'Brunson' in col]
new_players.iloc[0,:]
new_players.sort_values('FGA',ascending=False)

new_fga=new_players[['Player','FGA']]
new_fga.sort_values('FGA')


