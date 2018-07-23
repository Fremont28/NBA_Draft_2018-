#7/16/18
### cleaning text
draft1=read.csv("2018_metrics.csv")

#remove text 
test=gsub('',,draft1$Height.wo.shoes)
dat <- draft1[!is.na(as.numeric(as.character(draft1$Height.wo.shoes))),]
dat1=gsub("\\D+","",draft1$Height.wo.shoes)
Height.wo.shoes=as.data.frame(dat1)
names(Height.wo.shoes)[1]<-"Height.wo.shoes"
Height.wo.shoes$Height.wo.shoes=as.numeric(as.character(Height.wo.shoes$Height.wo.shoes))

#round height without shoes
library(stringr)
Height.wo.shoes=substr(Height.wo.shoes$Height.wo.shoes,start=1,stop=2)
Height.wo.shoes=as.data.frame(Height.wo.shoes)
names(Height.wo.shoes)[1]="Height.wo.shoes"

#remove % symbol
bodyfat=gsub('[%]','',draft1$Body.Fat..)
bodyfat=as.data.frame(bodyfat)
names(bodyfat)[1]="Body.Fat.."

#visualizations 
nba_metrics=read.csv("2018_metrics.csv")

#Wingspan by Position 
library(ggthemes)
library(ggplot2)

g<-ggplot(nba_metrics,aes(POS.x,Wingspan))
g+geom_tufteboxplot()+
  theme(axis.text.x=element_text(angle=65,vjust=0.6))+
  labs(title="Centers Have Imposing Wingspans",caption="Source: Basketball Reference",
       x="Position",y="Wingspan (feet)")+
  theme(plot.title = element_text(hjust = 0.5))

vorp_pred=read.csv("pred_vorpy.csv")
head(vorp_pred)

g1=ggplot(vorp_pred,aes(x=reorder(Player,-VORP),y=VORP,fill=Player))
g1+geom_bar(varwidth=T, fill="plum",stat="identity") + 
  labs(title="Trae Young Projects the Highest VORP Over His First Five Seasons", 
       caption="Source: Basketball Reference",
       x="",
       y="Predicted VORP")+theme(axis.text.x=element_text(angle=65,vjust=0.6))+
  theme(plot.title = element_text(hjust = 0.5))


rf_imp=read.csv("plum.csv")  
g2=ggplot(rf_imp,aes(x=reorder(X,-Importance),y=Importance,fill=Importance))
g2+geom_bar(varwidth=T, fill="orange",stat="identity") + 
  labs(title="Could Steals Be the Most Important Indicator of NBA Success?", 
       caption="Source: Basketball Reference",
       x="",
       y="Variable Importance")+theme(axis.text.x=element_text(angle=65,vjust=0.6))+
  theme(plot.title = element_text(hjust = 0.5))


