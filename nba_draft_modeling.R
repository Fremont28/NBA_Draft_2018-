#import libraries 
library(ggplot2)
library(dplyr)
library(plyr)

#Predicting 2017 NBA draft prospects NBA VORP based on physical features and college statistics
#using linear mixed models, linear regression, and logistic regression 

#read-in csv file 
draft_anth=read.csv("DraftClasselite.csv")
draft_17=read.csv("17_draft.csv")
draft_17$vorp_plus=ifelse(draft_17$VORP>=0.21,1,0)
draft_final=merge(draft_anth,draft_17,by="Player")

hist(draft_anth$Wingspan) #histogram wingspaon 

#predicting VORP (value above replacement based on physical features)
model1=glm(vorp_plus~Height.wo.shoes+Body.Fat..+
             Hand.Length+Hand.Width+
             Standing.Reach.Height.w.shoes+
             Weight+Wingspan+
             Lane.Agiility+standing.vertical+
             max.vertical.leap,data=draft_final)


#predict gives the predicted value in terms of logits
plot_dat <- data.frame(prob = draft_final$vorp_plus/30,
                       height_wo = draft_final$Height.wo.shoes,
                       fit = predict(model1, draft_final))

#convert logit values to probabilities
plot_dat$fit<-exp(plot_dat$fit/(1+exp(plot_dat$fit)))
ggplot(plot_dat,aes(x=height_wo,y=prob))+
  geom_point()+geom_line(aes(x=height_wo,y=prob))

# an odds ratio of 1 indicates no change 
exp(coef(model1)) #a one unit increase in height decreases the odds of having an above average vorp by a factor of 0.06 

#summary metrics 
IQR(draft_anth$Body.Fat..)
#dot plot
ggplot(draft_anth,aes(x=Wingspan))+geom_dotplot()
ggplot(draft_anth,aes(x=max.vertical.leap))+geom_histogram(bins=10)

#within subject variability (based on conference)
for(i in 1:ncol(draft_final)){
  draft_final[is.na(draft_final[,i]), i] <- mean(draft_final[,i], na.rm = TRUE)
}

draft_final$Power<-as.factor(draft_final$Power)


#linear mixed model -----------
m1=lme(VORP~Height.wo.shoes+Body.Fat..+
         Hand.Length+Hand.Width+
         Standing.Reach.Height.w.shoes+
         Weight+Wingspan+
         Lane.Agiility+standing.vertical+ #variance is allowed to var by position 
         max.vertical.leap+POS.x,random=~1|Conf,data=draft_final) #random=~1|Conf

#coefficients for linear mixed model 
coef(summary(m1))
#coeffients for random effects 
coef(ranef(m1))

#allow for variance among positions 
m2=lme(VORP~Height.wo.shoes+Body.Fat..+
         Hand.Length+Hand.Width+
         Standing.Reach.Height.w.shoes+
         Weight+Wingspan+
         Lane.Agiility+standing.vertical+ #variance is allowed to var by position 
         max.vertical.leap+POS.x,random=~1|Conf,weights=varIdent(form=~1|POS.x),data=draft_final)

#fixed effects
coef(summary(m2))
#coeffcients for random effects 
coef(ranef(m2))

#---------
m3=lmer(VORP~Height.wo.shoes+Body.Fat..+
          Hand.Length+Hand.Width+
          Standing.Reach.Height.w.shoes+
          Weight+Wingspan+
          Lane.Agiility+standing.vertical+ #variance is allowed to var by position 
          max.vertical.leap+(max.vertical.leap|POS.x),draft_final)


#i. extract indiviudal (or group differences) 
#i.e., the fixed effect coefficients are added to the random effects. The results are individual intercepts and slopes.
coef(m3)$POS.x 
#ii.coeff fixed effects (mean for all subjects)
coef(summary(m3))[ , "Estimate"]
#iii. random effects
ranef(m3)$POS.x 
colMeans(ranef(m3)$POS.x)

anova(m1,m2) #p>0.05 indicates no difference in vorp among positions? 

##bayesian modeling- can the VORP be predicted physical and college numbers? 

#i. correlation matrix---- 
corr_data=draft_final[,6:22]
corr_data1=draft_final[,24:34]
corr_data2=draft_final[,55]
names(corr_data2)[1]<-"VORP"
corrs=cbind(corr_data,corr_data1,corr_data2)
res<-cor(corrs)

dff<-draft_final 
#total rebounds and VORP 0.393 r
#VORP distn 
ggplot(dff, aes(x=VORP)) +
  geom_histogram(fill="lightblue", alpha = 0.7,bins=10)+
  theme_bw()+
  labs(x = "VORP", y= "Count", title = "Distribution of Value Above Replacement")

#standing vertical by position 
ggplot(dff, aes(x=dff$POS.x, y = dff$standing.vertical))+
  geom_boxplot(alpha = 0.7)+
  theme_bw()+
  labs(x = "Position", y= "Standing Vertical (feet)")+ 
  theme(axis.text.x=element_text(angle=90, hjust = 1, vjust = 0))+
  theme(legend.position="none")


train=corrs[0:20,]
test=corrs[20:30,]
test=test[,1:28]

#i. multiple linear regression model (recibi parsimonious model??)
full_model<-lm(corr_data2~.,data=corrs)

#ii. step 1 model
step1_model<-lm(corr_data2~Lane.Agiility+Standing.Reach.Height.w.shoes+max.vertical.leap+
                  three.quarter.sprint,data=train)

#collinearity (VIF>10 is threshold)
car::vif(step1_model)

#predictions 
test$pred_vorp<-predict(step1_model,test)
sub_dff=dff[20:30,]
players=subset(sub_dff,select=c("Player","VORP"))
final_proj=cbind(players,test$pred_vorp)
write.csv(final_proj, file="rumX.csv")

#MSE 
#i. linear relationship entre response and predictors 
ggplot(augment(step1_model), aes(x = corrs$max.vertical.leap, y = .resid))+
  geom_point(colour = "red", size = 2, alpha = 0.5)+
  theme_classic()+
  geom_smooth(se=FALSE)+
  labs(x = "Max Vertical Leap ", y= "residuals")+
  geom_hline(yintercept = 0, linetype = "dashed")

#ii. qq plot 
par(mfrow=c(1,2))
hist(step1_model$residuals, main = "Residual Distribution", 
     xlab = "Residuals", col = "lightgreen")
qqnorm(step1_model$residuals, col = "blue")
qqline(step1_model$residuals, col = "red")

#iii. residuals are equally variable for high and low variables for the fitted predicted response
ggplot(augment(step1_model), aes(x= .fitted, y= .resid))+
  geom_point(colour = "blue", size = 2, alpha = 0.5)+
  theme_classic()+
  geom_smooth()+
  geom_hline(yintercept = 0, linetype = "dashed")+
  labs(x = "Fitted", y = "Residuals")

#iv. pattern in the residuals 
ggplot(augment(step1_model), aes(x = seq_along(.resid), y = .resid)) + 
  geom_point(colour = "lightgreen", size = 2.5, alpha = 0.6)+
  theme_classic()+
  labs(x = "Order of Collection", y = "Residuals", title = "Residuals vs. Order of Collection")


