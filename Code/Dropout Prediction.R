
#Regressione Logistica Senza Parametro Temporale
#Alessandro Wiget, Sofia Sannino, Pietro Masini, Giulia Riccardi

## Librerie

library(readxl)
library(dplyr)
library( faraway )
library( leaps )
library(MASS)
library( GGally)
library(BAS)
library(rgl)
library(corrplot)
library(pscl)
library(plm)
library(AICcmodavg)
library(caret)
library(pROC)
library(e1071)
library(stats)
setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto/Code")

# Il Dataset

#Prima di tutto definiamo la working directory:
  
#IMPORTANTE! Cambiare la directoy a seconda del pc.

#Importiamo il Dataset, presente nella cartella `Dati/`:

setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto/Code")
library(readxl)
df <- read_excel("../Dati/Dropout20240226_IngMate.xlsx")
View(df)


# Analisi esplorativa


#togliamo covariate inutili, togliamo gli attivi dei quali non sappiamo ancora se hanno droppato o no.
df$career_anonymous_id <- NULL
df$career_time <- NULL
df$stud_career_degree_start_id <- NULL
df$stud_career_degree_changed <- NULL
df$stud_career_degree_name <- NULL
df$stud_ofa_flst <- NULL
df$stud_ofa_fltp <- NULL
df$stud_career_degree_area <- NULL
df$stud_career_degree_code <- NULL
df$stud_career_degree_code_CdS <-NULL
df$highschool_type <- NULL
hist(df$stud_career_admission_age,main="Età di ingresso degli studenti", xlab="Età", ylab="Numero studenti")
df$stud_career_admission_age<-NULL

df$highschool_type_code <- NULL #abbiamo cancellato queste variabili perchè possiamo separare fra classico, scientifico e altro con un'altra variabile
df$stud_admis_convent_start_dt <- NULL
df$stud_career_end_ay <-NULL
#teniamo solo le persone che non sono attive e delle quali sappiamo già se hanno droppato o no
filtered_df <- df %>% filter(stud_career_status != 'A')

#selezioniamo solo le variabili numeriche
numerical_vars <- sapply(filtered_df, is.numeric) 
numerical_df <- filtered_df[, numerical_vars]
numerical_df = na.omit(numerical_df)

#analisi esplorativa dei dati: notiamo che ci sono persone che non hanno iniziato il poli e in più ci sono persone che sono iscritte da 3 anni o più 
plot(numerical_df$exa_cfu_pass, numerical_df$career_time_conv,,xlab="cfu sostenuti",ylab="giorni di iscrizione")
abline( h=1200, lty = 2, col = 'red' )

#RITENIAMO A QUESTO PUNTO LA VARIABILE TEMPO NON RILEVANTE PERCHE' A NOI INTERESSA PREVEDERE IL DROPOUT ED EVITARLO, PROGETTANDO UN'AZIONE BASATA SU DATI DEL PRIMO SEMESTRE DEL PRIMO ANNO E PORTATA AVANTI IMMEDIATATAMENTE IL PRIMO ANNO STESSO. IL FATTO CHE LA CARRIERA DURI PIU O MENO E' FUORI CONTESTO: NOI VOGLIAMO CHE Y=0. 
numerical_df = numerical_df[which(!((numerical_df$career_time_conv > 1300)| numerical_df$career_time_conv<0)),] 
#numerical_df = numerical_df[which(!(numerical_df$career_time_conv<0)),]
numerical_df$career_time_conv <- NULL

View(numerical_df)

X = numerical_df[, -3]
X=X[,-1]
corrplot(cor(X), method='color')


#Notiamo una correlazione importante tra `exa_cfu_pass` ovvero i cfu sostenuti e `exa_grade_average` e tra media esami e tentativi medi esami.

min(numerical_df$exa_cfu_pass)
max(numerical_df$exa_cfu_pass)

x  = c( 0, 7, 10, 17, 20, 27, 40) #supporto cfu

# Calcoliamo i punti medi degli intervalli che abbiamo creato
mid = c( ( x [ 2:7 ] + x [ 1:6 ] )/2 )

# Suddividiamo i dati nelle classi che abbiamo creato
GRAGE = cut( numerical_df$exa_cfu_pass, breaks = x, include.lowest = TRUE, right = FALSE )
GRAGE

# Calcoliamo quindi la media della variabile AGE stratificata e sovrapponiamo 
# i valori di y al grafico precedente.

y = tapply( numerical_df$dropout, GRAGE, mean )
y

#Aggiungere legenda
plot( numerical_df$exa_cfu_pass, numerical_df$dropout, pch = ifelse( numerical_df$dropout  == 1, 3, 4 ),
      col = ifelse( numerical_df$dropout== 1, 'forestgreen', 'red' ),
      xlab = 'cfu', ylab = 'dropout', main = 'cfu vs. dropout', lwd = 2, cex = 1.5 )
points( mid, y, col = "blue", pch = 16 )


# La Prima Regressione Logistica

#Effettuiamo una prima regressione logistica fra le variabili numeriche del dataset:
  
# Create a formula for linear model
formula_num <- as.formula(paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + ")))

# Fit the linear model
model_init <- glm(formula_num, data = numerical_df, family=binomial( link = logit ))

# Print the summary of the model
summary(model_init)

covariate = paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + "))

#Covariate rimosse durante la semplificazione, in ordine

formula_num <- as.formula(covariate)

# Fit the linear model
model_back <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)
#--------------------------------
model_back = update(model_back, . ~ . -stud_admission_score )

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)
#-----------------------------
model_back = update(model_back, . ~ . - career_start_ay)

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)
#-------------------------------
model_back = update(model_back, . ~ . - stud_career_admission_age)

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)
#-------------------------------
model_back = update(model_back, . ~ . - exa_avg_attempts )

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)
#-------------------------------
model_back_2 = update(model_back, . ~ . - highschool_grade)

# Print the summary of the model
drop1(model_back, test="Chisq")
AIC(model_back)

drop1(model_back_2, test="Chisq")
AIC(model_back)
#-------------------------------

anova(model_back_2, model_back, test="Chisq")

mod=glm("dropout~1+exa_cfu_pass+exa_grade_average+highschool_grade",data = numerical_df, family=binomial)
summary(mod)


## Introduzione Interazioni fra Variabili Numeriche

mod_int = update(mod, . ~ . + exa_cfu_pass*exa_grade_average)

summary(mod_int)
anova(mod,mod_int)

## Introduzione delle Variabili Categoriche

MODEL = mod

filtered_df <- df %>% filter(stud_career_status != 'A')
filtered_df_no_na = na.omit(filtered_df)

#Partendo dal modello di ottimo trovato prima costruisco la matrice solo con quelle covariate:
cat_df <- filtered_df_no_na

cat_no_lev_df = cat_df[which(!((cat_df$career_time_conv > 1300)| cat_df$career_time_conv<0)),]
#cat_no_lev_df=cat_df[which(!(cat_df$career_time_conv<0)),]
cat_no_lev_df$career_time_conv <- NULL

cat_no_lev_df$stud_gender = factor(cat_no_lev_df$stud_gender, ordered=F)
cat_no_lev_df$previousStudies = factor(cat_no_lev_df$previousStudies, ordered=F)
cat_no_lev_df$origins = factor(cat_no_lev_df$origins, ordered=F)
cat_no_lev_df$income_bracket_normalized_on4 = factor(cat_no_lev_df$income_bracket_normalized_on4, ordered=F)


cat_no_lev_df$dropped_on_180<-NULL
cat_no_lev_df$stud_career_status <-NULL


View(cat_no_lev_df)

model_cat = glm("dropout ~ 1 + 
    exa_cfu_pass + exa_grade_average  +stud_gender + previousStudies + origins + income_bracket_normalized_on4+ highschool_grade", data=cat_no_lev_df, family=binomial)

drop1(model_cat, test="Chisq")

model_cat = update(model_cat,  . ~ . - origins)
drop1(model_cat, test="Chisq")


model_cat = update(model_cat,  . ~ . - income_bracket_normalized_on4)
drop1(model_cat, test="Chisq")

model_cat = update(model_cat,  . ~ . - stud_gender)
drop1(model_cat, test="Chisq")
anova(model_cat,mod)
model_cat = update(model_cat,  . ~ . - previousStudies)
drop1(model_cat, test="Chisq")

summary(model_cat)

## Confusion Matrix e ROC Curve

set.seed(123)
trainIndex <- createDataPartition(cat_no_lev_df$dropout, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

train <- cat_no_lev_df[ trainIndex,]
test  <- cat_no_lev_df[-trainIndex,]

fit <- glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average + highschool_grade", data = train, family = binomial)

predicted_probs <- predict(fit, newdata = test, type = "response")

roc_obj <- roc(test$dropout, predicted_probs)

plot(roc_obj, print.auc = TRUE)

auc <- auc(roc_obj)
print(auc)

#trovare soglia opt
best=coords(roc_obj, "best", ret="threshold", best.method="youden") #best circa 0.2
print(best)

test$dropout=as.factor(test$dropout)

predicted_classes <- ifelse(predicted_probs > as.numeric(best), 1, 0)
predicted_classes=as.factor(predicted_classes)

cm <- confusionMatrix(predicted_classes, test$dropout)
print(cm)

attivi_df <- df %>% filter(stud_career_status == 'A')
attivi_df$dropout <- NULL

attivi_df$stud_career_end_date <- NULL
attivi_df = na.omit(attivi_df)
#attivi_df = attivi_df[which(!((attivi_df$career_time_conv > 1000 & attivi_df$exa_cfu_pass==0) | attivi_df$career_start_ay!=2023 | attivi_df$career_time_conv<177)),]
attivi_df = attivi_df[which((attivi_df$career_start_ay==2023)|(attivi_df$career_start_ay==2022)|(attivi_df$career_start_ay==2021)),]
#attivi_df$stud_gender = factor(attivi_df$stud_gender, ordered=F)
#attivi_df$previousStudies = factor(attivi_df$previousStudies, ordered=F)

View(attivi_df)

model_opt = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average ", data = train, family=binomial)
summary(model_opt)

predizione <- predict(model_opt, newdata = attivi_df, type = "response")

binary_output <- ifelse(predizione > round(as.numeric(best),1), 1, 0)

attivi_df$dropout_prediction = binary_output

sum(binary_output)/length(binary_output)

View(attivi_df)

#K-fold cross Validation

set.seed(123) 
# definire il numero di folds
K <- 10
sensitivity_value1=rep(0,10)
sensitivity_value2=rep(0,10)
sensitivity_value3=rep(0,10)
sensitivity_value4=rep(0,10)

cat_no_lev_df <- cat_no_lev_df[sample(nrow(cat_no_lev_df)),]
View(cat_no_lev_df)
folds <- cut(seq(1, nrow(cat_no_lev_df)), breaks=K, labels=FALSE)


# Cross validazione
for(i in 1:K) {
  # segmentazione dei dati
  test_indices <- which(folds == i, arr.ind=TRUE)
  test_data <- cat_no_lev_df[test_indices, ]
  train_data <- cat_no_lev_df[-test_indices, ]
  # Allenare il modello
 model1 = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average ", data = train_data, family = binomial)
model2 = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average+highschool_grade", data = train_data, family = binomial)
model3 = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average + highschool_grade + exa_cfu_pass*exa_grade_average", data = train_data, family = binomial)
model4 = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average + previousStudies + highschool_grade", data=train_data, family=binomial)

  
  predictions1 <- predict(model1, test_data, type="response")
  predictions2 <- predict(model2, test_data, type="response")
  predictions3 <- predict(model3, test_data, type="response")
  predictions4 <- predict(model4, test_data, type="response")
  roc_obj1 <- roc(test_data$dropout, predictions1)
  b1=coords(roc_obj1, "best", ret="threshold", best.method="youden")
  roc_obj2 <- roc(test_data$dropout, predictions2)
  b2=coords(roc_obj2, "best", ret="threshold", best.method="youden")
  roc_obj3 <- roc(test_data$dropout, predictions3)
  b3=coords(roc_obj3, "best", ret="threshold", best.method="youden")
  roc_obj4 <- roc(test_data$dropout, predictions4)
  b4=coords(roc_obj4, "best", ret="threshold", best.method="youden")
  
  predicted_classes1 <- ifelse(as.numeric(predictions1) > as.numeric(b1), 1, 0)
  predicted_classes2 <- ifelse(as.numeric(predictions2) > as.numeric(b2), 1, 0)
  predicted_classes3 <- ifelse(as.numeric(predictions3) > as.numeric(b3), 1, 0)
  predicted_classes4 <- ifelse(as.numeric(predictions4) > as.numeric(b4), 1, 0)
  predicted_classes1=as.factor(predicted_classes1)
  predicted_classes2=as.factor(predicted_classes2)
  predicted_classes3=as.factor(predicted_classes3)
  predicted_classes4=as.factor(predicted_classes4)
  test_data$dropout=as.factor(test_data$dropout)
  cm1 <- confusionMatrix(predicted_classes1, test_data$dropout)
  cm2<- confusionMatrix(predicted_classes2, test_data$dropout)
  cm3<- confusionMatrix(predicted_classes3, test_data$dropout)
  cm4<- confusionMatrix(predicted_classes4, test_data$dropout)
  sensitivity_value1[i] <- cm1$byClass["Sensitivity"]
  sensitivity_value2[i] <- cm2$byClass["Sensitivity"]
  sensitivity_value3[i] <- cm3$byClass["Sensitivity"]
  sensitivity_value4[i] <- cm4$byClass["Sensitivity"]
}
boxplot(sensitivity_value1,sensitivity_value2,sensitivity_value3,sensitivity_value4,xlab="Modelli", ylab="Sensitività")




