---
title: "Regressione_new"
author: "Alessandro Wiget"
date: "`r Sys.Date()`"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Librerie

```{r, results='hide', warning= FALSE, message=FALSE}
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
#library(glmulti)
library(AICcmodavg)
library(glmtoolbox)
library(caret)
library(pROC)
library(e1071)
library(stats)



```

## Il Dataset

Prima di tutto definiamo la working directory:

IMPORTANTE! Cambiare la directoy a seconda del pc.

Importiamo il Dataset, presente nella cartella `Dati/`:

```{r}
setwd("C:/Users/sanni/OneDrive/Desktop/POLIMI/3 anno/inferenza/Progetto-Inferenza-Statistica-main/Code")
df <- read_excel("../Dati/Dropout20240226_IngMate.xlsx")
View(df)
```

## Regressione Logistica

Consideriamo innanzitutto solo gli studenti con carriere terminate, cioè o che si sono laureati o che hanno abbandonato il corso di studio:

```{r}
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
df$highschool_type_code <- NULL #abbiamo cancellato queste variabili operche possiamo separare fra classico, scientifico e altro con un'altra variabile
df$stud_admis_convent_start_dt <- NULL
df$stud_career_end_ay <-NULL

filtered_df <- df %>% filter(stud_career_status != 'A')
```

Selezioniamo dal dataset le variabili numeriche:

```{r}
numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
numerical_df = na.omit(numerical_df)
```

Osserviamo se esistono correlazioni significative fra i dati numerici:

```{r}
X = numerical_df[, -4]
corrplot(cor(X), method='color')
```

## La Prima Regressione Logistica

Effettuiamo la regressione logistica fra le variabili numeriche del dataset, e vediamo quanto vale inizialmente l'adjustedR2:

```{r}


# Create a formula for linear model
formula_num <- as.formula(paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + ")))

# Fit the linear model
model_init <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
summary(model_init)

pseudo_r2 <- pR2(model_init)

pseudo_r2['McFadden']
```

Iniziamo da un valore di adjustedR2 di 0.656, quindi già buono, vediamo adesso di trovare un buon modello logistico, che dunque minimizzi l'AIC.

Restringiamoci al miglior modello per ogni numero di variabli (l'intercetta conta come variabile extra), e mostriamo anche i rispettivi adjustedR2:

Non notiamo un peggioramento troppo elevato né dell'AIC né dell'adjR2 utilizzando il modello a 5 covariate, quindi prendiamo in considerazione quest'ultimo.

Effettuiamo una ricerca backward dal modello finale per cercare di trovare un modello simile ma con un processo più logico.

```{r}
covariate = paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + "))

#Covariate rimosse durante la semplificazione, in ordine

formula_num <- as.formula(covariate)

# Fit the linear model
model_back <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")

#-----------------------------
covariate = paste(covariate, "- stud_admission_score")

formula_num <- as.formula(covariate)

# Fit the linear model
model_back <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
#-------------------------------
covariate = paste(covariate, "- exa_avg_attempts")

formula_num <- as.formula(covariate)

# Fit the linear model
model_back <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
#-----------------------------------------
covariate = paste(covariate, "- career_start_ay")

formula_1 <- as.formula(covariate)

# Fit the linear model
model_1 <- glm(formula_1, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
#------------------------------------------
covariate = paste(covariate, "- stud_career_admission_age")

formula_2 <- as.formula(covariate)

# Fit the linear model
model_2 <- glm(formula_2, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")

anova(model_1, model_2, test="Chisq")
```

Aggiungere che teniamo il modello con 5 covariate, rifiutando il modello con 4 per via del tenst sulle devianze.

## Analisi dei Punti Influenti

Valutiamo l'impatto sul modello di eventuali punti leva e outliers. Partiamo dai punti leva:

```{r}
Z=model.matrix(model_1) #matrice disegno, i leverages su diag princ
# __Rule of thumb:__ Given a point h_ii diagonal element of H, the i-th observation is a leverage if:
#  h_ii > 2*(p)/n
lev=hat(Z) #h_ii
p = model_1$rank #nro covariate+1  
n = dim(Z)[1] #nro osservazioni

#plot dei leverages in funzione dell'osservazione
plot(model_1$fitted.values, lev, ylab = "Leverages", main = "Plot of Leverages", 
      pch = 16, col = 'black' )
abline( h = 2 * p/n, lty = 2, col = 'red' ) #COSTRUISCO LEVA PER VEDERE SE QUALCUNO SUPERA
watchout_points_lev = lev[ which( lev > 2 * p/n  ) ]
watchout_ids_lev = seq_along( lev )[ which( lev > 2 * p/n ) ] ## identify the rows relative to leverage points
points( model_1$fitted.values[ watchout_ids_lev ], watchout_points_lev, col = 'red', pch = 16 )

sum( lev [ lev >  2 * p / n ] ) # 1.36234/5= 0.272468 i leverages pesano quasi il 30 percento

```

```{r}
no_lev_df = numerical_df[which(!((numerical_df$career_time_conv > 1000 & numerical_df$exa_cfu_pass==0)| numerical_df$career_time_conv<0)),]

model_no_lev = glm("dropout ~ stud_career_admission_age + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv  ", data=no_lev_df, family=binomial)
summary(model_no_lev)
```

```{r}
drop1(model_no_lev, test="Chisq")

model_no_lev_opt = update(model_no_lev, . ~ . - stud_career_admission_age)

anova(model_no_lev, model_no_lev_opt, test="Chisq")

summary(model_no_lev_opt)
```

## Introduzione Interazioni fra Variabili Numeriche

Prima di procedere con l'aggiunta di variabili categoriche cerchiamo di comprendere se le aggiunte di interazioni fra variabili numeriche ci permettono di migliorare il nostro modello. Osservando la matrice delle covariate costruita all'inizio possiamo osservare che `career_start_ay` e `stud_career_end_ay` sono estremamente correlate, aggiungiamo quindi al modello: `career_start_ay*stud_career_end_ay`. Seguendo lo stesso ragionamento un'altra coppia di covariate che appaiono essere molto correlate sono `exa_cfu_pass` e `exa_grade_average`, introduciamo la loro interazione `exa_cfu_pass*exa_grade_average`:

```{r}
model_int_grade = glm("dropout ~ 1 + stud_career_admission_age + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv + exa_cfu_pass*exa_grade_average", data=no_lev_df, family=binomial)

gvif(model_1)
gvif(model_int_grade)

anova(model_no_lev_opt, model_int_grade, test="Chisq")


```

Non accetto il nuovo modello con l'interazione poichè il GVIF è troppo elevato.

## Introduzione delle Variabili Categoriche

Rendiamo tutte le variabili del Dataset di tipo `factor` affinchè siano utilizzabili nella regressione logistica.

```{r}
MODEL = model_no_lev_opt

filtered_df <- df %>% filter(stud_career_status != 'A')
filtered_df_no_na = na.omit(filtered_df)

#Partendo dal modello di ottimo trovato prima costruisco la matrice solo con quelle covariate:
cat_df <- filtered_df_no_na

cat_no_lev_df = cat_df[which(!((cat_df$career_time_conv > 1000 & cat_df$exa_cfu_pass==0)| cat_df$career_time_conv<0)),]

cat_no_lev_df$stud_gender = factor(cat_no_lev_df$stud_gender, ordered=F)
cat_no_lev_df$previousStudies = factor(cat_no_lev_df$previousStudies, ordered=F)
cat_no_lev_df$origins = factor(cat_no_lev_df$origins, ordered=F)
cat_no_lev_df$income_bracket_normalized_on4 = factor(cat_no_lev_df$income_bracket_normalized_on4, ordered=F)


cat_no_lev_df$dropped_on_180<-NULL
cat_no_lev_df$stud_career_status <-NULL


View(cat_no_lev_df)
```

```{r}
male = cat_no_lev_df[which(cat_no_lev_df$stud_gender=="M"),]
male_mean = mean(male$dropout)
male_mean

female = cat_no_lev_df[which(cat_no_lev_df$stud_gender=="F"),]
female_mean = mean(female$dropout)
female_mean

# Facciamo uno z-test con:
#                       H0: pM == pF
#                       H1: pM != pF

p_hat = (sum(female$dropout) + sum(male$dropout))/(nrow(cat_no_lev_df))

z_test = (male_mean - female_mean)/sqrt(p_hat*(1-p_hat)*(1/nrow(male) + 1/nrow(female)))

p_value = 2*pnorm(z_test)
```

Non posso rifiutare H0. Non noto una differenza di probabilità di dropout fra maschi e femmine. Non aggiungiamo questa categoria nel modello finale.

```{r}
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$previousStudies, data=cat_no_lev_df)

summary(model_anova)
```

```{r}
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$origins, data=cat_no_lev_df)

summary(model_anova)
```

```{r}
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$income_bracket_normalized_on4, data=cat_no_lev_df)

summary(model_anova)
```

Aggiungere dei boxplot

```{r}
model_cat = glm("dropout ~ 1 + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv + previousStudies + origins + income_bracket_normalized_on4", data=cat_no_lev_df, family=binomial)


drop1(model_cat, test="Chisq")

model_cat = update(model_cat,  . ~ . - origins)
drop1(model_cat, test="Chisq")

model_cat = update(model_cat,  . ~ . - income_bracket_normalized_on4)
drop1(model_cat, test="Chisq")

model_cat = update(model_cat,  . ~ . - previousStudies)
drop1(model_cat, test="Chisq")
```

Abbiamo trovato che il modello non ha bisogno di covariate categoriche.

## Confusion Matrix

```{r}


# Assume you have a dataframe data with predictors and a response variable
# split the data into train and test
set.seed(123)
trainIndex <- createDataPartition(no_lev_df$dropout, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

train <- no_lev_df[ trainIndex,]
test  <- no_lev_df[-trainIndex,]

# Fit the logistic regression model
fit <- glm(dropout ~ 1+exa_cfu_pass+exa_grade_average+highschool_grade+career_time_conv     , data = train, family = binomial)

# Make predictions on the test set
predicted_probs <- predict(fit, newdata = test, type = "response")
#

# Compute the confusion matrix
#predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)
#cm <- confusionMatrix(predicted_classes, test$response)
#print(cm)

# Compute the ROC curve
roc_obj <- roc(test$dropout, predicted_probs)

# Plot the ROC curve
plot(roc_obj, print.auc = TRUE)

# Compute the AUC
auc <- auc(roc_obj)
print(auc)

#trovare soglia opt
best=coords(roc_obj, "best", ret="threshold", best.method="youden")
print(best)
#
test$dropout=as.factor(test$dropout)

# Compute the confusion matrix
predicted_classes <- ifelse(predicted_probs > 0.1915817, 1, 0)
predicted_classes=as.factor(predicted_classes)

cm <- confusionMatrix(predicted_classes, test$dropout)
print(cm)

```