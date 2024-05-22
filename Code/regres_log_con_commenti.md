Regressione Logistica Senza Parametro Temporale
================
Alessandro Wiget, Sofia Sannino, Pietro Masini, Giulia Riccardi
2024-05-22

## Librerie

``` r
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
#library(glmtoolbox)
library(caret)
library(pROC)
library(e1071)
library(stats)
setwd("C:/Users/lenovo/OneDrive/Documenti/R")
```

## Il Dataset

Prima di tutto definiamo la working directory:

IMPORTANTE! Cambiare la directoy a seconda del pc.

Importiamo il Dataset, presente nella cartella `Dati/`:

``` r
setwd("C:/Users/lenovo/OneDrive/Documenti/R")
library(readxl)
df <- read_excel("Dropout20240226_IngMate.xlsx")
#View(df)
```

## Regressione Logistica

Consideriamo innanzitutto solo gli studenti con carriere terminate, cioè
o che si sono laureati o che hanno abbandonato il corso di studio:

``` r
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
df$highschool_type_code <- NULL #abbiamo cancellato queste variabili operche possiamo separare fra classico, scientifico e altro con un'altra variabile
df$stud_admis_convent_start_dt <- NULL
df$stud_career_end_ay <-NULL
#teniamo solo le persone che non sono attive e delel quali sappiamo già se hanno droppato o no
filtered_df <- df %>% filter(stud_career_status != 'A')
```

Selezioniamo dapprima dal dataset le variabili numeriche, inoltre
operiamo la nostra analisi su studenti in una certa finestra temporale,
eliminando coloro che si erano immatricolati e poi non hanno iniziato
gli studi al Politecnico e coloro che sono iscritti da più di tre anni e
non hanno conseguito alcun esame:

``` r
numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
numerical_df = na.omit(numerical_df)
#analisi esplorativa dei dati: notiamo che ci sono persone che non hanno iniziato il poli e in più ci sono persone che sono iscritte da 3 anni o più senza dare esami, ci sono persone iscritte da più di 4,5 anni
plot(numerical_df$exa_cfu_pass, numerical_df$career_time_conv)
abline( h=1200, lty = 2, col = 'red' ) 
```

![](regres_log_con_commenti_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
numerical_df = numerical_df[which(!((numerical_df$career_time_conv > 1000 & numerical_df$exa_cfu_pass==0)| numerical_df$career_time_conv<0)),]
#numerical_df = numerical_df[which(!(numerical_df$career_time_conv<0)),]
numerical_df$career_time_conv <- NULL

#View(numerical_df)
```

Osserviamo se esistono correlazioni significative fra le variabili numeriche:

``` r
X = numerical_df[, -4]
corrplot(cor(X), method='color')
```

![](regres_log_con_commenti_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Notiamo una correlazione importante tra cfu sostenuti e media esami e
tra media esami e tentativi medi esami.

## La Prima Regressione Logistica

Effettuiamo la regressione logistica fra le variabili numeriche del
dataset, e vediamo quanto vale inizialmente lo pseudo adjustedR2 (indice
di McFadden):

``` r
# Create a formula for linear model
formula_num <- as.formula(paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + ")))

# Fit the linear model
model_init <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
summary(model_init)
```

    ## 
    ## Call:
    ## glm(formula = formula_num, family = binomial, data = numerical_df)
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)               -38.331738  58.805146  -0.652    0.515    
    ## career_start_ay             0.019699   0.028977   0.680    0.497    
    ## stud_admission_score        0.008001   0.008964   0.893    0.372    
    ## stud_career_admission_age   0.195287   0.132928   1.469    0.142    
    ## exa_cfu_pass               -0.120868   0.012761  -9.472   <2e-16 ***
    ## exa_grade_average          -0.185883   0.022148  -8.393   <2e-16 ***
    ## exa_avg_attempts           -0.363742   0.211153  -1.723    0.085 .  
    ## highschool_grade           -0.009218   0.008189  -1.126    0.260    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2383.3  on 2317  degrees of freedom
    ## Residual deviance: 1071.2  on 2310  degrees of freedom
    ## AIC: 1087.2
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
pseudo_r2 <- pR2(model_init)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.5505247

Iniziamo da un modello con uno pseudo adjusted R^2 relativamente buono,
ma notiamo che il modello presenta molte covariate non significative,
pertanto operiamo una ricerca backward per eliminare quelle non
rilevanti.

``` r
covariate = paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + "))

#Covariate rimosse durante la semplificazione, in ordine

formula_num <- as.formula(covariate)

# Fit the linear model
model_back <- glm(formula_num, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ career_start_ay + stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade
    ##                           Df Deviance    AIC     LRT Pr(>Chi)    
    ## <none>                         1071.2 1087.2                     
    ## career_start_ay            1   1071.7 1085.7   0.464  0.49573    
    ## stud_admission_score       1   1072.0 1086.0   0.797  0.37190    
    ## stud_career_admission_age  1   1073.4 1087.4   2.196  0.13835    
    ## exa_cfu_pass               1   1169.5 1183.5  98.313  < 2e-16 ***
    ## exa_grade_average          1   1177.6 1191.6 106.414  < 2e-16 ***
    ## exa_avg_attempts           1   1074.3 1088.3   3.094  0.07859 .  
    ## highschool_grade           1   1072.5 1086.5   1.257  0.26216    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#--------------------------------
model_back = update(model_back, . ~ . - career_start_ay)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade
    ##                           Df Deviance    AIC     LRT Pr(>Chi)    
    ## <none>                         1071.7 1085.7                     
    ## stud_admission_score       1   1072.2 1084.2   0.534  0.46476    
    ## stud_career_admission_age  1   1073.7 1085.7   2.052  0.15197    
    ## exa_cfu_pass               1   1169.7 1181.7  98.013  < 2e-16 ***
    ## exa_grade_average          1   1182.0 1194.0 110.302  < 2e-16 ***
    ## exa_avg_attempts           1   1074.9 1086.9   3.176  0.07473 .  
    ## highschool_grade           1   1072.8 1084.8   1.111  0.29190    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-----------------------------
model_back = update(model_back, . ~ . - stud_admission_score)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ stud_career_admission_age + exa_cfu_pass + exa_grade_average + 
    ##     exa_avg_attempts + highschool_grade
    ##                           Df Deviance    AIC     LRT Pr(>Chi)    
    ## <none>                         1072.2 1084.2                     
    ## stud_career_admission_age  1   1074.3 1084.3   2.081  0.14912    
    ## exa_cfu_pass               1   1170.0 1180.0  97.811  < 2e-16 ***
    ## exa_grade_average          1   1182.8 1192.8 110.526  < 2e-16 ***
    ## exa_avg_attempts           1   1075.5 1085.5   3.309  0.06891 .  
    ## highschool_grade           1   1073.3 1083.3   1.098  0.29481    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-------------------------------
model_back = update(model_back, . ~ . - highschool_grade)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ stud_career_admission_age + exa_cfu_pass + exa_grade_average + 
    ##     exa_avg_attempts
    ##                           Df Deviance    AIC     LRT Pr(>Chi)    
    ## <none>                         1073.3 1083.3                     
    ## stud_career_admission_age  1   1075.7 1083.7   2.338  0.12622    
    ## exa_cfu_pass               1   1179.7 1187.7 106.409  < 2e-16 ***
    ## exa_grade_average          1   1187.5 1195.5 114.200  < 2e-16 ***
    ## exa_avg_attempts           1   1076.4 1084.4   3.046  0.08092 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-------------------------------
model_back = update(model_back, . ~ . - stud_career_admission_age)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + exa_avg_attempts
    ##                   Df Deviance    AIC    LRT Pr(>Chi)    
    ## <none>                 1075.7 1083.7                    
    ## exa_cfu_pass       1   1183.4 1189.4 107.76  < 2e-16 ***
    ## exa_grade_average  1   1190.2 1196.2 114.56  < 2e-16 ***
    ## exa_avg_attempts   1   1078.8 1084.8   3.18  0.07452 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-------------------------------
model_back = update(model_back, . ~ . - exa_avg_attempts)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average
    ##                   Df Deviance    AIC    LRT  Pr(>Chi)    
    ## <none>                 1078.8 1084.8                     
    ## exa_cfu_pass       1   1188.2 1192.2 109.33 < 2.2e-16 ***
    ## exa_grade_average  1   1239.9 1243.9 161.03 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-------------------------------
```

Il modello risultante presenta due covariate numeriche significative: la
media esami relativa al primo semestre e i cfu sostenuti.

## Analisi dei Punti Influenti

Valutiamo l’impatto sul modello di eventuali punti leva. Partiamo dai
punti leva:

``` r
model_1=model_back
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
```

![](regres_log_con_commenti_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
sum( lev [ lev >  2 * p / n ] ) # 1.36234/5= 0.272468 i leverages pesano quasi il 30 percento
```

    ## [1] 1.183264

``` r
colors = rep( 'black', nrow( numerical_df ) )
colors = rep( 'black', nrow( numerical_df ) )
colors[ watchout_ids_lev ] = c('red', 'blue', 'green', 'orange')
```

    ## Warning in colors[watchout_ids_lev] = c("red", "blue", "green", "orange"): il
    ## numero di elementi da sostituire non è un multiplo della lunghezza di
    ## sostituzione

``` r
pairs( numerical_df[ , c( 'dropout', 'exa_cfu_pass', 'exa_grade_average' ) ], 
       pch = 16, col = colors, cex = 1 + 0.5 * as.numeric( colors != 'black' ))
```

![](regres_log_con_commenti_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

``` r
#View(numerical_df)
```

Dal grafico dei punti leva, notiamo che esistono vari pattern degli
studenti in base alle possibili combinazioni di cfu sostenuti al primo
semestre (0, 7, 10, 17, 27).

``` r
mod=glm("dropout~1+exa_cfu_pass+exa_grade_average",data = numerical_df, family=binomial)
summary(mod)
```

    ## 
    ## Call:
    ## glm(formula = "dropout~1+exa_cfu_pass+exa_grade_average", family = binomial, 
    ##     data = numerical_df)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)        4.49581    0.41101  10.939   <2e-16 ***
    ## exa_cfu_pass      -0.12281    0.01233  -9.963   <2e-16 ***
    ## exa_grade_average -0.19116    0.02018  -9.475   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2383.3  on 2317  degrees of freedom
    ## Residual deviance: 1078.8  on 2315  degrees of freedom
    ## AIC: 1084.8
    ## 
    ## Number of Fisher Scoring iterations: 6

## Introduzione Interazioni fra Variabili Numeriche

Prima di procedere con l’analisi delle variabili categoriche cerchiamo
di comprendere se le aggiunte di interazioni fra variabili numeriche ci
permettono di migliorare il nostro modello. Osservando la matrice delle
covariate costruita all’inizio si rileva che `exa_cfu_pass` e
`exa_grade_average` sono molto correlate pertanto aggiungo il termine di
interazione al modello e successivamente valuto l’equivalenza o meno dei
due modelli tramite il test ANOVA (Chiquadro):

``` r
mod_int = update(mod, . ~ . + exa_cfu_pass*exa_grade_average)

summary(mod_int)
```

    ## 
    ## Call:
    ## glm(formula = dropout ~ exa_cfu_pass + exa_grade_average + exa_cfu_pass:exa_grade_average, 
    ##     family = binomial, data = numerical_df)
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                     4.966119   0.635959   7.809 5.77e-15 ***
    ## exa_cfu_pass                   -0.183010   0.052258  -3.502 0.000462 ***
    ## exa_grade_average              -0.211689   0.029279  -7.230 4.83e-13 ***
    ## exa_cfu_pass:exa_grade_average  0.002574   0.002156   1.194 0.232561    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2383.3  on 2317  degrees of freedom
    ## Residual deviance: 1077.3  on 2314  degrees of freedom
    ## AIC: 1085.3
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
anova(mod, mod_int, test="Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: dropout ~ 1 + exa_cfu_pass + exa_grade_average
    ## Model 2: dropout ~ exa_cfu_pass + exa_grade_average + exa_cfu_pass:exa_grade_average
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)
    ## 1      2315     1078.8                     
    ## 2      2314     1077.3  1    1.494   0.2216

Il p-value della covariata relativa al termine di interazione è alto
pertanto non è significativa, inoltra il p-value alto del test di
confronto indica che i due modelli sono equivalenti, pertanto seleziono
quello più parsimonioso.

## Introduzione delle Variabili Categoriche

Rendiamo tutte le variabili del Dataset di tipo `factor` affinchè siano
utilizzabili nella regressione logistica.

``` r
MODEL = mod

filtered_df <- df %>% filter(stud_career_status != 'A')
filtered_df_no_na = na.omit(filtered_df)

#Partendo dal modello di ottimo trovato prima costruisco la matrice solo con quelle covariate:
cat_df <- filtered_df_no_na

cat_no_lev_df = cat_df[which(!((cat_df$career_time_conv > 1000 & cat_df$exa_cfu_pass==0)| cat_df$career_time_conv<0 | cat_df$career_time_conv>1600)),]
#cat_no_lev_df=cat_df[which(!(cat_df$career_time_conv<0)),]
cat_no_lev_df$career_time_conv <- NULL

cat_no_lev_df$stud_gender = factor(cat_no_lev_df$stud_gender, ordered=F)
cat_no_lev_df$previousStudies = factor(cat_no_lev_df$previousStudies, ordered=F)
cat_no_lev_df$origins = factor(cat_no_lev_df$origins, ordered=F)
cat_no_lev_df$income_bracket_normalized_on4 = factor(cat_no_lev_df$income_bracket_normalized_on4, ordered=F)


cat_no_lev_df$dropped_on_180<-NULL
cat_no_lev_df$stud_career_status <-NULL


#View(cat_no_lev_df)
```

Aggiorno il modello considerando le variabili categoriche.

``` r
model_cat = glm("dropout ~ 1 + 
    exa_cfu_pass + exa_grade_average  +stud_gender + previousStudies + origins + income_bracket_normalized_on4", data=cat_no_lev_df, family=binomial)

drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ 1 + exa_cfu_pass + exa_grade_average + stud_gender + 
    ##     previousStudies + origins + income_bracket_normalized_on4
    ##                               Df Deviance     AIC     LRT  Pr(>Chi)    
    ## <none>                             858.36  884.36                      
    ## exa_cfu_pass                   1   982.50 1006.50 124.139 < 2.2e-16 ***
    ## exa_grade_average              1   988.36 1012.36 130.003 < 2.2e-16 ***
    ## stud_gender                    1   862.56  886.56   4.199  0.040458 *  
    ## previousStudies                3   869.91  889.91  11.552  0.009089 ** 
    ## origins                        3   860.32  880.32   1.961  0.580636    
    ## income_bracket_normalized_on4  3   865.36  885.36   7.002  0.071841 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_cat = update(model_cat,  . ~ . - origins)
drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + stud_gender + previousStudies + 
    ##     income_bracket_normalized_on4
    ##                               Df Deviance     AIC     LRT  Pr(>Chi)    
    ## <none>                             860.32  880.32                      
    ## exa_cfu_pass                   1   983.33 1001.33 123.010 < 2.2e-16 ***
    ## exa_grade_average              1   991.27 1009.27 130.949 < 2.2e-16 ***
    ## stud_gender                    1   864.69  882.69   4.373  0.036512 *  
    ## previousStudies                3   872.05  886.05  11.729  0.008371 ** 
    ## income_bracket_normalized_on4  3   867.47  881.47   7.147  0.067347 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_cat = update(model_cat,  . ~ . - income_bracket_normalized_on4)
drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + stud_gender + previousStudies
    ##                   Df Deviance     AIC     LRT Pr(>Chi)    
    ## <none>                 867.47  881.47                     
    ## exa_cfu_pass       1   991.06 1003.06 123.588  < 2e-16 ***
    ## exa_grade_average  1   996.66 1008.66 129.188  < 2e-16 ***
    ## stud_gender        1   871.74  883.74   4.267  0.03886 *  
    ## previousStudies    3   878.42  886.42  10.951  0.01199 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
summary(model_cat)
```

    ## 
    ## Call:
    ## glm(formula = dropout ~ exa_cfu_pass + exa_grade_average + stud_gender + 
    ##     previousStudies, family = binomial, data = cat_no_lev_df)
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                 4.67340    0.58874   7.938 2.06e-15 ***
    ## exa_cfu_pass               -0.14436    0.01370 -10.537  < 2e-16 ***
    ## exa_grade_average          -0.19981    0.02378  -8.404  < 2e-16 ***
    ## stud_genderM                0.38820    0.19001   2.043   0.0411 *  
    ## previousStudiesOthers       1.19143    0.59502   2.002   0.0453 *  
    ## previousStudiesScientifica  0.14426    0.35236   0.409   0.6822    
    ## previousStudiesTecnica      1.19671    0.50746   2.358   0.0184 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2211.89  on 2107  degrees of freedom
    ## Residual deviance:  867.47  on 2101  degrees of freedom
    ## AIC: 881.47
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
pseudo_r2 <- pR2(model_cat)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6078153

Abbiamo trovato che il modello ha bisogno di covariate categoriche,
ovvero stud_gender e previousStudies. Le origini dello studente e la sua
fascia di reddito non sono significative.

## Confusion Matrix

Eseguiamo ora la classificazione pertanto computiamo la confusion
matrix. Splittiamo il dataset in due parti: il training set che
comprende l’80% dei dati, il test set il restante 20%. Rifittiamo il
modello sul training set e facciamo predizione sul test set. Costruiamo
la curva ROC e valutiamo la soglia ottimale.

``` r
# Assume you have a dataframe data with predictors and a response variable
# split the data into train and test
set.seed(123)
trainIndex <- createDataPartition(cat_no_lev_df$dropout, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

train <- cat_no_lev_df[ trainIndex,]
test  <- cat_no_lev_df[-trainIndex,]

# Fit the logistic regression model
fit <- glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average + previousStudies + stud_gender", data = train, family = binomial)

# Make predictions on the test set
predicted_probs <- predict(fit, newdata = test, type = "response")
#

# Compute the confusion matrix
#predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)
#cm <- confusionMatrix(predicted_classes, test$response)
#print(cm)

# Compute the ROC curve
roc_obj <- roc(test$dropout, predicted_probs)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

``` r
# Plot the ROC curve
plot(roc_obj, print.auc = TRUE)
```

![](regres_log_con_commenti_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
# Compute the AUC
auc <- auc(roc_obj)
print(auc)
```

    ## Area under the curve: 0.9613

``` r
#trovare soglia opt
best=coords(roc_obj, "best", ret="threshold", best.method="youden")
print(best)
```

    ##   threshold
    ## 1 0.1913375

``` r
#
test$dropout=as.factor(test$dropout)

# Compute the confusion matrix
predicted_classes <- ifelse(predicted_probs > as.numeric(best), 1, 0)
predicted_classes=as.factor(predicted_classes)

cm <- confusionMatrix(predicted_classes, test$dropout)
print(cm)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 278   9
    ##          1  35  99
    ##                                          
    ##                Accuracy : 0.8955         
    ##                  95% CI : (0.8622, 0.923)
    ##     No Information Rate : 0.7435         
    ##     P-Value [Acc > NIR] : 4.749e-15      
    ##                                          
    ##                   Kappa : 0.746          
    ##                                          
    ##  Mcnemar's Test P-Value : 0.000164       
    ##                                          
    ##             Sensitivity : 0.8882         
    ##             Specificity : 0.9167         
    ##          Pos Pred Value : 0.9686         
    ##          Neg Pred Value : 0.7388         
    ##              Prevalence : 0.7435         
    ##          Detection Rate : 0.6603         
    ##    Detection Prevalence : 0.6817         
    ##       Balanced Accuracy : 0.9024         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

L’indice AUC, la sensitività, la specificità e l’accuratezza sono tutti
valori soddisfacenti.

## Predizione sugli studenti in corso:

Costruiamo il dataset degli studenti attivi, quindi quelli ancora in
corso:

``` r
attivi_df <- df %>% filter(stud_career_status == 'A')
attivi_df$dropout <- NULL

attivi_df$stud_career_end_date <- NULL
attivi_df = na.omit(attivi_df)
#attivi_df = attivi_df[which(!((attivi_df$career_time_conv > 1000 & attivi_df$exa_cfu_pass==0) | attivi_df$career_start_ay!=2023 | attivi_df$career_time_conv<177)),]
attivi_df = attivi_df[which((attivi_df$career_start_ay==2023)),]
attivi_df$stud_gender = factor(attivi_df$stud_gender, ordered=F)
attivi_df$previousStudies = factor(attivi_df$previousStudies, ordered=F)

#View(attivi_df)
```

Il modello ottimale è il seguente:

``` r
model_opt = glm("dropout ~ 1 + exa_cfu_pass + exa_grade_average + previousStudies + stud_gender", data = train, family=binomial)
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = "dropout ~ 1 + exa_cfu_pass + exa_grade_average + previousStudies + stud_gender", 
    ##     family = binomial, data = train)
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                 4.52526    0.62839   7.201 5.96e-13 ***
    ## exa_cfu_pass               -0.14321    0.01528  -9.376  < 2e-16 ***
    ## exa_grade_average          -0.18971    0.02559  -7.414 1.23e-13 ***
    ## previousStudiesOthers       1.02682    0.65008   1.580   0.1142    
    ## previousStudiesScientifica  0.02873    0.37434   0.077   0.9388    
    ## previousStudiesTecnica      1.16421    0.56529   2.059   0.0394 *  
    ## stud_genderM                0.41294    0.21170   1.951   0.0511 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1728.06  on 1686  degrees of freedom
    ## Residual deviance:  702.76  on 1680  degrees of freedom
    ## AIC: 716.76
    ## 
    ## Number of Fisher Scoring iterations: 7

Eseguiamo la predizione tramite il nostro modello per gli attivi:

``` r
predizione <- predict(model_opt, newdata = attivi_df, type = "response")

binary_output <- ifelse(predizione > as.numeric(best), 1, 0)

attivi_df$dropout_prediction = binary_output

sum(binary_output)/length(binary_output)
```

    ## [1] 0.3587302

``` r
#View(attivi_df)
```

La probabilità di dropout stimata è 0.36 che sovrastima quella rilevata
sugli studenti a carriera conclusa. Questo, in realtà, è ragionevole
considerando che l’obiettivo è prevenire il dropout, quindi una
sovrastima è accettabile.
