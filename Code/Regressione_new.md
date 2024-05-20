Regressione_new
================
Alessandro Wiget
2024-05-20

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
library(glmtoolbox)
```

## Il Dataset

Prima di tutto definiamo la working directory:

IMPORTANTE! Cambiare la directoy a seconda del pc.

Importiamo il Dataset, presente nella cartella `Dati/`:

``` r
setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto/Code")
df <- read_excel("../Dati/Dropout20240226_IngMate.xlsx")
#View(df)
```

## Regressione Logistica

Consideriamo innanzitutto solo gli studenti con carriere terminate, cioè
o che si sono laureati o che hanno abbandonato il corso di studio:

``` r
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

``` r
numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
numerical_df = na.omit(numerical_df)
```

Osserviamo se esistono correlazioni significative fra i dati numerici:

``` r
X = numerical_df[, -4]
corrplot(cor(X), method='color')
```

![](Regressione_new_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

## La Prima Regressione Logistica

Effettuiamo la regressione logistica fra le variabili numeriche del
dataset, e vediamo quanto vale inizialmente l’adjustedR2:

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
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.8683  -0.2974  -0.2021  -0.0920   4.7038  
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                1.098e+02  6.432e+01   1.706  0.08793 .  
    ## career_start_ay           -5.354e-02  3.164e-02  -1.692  0.09063 .  
    ## stud_admission_score       1.200e-02  9.971e-03   1.204  0.22866    
    ## stud_career_admission_age  3.882e-01  1.426e-01   2.723  0.00647 ** 
    ## exa_cfu_pass              -1.362e-01  1.506e-02  -9.045  < 2e-16 ***
    ## exa_grade_average         -5.672e-02  1.769e-02  -3.206  0.00135 ** 
    ## exa_avg_attempts           3.800e-01  2.377e-01   1.598  0.10997    
    ## highschool_grade          -4.188e-02  9.064e-03  -4.621 3.82e-06 ***
    ## career_time_conv          -4.514e-03  2.664e-04 -16.944  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  910.71  on 2454  degrees of freedom
    ## AIC: 928.71
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
pseudo_r2 <- pR2(model_init)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6477962

Iniziamo da un valore di adjustedR2 di 0.656, quindi già buono, vediamo
adesso di trovare un buon modello logistico, che dunque minimizzi l’AIC.

Restringiamoci al miglior modello per ogni numero di variabli
(l’intercetta conta come variabile extra), e mostriamo anche i
rispettivi adjustedR2:

Non notiamo un peggioramento troppo elevato né dell’AIC né dell’adjR2
utilizzando il modello a 5 covariate, quindi prendiamo in considerazione
quest’ultimo.

Effettuiamo una ricerca backward dal modello finale per cercare di
trovare un modello simile ma con un processo più logico.

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
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         910.71  928.71                     
    ## career_start_ay            1   913.56  929.56   2.85  0.091236 .  
    ## stud_admission_score       1   912.16  928.16   1.46  0.227631    
    ## stud_career_admission_age  1   918.47  934.47   7.77  0.005322 ** 
    ## exa_cfu_pass               1   999.25 1015.25  88.54 < 2.2e-16 ***
    ## exa_grade_average          1   921.17  937.17  10.46  0.001218 ** 
    ## exa_avg_attempts           1   913.18  929.18   2.47  0.115855    
    ## highschool_grade           1   932.20  948.20  21.49 3.557e-06 ***
    ## career_time_conv           1  1515.24 1531.24 604.53 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-----------------------------
covariate = paste(covariate, "- stud_admission_score")

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
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         912.16  928.16                     
    ## career_start_ay            1   916.80  930.80   4.63  0.031370 *  
    ## stud_career_admission_age  1   920.08  934.08   7.92  0.004895 ** 
    ## exa_cfu_pass               1   999.41 1013.41  87.25 < 2.2e-16 ***
    ## exa_grade_average          1   921.81  935.81   9.65  0.001893 ** 
    ## exa_avg_attempts           1   914.23  928.23   2.06  0.150897    
    ## highschool_grade           1   932.97  946.97  20.81 5.073e-06 ***
    ## career_time_conv           1  1516.77 1530.77 604.61 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-------------------------------
covariate = paste(covariate, "- exa_avg_attempts")

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
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score - exa_avg_attempts
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         914.23  928.23                     
    ## career_start_ay            1   918.62  930.62   4.39  0.036179 *  
    ## stud_career_admission_age  1   921.88  933.88   7.65  0.005666 ** 
    ## exa_cfu_pass               1  1000.64 1012.64  86.41 < 2.2e-16 ***
    ## exa_grade_average          1   922.17  934.17   7.94  0.004829 ** 
    ## highschool_grade           1   936.47  948.47  22.24   2.4e-06 ***
    ## career_time_conv           1  1519.16 1531.16 604.94 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#-----------------------------------------
covariate = paste(covariate, "- career_start_ay")

formula_1 <- as.formula(covariate)

# Fit the linear model
model_1 <- glm(formula_1, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ career_start_ay + stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score - exa_avg_attempts
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         914.23  928.23                     
    ## career_start_ay            1   918.62  930.62   4.39  0.036179 *  
    ## stud_career_admission_age  1   921.88  933.88   7.65  0.005666 ** 
    ## exa_cfu_pass               1  1000.64 1012.64  86.41 < 2.2e-16 ***
    ## exa_grade_average          1   922.17  934.17   7.94  0.004829 ** 
    ## highschool_grade           1   936.47  948.47  22.24   2.4e-06 ***
    ## career_time_conv           1  1519.16 1531.16 604.94 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
#------------------------------------------
covariate = paste(covariate, "- stud_career_admission_age")

formula_2 <- as.formula(covariate)

# Fit the linear model
model_2 <- glm(formula_2, data = numerical_df, family=binomial)

# Print the summary of the model
drop1(model_back, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ career_start_ay + stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score - exa_avg_attempts
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         914.23  928.23                     
    ## career_start_ay            1   918.62  930.62   4.39  0.036179 *  
    ## stud_career_admission_age  1   921.88  933.88   7.65  0.005666 ** 
    ## exa_cfu_pass               1  1000.64 1012.64  86.41 < 2.2e-16 ***
    ## exa_grade_average          1   922.17  934.17   7.94  0.004829 ** 
    ## highschool_grade           1   936.47  948.47  22.24   2.4e-06 ***
    ## career_time_conv           1  1519.16 1531.16 604.94 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
anova(model_1, model_2, test="Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: dropout ~ career_start_ay + stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score - exa_avg_attempts - 
    ##     career_start_ay
    ## Model 2: dropout ~ career_start_ay + stud_admission_score + stud_career_admission_age + 
    ##     exa_cfu_pass + exa_grade_average + exa_avg_attempts + highschool_grade + 
    ##     career_time_conv - stud_admission_score - exa_avg_attempts - 
    ##     career_start_ay - stud_career_admission_age
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)   
    ## 1      2457     918.62                        
    ## 2      2458     927.77 -1  -9.1584 0.002476 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Aggiungere che teniamo il modello con 5 covariate, rifiutando il modello
con 4 per via del tenst sulle devianze.

## Analisi dei Punti Influenti

Valutiamo l’impatto sul modello di eventuali punti leva e outliers.
Partiamo dai punti leva:

``` r
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

![](Regressione_new_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
sum( lev [ lev >  2 * p / n ] ) # 1.36234/5= 0.272468 i leverages pesano quasi il 30 percento
```

    ## [1] 1.399353

``` r
no_lev_df = numerical_df[which(!((numerical_df$career_time_conv > 1000 & numerical_df$exa_cfu_pass==0)| numerical_df$career_time_conv<0)),]

model_no_lev = glm("dropout ~ stud_career_admission_age + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv  ", data=no_lev_df, family=binomial)
summary(model_no_lev)
```

    ## 
    ## Call:
    ## glm(formula = "dropout ~ stud_career_admission_age + \n    exa_cfu_pass + exa_grade_average + highschool_grade + \n    career_time_conv  ", 
    ##     family = binomial, data = no_lev_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.5814  -0.2776  -0.1834  -0.0874   4.7369  
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                5.6428403  3.4665999   1.628   0.1036    
    ## stud_career_admission_age  0.3478414  0.1787461   1.946   0.0517 .  
    ## exa_cfu_pass              -0.1345820  0.0150594  -8.937  < 2e-16 ***
    ## exa_grade_average         -0.1194297  0.0302602  -3.947 7.92e-05 ***
    ## highschool_grade          -0.0427510  0.0104675  -4.084 4.42e-05 ***
    ## career_time_conv          -0.0046433  0.0003238 -14.341  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2383.29  on 2317  degrees of freedom
    ## Residual deviance:  721.67  on 2312  degrees of freedom
    ## AIC: 733.67
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
drop1(model_no_lev, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ stud_career_admission_age + exa_cfu_pass + exa_grade_average + 
    ##     highschool_grade + career_time_conv
    ##                           Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                         721.67  733.67                     
    ## stud_career_admission_age  1   725.52  735.52   3.86   0.04959 *  
    ## exa_cfu_pass               1   808.88  818.88  87.21 < 2.2e-16 ***
    ## exa_grade_average          1   741.12  751.12  19.46 1.030e-05 ***
    ## highschool_grade           1   738.28  748.28  16.61 4.595e-05 ***
    ## career_time_conv           1  1075.54 1085.54 353.87 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_no_lev_opt = update(model_no_lev, . ~ . - stud_career_admission_age)

anova(model_no_lev, model_no_lev_opt, test="Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: dropout ~ stud_career_admission_age + exa_cfu_pass + exa_grade_average + 
    ##     highschool_grade + career_time_conv
    ## Model 2: dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)  
    ## 1      2312     721.67                       
    ## 2      2313     725.52 -1  -3.8551  0.04959 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
summary(model_no_lev_opt)
```

    ## 
    ## Call:
    ## glm(formula = dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv, family = binomial, data = no_lev_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.5134  -0.2774  -0.1832  -0.0910   4.6769  
    ## 
    ## Coefficients:
    ##                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)       12.0884882  1.1289173  10.708  < 2e-16 ***
    ## exa_cfu_pass      -0.1350953  0.0150428  -8.981  < 2e-16 ***
    ## exa_grade_average -0.1202473  0.0301641  -3.986 6.71e-05 ***
    ## highschool_grade  -0.0429127  0.0104711  -4.098 4.16e-05 ***
    ## career_time_conv  -0.0046047  0.0003201 -14.385  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2383.29  on 2317  degrees of freedom
    ## Residual deviance:  725.52  on 2313  degrees of freedom
    ## AIC: 735.52
    ## 
    ## Number of Fisher Scoring iterations: 7

## Introduzione Interazioni fra Variabili Numeriche

Prima di procedere con l’aggiunta di variabili categoriche cerchiamo di
comprendere se le aggiunte di interazioni fra variabili numeriche ci
permettono di migliorare il nostro modello. Osservando la matrice delle
covariate costruita all’inizio possiamo osservare che `career_start_ay`
e `stud_career_end_ay` sono estremamente correlate, aggiungiamo quindi
al modello: `career_start_ay*stud_career_end_ay`. Seguendo lo stesso
ragionamento un’altra coppia di covariate che appaiono essere molto
correlate sono `exa_cfu_pass` e `exa_grade_average`, introduciamo la
loro interazione `exa_cfu_pass*exa_grade_average`:

``` r
model_int_grade = glm("dropout ~ 1 + stud_career_admission_age + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv + exa_cfu_pass*exa_grade_average", data=no_lev_df, family=binomial)

gvif(model_1)
```

    ##                             GVIF df GVIF^(1/(2*df))
    ## stud_career_admission_age 1.0340  1          1.0169
    ## exa_cfu_pass              2.4548  1          1.5668
    ## exa_grade_average         2.4003  1          1.5493
    ## highschool_grade          1.2878  1          1.1348
    ## career_time_conv          1.2405  1          1.1138

``` r
gvif(model_int_grade)
```

    ##                                   GVIF df GVIF^(1/(2*df))
    ## stud_career_admission_age       1.0135  1          1.0067
    ## exa_cfu_pass                   24.4428  1          4.9440
    ## exa_grade_average               2.7528  1          1.6592
    ## highschool_grade                1.1922  1          1.0919
    ## career_time_conv                1.1260  1          1.0611
    ## exa_cfu_pass:exa_grade_average 30.1991  1          5.4954

``` r
anova(model_no_lev_opt, model_int_grade, test="Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv
    ## Model 2: dropout ~ 1 + stud_career_admission_age + exa_cfu_pass + exa_grade_average + 
    ##     highschool_grade + career_time_conv + exa_cfu_pass * exa_grade_average
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)
    ## 1      2313     725.52                     
    ## 2      2311     721.63  2   3.8987   0.1424

Non accetto il nuovo modello con l’interazione poichè il GVIF è troppo
elevato.

## Introduzione delle Variabili Categoriche

Rendiamo tutte le variabili del Dataset di tipo `factor` affinchè siano
utilizzabili nella regressione logistica.

``` r
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

``` r
male = cat_no_lev_df[which(cat_no_lev_df$stud_gender=="M"),]
male_mean = mean(male$dropout)
male_mean
```

    ## [1] 0.207483

``` r
female = cat_no_lev_df[which(cat_no_lev_df$stud_gender=="F"),]
female_mean = mean(female$dropout)
female_mean
```

    ## [1] 0.2146226

``` r
# Facciamo uno z-test con:
#                       H0: pM == pF
#                       H1: pM != pF

p_hat = (sum(female$dropout) + sum(male$dropout))/(nrow(cat_no_lev_df))

z_test = (male_mean - female_mean)/sqrt(p_hat*(1-p_hat)*(1/nrow(male) + 1/nrow(female)))

p_value = 2*pnorm(z_test)
```

Non posso rifiutare H0. Non noto una differenza di probabilità di
dropout fra maschi e femmine. Non aggiungiamo questa categoria nel
modello finale.

``` r
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$previousStudies, data=cat_no_lev_df)

summary(model_anova)
```

    ##                                 Df Sum Sq Mean Sq F value   Pr(>F)    
    ## cat_no_lev_df$previousStudies    3    5.0  1.6517   10.06 1.38e-06 ***
    ## Residuals                     2314  379.7  0.1641                     
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$origins, data=cat_no_lev_df)

summary(model_anova)
```

    ##                         Df Sum Sq Mean Sq F value   Pr(>F)    
    ## cat_no_lev_df$origins    3    3.2  1.0774   6.536 0.000213 ***
    ## Residuals             2314  381.5  0.1648                     
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_anova = aov(cat_no_lev_df$dropout ~ cat_no_lev_df$income_bracket_normalized_on4, data=cat_no_lev_df)

summary(model_anova)
```

    ##                                               Df Sum Sq Mean Sq F value
    ## cat_no_lev_df$income_bracket_normalized_on4    3    4.4  1.4752   8.977
    ## Residuals                                   2314  380.3  0.1643        
    ##                                               Pr(>F)    
    ## cat_no_lev_df$income_bracket_normalized_on4 6.54e-06 ***
    ## Residuals                                               
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Aggiungere dei boxplot

``` r
model_cat = glm("dropout ~ 1 + 
    exa_cfu_pass + exa_grade_average + highschool_grade + 
    career_time_conv + previousStudies + origins + income_bracket_normalized_on4", data=cat_no_lev_df, family=binomial)


drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ 1 + exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv + previousStudies + origins + income_bracket_normalized_on4
    ##                               Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                             712.66  740.66                     
    ## exa_cfu_pass                   1   792.95  818.95  80.29 < 2.2e-16 ***
    ## exa_grade_average              1   735.65  761.65  22.99 1.627e-06 ***
    ## highschool_grade               1   729.74  755.74  17.08 3.588e-05 ***
    ## career_time_conv               1  1062.19 1088.19 349.52 < 2.2e-16 ***
    ## previousStudies                3   718.53  740.53   5.87    0.1182    
    ## origins                        3   714.79  736.79   2.12    0.5471    
    ## income_bracket_normalized_on4  3   717.69  739.69   5.03    0.1696    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_cat = update(model_cat,  . ~ . - origins)
drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv + previousStudies + income_bracket_normalized_on4
    ##                               Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                             714.79  736.79                     
    ## exa_cfu_pass                   1   796.21  816.21  81.42 < 2.2e-16 ***
    ## exa_grade_average              1   737.69  757.69  22.91 1.699e-06 ***
    ## highschool_grade               1   730.56  750.56  15.77 7.140e-05 ***
    ## career_time_conv               1  1064.22 1084.22 349.43 < 2.2e-16 ***
    ## previousStudies                3   720.95  736.95   6.16    0.1040    
    ## income_bracket_normalized_on4  3   719.95  735.95   5.16    0.1605    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_cat = update(model_cat,  . ~ . - income_bracket_normalized_on4)
drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv + previousStudies
    ##                   Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                 719.95  735.95                     
    ## exa_cfu_pass       1   801.43  815.43  81.49 < 2.2e-16 ***
    ## exa_grade_average  1   741.28  755.28  21.33 3.860e-06 ***
    ## highschool_grade   1   735.97  749.97  16.02 6.252e-05 ***
    ## career_time_conv   1  1071.00 1085.00 351.06 < 2.2e-16 ***
    ## previousStudies    3   725.52  735.52   5.58     0.134    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
model_cat = update(model_cat,  . ~ . - previousStudies)
drop1(model_cat, test="Chisq")
```

    ## Single term deletions
    ## 
    ## Model:
    ## dropout ~ exa_cfu_pass + exa_grade_average + highschool_grade + 
    ##     career_time_conv
    ##                   Df Deviance     AIC    LRT  Pr(>Chi)    
    ## <none>                 725.52  735.52                     
    ## exa_cfu_pass       1   813.60  821.60  88.07 < 2.2e-16 ***
    ## exa_grade_average  1   745.46  753.46  19.93 8.029e-06 ***
    ## highschool_grade   1   742.27  750.27  16.74 4.283e-05 ***
    ## career_time_conv   1  1077.77 1085.77 352.25 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Abbiamo trovato che il modello non ha bisogno di covariate categoriche.

## Confusion Matrix
