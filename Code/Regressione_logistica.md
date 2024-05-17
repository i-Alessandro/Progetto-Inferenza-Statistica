Regressione Logistica
================
Alessandro Wiget
2024-05-17

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
```

## Il Dataset

Prima di tutto definiamo la working directory:

IMPORTANTE! Cambiare la directoy a seconda del pc.

Importiamo il Dataset, presente nella cartella `Dati/`:

``` r
setwd("/home/alessandro/Inferenza Statistica/Progetto/Code")
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

![](Regressione_logistica_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Effettuiamo la regressione logistica fra le variabili numeriche del
dataset:

``` r
# Create a formula for linear model
formula <- as.formula(paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + ")))

# Fit the linear model
model <- glm(formula, data = numerical_df, family=binomial)

# Print the summary of the model
summary(model)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = numerical_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7120  -0.2907  -0.1961  -0.0782   4.5886  
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)               120.995553  65.261589   1.854  0.06374 .  
    ## career_start_ay            -1.713397   0.357433  -4.794 1.64e-06 ***
    ## stud_admission_score        0.011294   0.010093   1.119  0.26314    
    ## stud_career_admission_age   0.400966   0.142329   2.817  0.00484 ** 
    ## exa_cfu_pass               -0.128186   0.015112  -8.482  < 2e-16 ***
    ## exa_grade_average          -0.058420   0.017805  -3.281  0.00103 ** 
    ## exa_avg_attempts            0.333251   0.240698   1.385  0.16620    
    ## stud_career_end_ay          1.654889   0.355049   4.661 3.15e-06 ***
    ## highschool_grade           -0.040491   0.009209  -4.397 1.10e-05 ***
    ## career_time_conv           -0.009152   0.001061  -8.629  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  888.21  on 2453  degrees of freedom
    ## AIC: 908.21
    ## 
    ## Number of Fisher Scoring iterations: 7

Cerchiamo di trovare il miglior modello con un Automatic Selection
Method. Massimizziamo l’adjr2 con la funzione `leaps()`, ovvero un
algoritmo Branch and Bound per trovare tali modelli.

``` r
x = model.matrix( model ) [ , -1 ]
y = numerical_df$dropout

adjr = leaps( x, y, method = "adjr2" )
names(adjr)
```

    ## [1] "which" "label" "size"  "adjr2"

``` r
bestmodel_adjr2_ind = which.max( adjr$adjr2 )
adjr$which[ bestmodel_adjr2_ind, ] 
```

    ##    1    2    3    4    5    6    7    8    9 
    ## TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE

``` r
maxadjr(adjr,50)
```

    ## 1,2,3,4,5,6,7,8,9   1,2,3,4,5,7,8,9     1,3,4,5,7,8,9   1,2,3,4,5,6,8,9 
    ##             0.647             0.647             0.646             0.646 
    ##   1,3,4,5,6,7,8,9     1,2,3,4,5,8,9   2,3,4,5,6,7,8,9     2,3,4,5,7,8,9 
    ##             0.646             0.646             0.646             0.646 
    ##   1,2,4,5,6,7,8,9     1,2,4,5,7,8,9       1,3,4,5,8,9     1,3,4,5,6,8,9 
    ##             0.646             0.646             0.645             0.645 
    ##       3,4,5,7,8,9     3,4,5,6,7,8,9     1,2,4,5,6,8,9       1,4,5,7,8,9 
    ##             0.645             0.645             0.645             0.645 
    ##     2,3,4,5,6,8,9     1,4,5,6,7,8,9       1,2,4,5,8,9     2,4,5,6,7,8,9 
    ##             0.645             0.645             0.645             0.645 
    ##       2,3,4,5,8,9       2,4,5,7,8,9         1,4,5,8,9       1,4,5,6,8,9 
    ##             0.645             0.645             0.644             0.644 
    ##         4,5,7,8,9       4,5,6,7,8,9       2,4,5,6,8,9         2,4,5,8,9 
    ##             0.644             0.644             0.644             0.644 
    ##         3,4,5,8,9       3,4,5,6,8,9           4,5,8,9         4,5,6,8,9 
    ##             0.643             0.643             0.642             0.642 
    ##   1,2,3,4,5,6,7,9   1,2,3,4,6,7,8,9         1,3,4,5,9         3,4,5,7,9 
    ##             0.640             0.640             0.640             0.639 
    ##         1,4,5,7,9         1,4,5,6,9         1,3,4,8,9           1,4,5,9 
    ##             0.638             0.638             0.638             0.638 
    ##           4,5,7,9           3,4,5,9           1,4,8,9           3,4,8,9 
    ##             0.638             0.637             0.637             0.636 
    ##           4,7,8,9           4,6,8,9           2,4,8,9           2,4,5,9 
    ##             0.636             0.636             0.636             0.635 
    ##             4,8,9             4,5,9 
    ##             0.635             0.635

Dal momento che non è presente una variazione significativa dell’adjr2
dei primi modelli, cerchiamo dunque di minimizzare il numero di
variabili utilizzate. La scelta ricade dunque sul modello che contiene
le features delle colonne `4,5,8,9`.

Il modello diventa dunque:

``` r
# Assuming 'df' is your dataframe and 'target' is your target variable
# Select only the columns you're interested in
selected_df <- numerical_df[, c(4,5,6,9,10)]

# Create a formula for the model
# This assumes that the first column is the target variable
formula <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))

# Fit the model
model_opt <- glm(formula, data = selected_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = selected_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7863  -0.2994  -0.2082  -0.0892   4.6452  
    ## 
    ## Coefficients:
    ##                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)        9.9681175  0.8801057  11.326  < 2e-16 ***
    ## exa_cfu_pass      -0.1319171  0.0145659  -9.057  < 2e-16 ***
    ## exa_grade_average -0.0391407  0.0138934  -2.817  0.00484 ** 
    ## highschool_grade  -0.0442867  0.0089058  -4.973  6.6e-07 ***
    ## career_time_conv  -0.0043681  0.0002557 -17.083  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  927.77  on 2458  degrees of freedom
    ## AIC: 937.77
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6411962

Riferendoci alla tabella delle covariate presentata in precedenza,
vogliamo capire se è possibile migliorare il modello introducendo una
interazione fra le covariate che sono più correlate, ovvero
`exa_cfu_pass` e `exa_grade_average`:

``` r
# Assuming 'df' is your dataframe and 'target' is your target variable
# Select only the columns you're interested in
selected_df <- numerical_df[, c(4,5,6,9,10)]

# Create a formula for the model
# This assumes that the first column is the target variable
covariate <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))

interazioni = " + exa_cfu_pass * exa_grade_average"

formula = as.formula(paste(covariate, interazioni))

# Fit the model
model_opt <- glm(formula, data = selected_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = selected_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7286  -0.3071  -0.1846  -0.0848   4.5482  
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                     9.3960796  0.8922343  10.531  < 2e-16 ***
    ## exa_cfu_pass                   -0.0201679  0.0445912  -0.452  0.65106    
    ## exa_grade_average              -0.0312671  0.0140616  -2.224  0.02618 *  
    ## highschool_grade               -0.0402753  0.0089343  -4.508 6.55e-06 ***
    ## career_time_conv               -0.0042731  0.0002544 -16.795  < 2e-16 ***
    ## exa_cfu_pass:exa_grade_average -0.0047988  0.0018546  -2.587  0.00967 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  920.77  on 2457  degrees of freedom
    ## AIC: 932.77
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6439052

Notiamo un miglioramento dell’AIC, tuttavia adesso la feature
`exa_cfu_pass` risulta superflua, procediamo quindi ad eliminarla:

``` r
# Assuming 'df' is your dataframe and 'target' is your target variable
# Select only the columns you're interested in
selected_df <- numerical_df[, c(4,5,6,9,10)]

# Create a formula for the model
# This assumes that the first column is the target variable
covariate <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))

interazioni = " + exa_cfu_pass * exa_grade_average"

rimuovere = "- exa_cfu_pass"

formula = as.formula(paste(covariate, interazioni, rimuovere))

# Fit the model
model_opt <- glm(formula, data = selected_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = selected_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7191  -0.3110  -0.1820  -0.0844   4.5346  
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                     9.3242295  0.8754440  10.651  < 2e-16 ***
    ## exa_grade_average              -0.0312854  0.0140656  -2.224   0.0261 *  
    ## highschool_grade               -0.0398612  0.0088688  -4.495 6.97e-06 ***
    ## career_time_conv               -0.0042589  0.0002518 -16.916  < 2e-16 ***
    ## exa_cfu_pass:exa_grade_average -0.0055960  0.0006029  -9.282  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  920.97  on 2458  degrees of freedom
    ## AIC: 930.97
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6438262

Confrontando i valori dell’adjusted R^2 vediamo che l’aggiunta di una
covariata data dal termine di interazioni, fa aumentare l’indice di
pochi millesimi. Dunque non risulta vantaggioso il tradeoff e manteniamo
il modello precedente.

## Variabili Categoriche:

``` r
filtered_df <- df %>% filter(stud_career_status != 'A')
filtered_df_no_na = na.omit(filtered_df)

new_df <- selected_df

new_df$stud_gender = factor(filtered_df_no_na$stud_gender, ordered = F)
new_df$previousStudies = factor(filtered_df_no_na$previousStudies, ordered = F)
new_df$origins = factor(filtered_df_no_na$origins, ordered = F)
new_df$income_bracket_normalized_on4 = factor(filtered_df_no_na$income_bracket_normalized_on4, ordered = F)
new_df$dropped_on_180 = factor(filtered_df_no_na$dropped_on_180, ordered = F)
# Create a formula for the model
# This assumes that the first column is the target variable
covariate <- paste("dropout ~", paste(names(new_df[,-which(names(new_df) == "dropout")]), collapse = " + "))

formula = as.formula(paste(covariate))

# Fit the model
model_opt <- glm(formula, data = new_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = new_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.0273  -0.2964  -0.2061  -0.0936   4.6758  
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error z value
    ## (Intercept)                                9.131e+00  1.025e+00   8.912
    ## exa_cfu_pass                              -1.324e-01  1.494e-02  -8.865
    ## exa_grade_average                         -4.147e-02  1.425e-02  -2.911
    ## highschool_grade                          -4.022e-02  9.490e-03  -4.238
    ## career_time_conv                          -4.350e-03  2.619e-04 -16.606
    ## stud_genderM                               3.724e-01  1.963e-01   1.897
    ## previousStudiesOthers                      6.465e-01  6.468e-01   1.000
    ## previousStudiesScientifica                 1.913e-01  3.451e-01   0.554
    ## previousStudiesTecnica                     8.726e-01  5.276e-01   1.654
    ## originsForeigner                           5.709e-01  1.116e+00   0.512
    ## originsMilanese                           -2.815e-02  2.096e-01  -0.134
    ## originsOffsite                             4.202e-01  3.779e-01   1.112
    ## income_bracket_normalized_on4fascia bassa  1.476e-01  2.460e-01   0.600
    ## income_bracket_normalized_on4fascia media  1.463e-01  2.250e-01   0.650
    ## income_bracket_normalized_on4LS           -3.531e-01  3.117e-01  -1.133
    ## dropped_on_180Y                            1.311e+01  5.931e+02   0.022
    ##                                           Pr(>|z|)    
    ## (Intercept)                                < 2e-16 ***
    ## exa_cfu_pass                               < 2e-16 ***
    ## exa_grade_average                           0.0036 ** 
    ## highschool_grade                          2.26e-05 ***
    ## career_time_conv                           < 2e-16 ***
    ## stud_genderM                                0.0578 .  
    ## previousStudiesOthers                       0.3175    
    ## previousStudiesScientifica                  0.5793    
    ## previousStudiesTecnica                      0.0982 .  
    ## originsForeigner                            0.6089    
    ## originsMilanese                             0.8932    
    ## originsOffsite                              0.2662    
    ## income_bracket_normalized_on4fascia bassa   0.5486    
    ## income_bracket_normalized_on4fascia media   0.5157    
    ## income_bracket_normalized_on4LS             0.2573    
    ## dropped_on_180Y                             0.9824    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  914.25  on 2447  degrees of freedom
    ## AIC: 946.25
    ## 
    ## Number of Fisher Scoring iterations: 17

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6464283

Cerchiamo di migliorare con una ricerca eliminando covariate:

``` r
covariate <- paste("dropout ~", paste(names(new_df[,-which(names(new_df) == "dropout")]), collapse = " + "))

delete = "- dropped_on_180 - origins -income_bracket_normalized_on4 "

formula = as.formula(paste(covariate, delete))

# Fit the model
model_opt <- glm(formula, data = new_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = new_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.9740  -0.2994  -0.2069  -0.0919   4.6999  
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                 9.1625149  0.9877156   9.276  < 2e-16 ***
    ## exa_cfu_pass               -0.1321471  0.0148563  -8.895  < 2e-16 ***
    ## exa_grade_average          -0.0424613  0.0140933  -3.013  0.00259 ** 
    ## highschool_grade           -0.0398083  0.0092096  -4.322 1.54e-05 ***
    ## career_time_conv           -0.0043574  0.0002562 -17.007  < 2e-16 ***
    ## stud_genderM                0.3723828  0.1955958   1.904  0.05693 .  
    ## previousStudiesOthers       0.8120946  0.5982824   1.357  0.17466    
    ## previousStudiesScientifica  0.2022192  0.3418250   0.592  0.55413    
    ## previousStudiesTecnica      0.8445640  0.5198693   1.625  0.10425    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  919.77  on 2454  degrees of freedom
    ## AIC: 937.77
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6442928

Introduciamo adesso delle interazioni per vedere se possiamo migliorare
il risultato:

``` r
covariate <- paste("dropout ~", paste(names(new_df[,-which(names(new_df) == "dropout")]), collapse = " + "))

delete = "- dropped_on_180 - origins -income_bracket_normalized_on4"

interactions = "+stud_gender*exa_cfu_pass +stud_gender*exa_grade_average +stud_gender*highschool_grade +stud_gender*career_time_conv"

formula = as.formula(paste(covariate, interactions ,delete))

# Fit the model
model_opt <- glm(formula, data = new_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = new_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.9260  -0.3078  -0.2088  -0.0929   5.0107  
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                     9.3677955  1.7515654   5.348 8.88e-08 ***
    ## exa_cfu_pass                   -0.1735430  0.0293061  -5.922 3.19e-09 ***
    ## exa_grade_average              -0.0364796  0.0250941  -1.454   0.1460    
    ## highschool_grade               -0.0293908  0.0172343  -1.705   0.0881 .  
    ## career_time_conv               -0.0050301  0.0004688 -10.730  < 2e-16 ***
    ## stud_genderM                   -0.0063706  1.9994103  -0.003   0.9975    
    ## previousStudiesOthers           0.7488349  0.6192845   1.209   0.2266    
    ## previousStudiesScientifica      0.2191399  0.3435578   0.638   0.5236    
    ## previousStudiesTecnica          0.8749175  0.5113180   1.711   0.0871 .  
    ## exa_cfu_pass:stud_genderM       0.0573740  0.0339210   1.691   0.0908 .  
    ## exa_grade_average:stud_genderM -0.0060690  0.0303603  -0.200   0.8416    
    ## highschool_grade:stud_genderM  -0.0159022  0.0203227  -0.782   0.4339    
    ## career_time_conv:stud_genderM   0.0010186  0.0005620   1.812   0.0699 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  912.44  on 2450  degrees of freedom
    ## AIC: 938.44
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6471267

Faccio adesso un test per vedere il massimo teorico per quanto riguarda
l’adjR2, aggiungendo quasi tutte le variabili possibili e semplificando
con leaps():

``` r
covariate <- paste("dropout ~", paste(names(new_df[,-which(names(new_df) == "dropout")]), collapse = " + "))

interactions = "+income_bracket_normalized_on4*exa_cfu_pass +income_bracket_normalized_on4*exa_grade_average + income_bracket_normalized_on4*highschool_grade + income_bracket_normalized_on4*career_time_conv +income_bracket_normalized_on4*highschool_grade +income_bracket_normalized_on4*career_time_conv +dropped_on_180*exa_grade_average + dropped_on_180*highschool_grade + dropped_on_180*career_time_conv +dropped_on_180*highschool_grade +dropped_on_180*career_time_conv"

# +stud_gender*exa_cfu_pass +stud_gender*exa_grade_average + stud_gender*highschool_grade +stud_gender*highschool_grade +stud_gender*career_time_conv   
# +origins*exa_cfu_pass +origins*exa_grade_average + origins*highschool_grade + origins*career_time_conv +origins*highschool_grade + origins*career_time_conv
# +previousStudies*exa_cfu_pass +previousStudies*exa_grade_average + previousStudies*highschool_grade + previousStudies*career_time_conv +previousStudies*highschool_grade +previousStudies*career_time_conv

formula = as.formula(paste(covariate, interactions))

# Fit the model
model_opt <- glm(formula, data = new_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = new_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.9847  -0.2975  -0.2037  -0.0764   4.4267  
    ## 
    ## Coefficients:
    ##                                                               Estimate
    ## (Intercept)                                                  8.135e+00
    ## exa_cfu_pass                                                -1.511e-01
    ## exa_grade_average                                           -2.785e-02
    ## highschool_grade                                            -2.327e-02
    ## career_time_conv                                            -4.824e-03
    ## stud_genderM                                                 4.015e-01
    ## previousStudiesOthers                                        7.197e-01
    ## previousStudiesScientifica                                   1.947e-01
    ## previousStudiesTecnica                                       9.042e-01
    ## originsForeigner                                             9.945e-01
    ## originsMilanese                                             -2.506e-02
    ## originsOffsite                                               4.432e-01
    ## income_bracket_normalized_on4fascia bassa                    4.288e-01
    ## income_bracket_normalized_on4fascia media                    1.498e+00
    ## income_bracket_normalized_on4LS                              8.055e+00
    ## dropped_on_180Y                                              9.772e+00
    ## exa_cfu_pass:income_bracket_normalized_on4fascia bassa       4.167e-02
    ## exa_cfu_pass:income_bracket_normalized_on4fascia media       2.993e-02
    ## exa_cfu_pass:income_bracket_normalized_on4LS                -3.841e-03
    ## exa_grade_average:income_bracket_normalized_on4fascia bassa -1.836e-02
    ## exa_grade_average:income_bracket_normalized_on4fascia media -1.693e-02
    ## exa_grade_average:income_bracket_normalized_on4LS           -4.708e-02
    ## highschool_grade:income_bracket_normalized_on4fascia bassa  -1.886e-02
    ## highschool_grade:income_bracket_normalized_on4fascia media  -2.899e-02
    ## highschool_grade:income_bracket_normalized_on4LS            -6.392e-02
    ## career_time_conv:income_bracket_normalized_on4fascia bassa   1.019e-03
    ## career_time_conv:income_bracket_normalized_on4fascia media   9.791e-04
    ## career_time_conv:income_bracket_normalized_on4LS            -1.826e-03
    ## exa_grade_average:dropped_on_180Y                            1.022e-01
    ## highschool_grade:dropped_on_180Y                             2.686e-02
    ## career_time_conv:dropped_on_180Y                             3.635e-03
    ##                                                             Std. Error z value
    ## (Intercept)                                                  1.653e+00   4.920
    ## exa_cfu_pass                                                 2.473e-02  -6.109
    ## exa_grade_average                                            2.464e-02  -1.130
    ## highschool_grade                                             1.634e-02  -1.425
    ## career_time_conv                                             4.823e-04 -10.002
    ## stud_genderM                                                 1.991e-01   2.017
    ## previousStudiesOthers                                        6.438e-01   1.118
    ## previousStudiesScientifica                                   3.505e-01   0.555
    ## previousStudiesTecnica                                       5.400e-01   1.675
    ## originsForeigner                                             1.146e+00   0.868
    ## originsMilanese                                              2.119e-01  -0.118
    ## originsOffsite                                               3.849e-01   1.152
    ## income_bracket_normalized_on4fascia bassa                    2.277e+00   0.188
    ## income_bracket_normalized_on4fascia media                    2.245e+00   0.667
    ## income_bracket_normalized_on4LS                              4.421e+00   1.822
    ## dropped_on_180Y                                              6.045e+03   0.002
    ## exa_cfu_pass:income_bracket_normalized_on4fascia bassa       3.987e-02   1.045
    ## exa_cfu_pass:income_bracket_normalized_on4fascia media       3.497e-02   0.856
    ## exa_cfu_pass:income_bracket_normalized_on4LS                 6.793e-02  -0.057
    ## exa_grade_average:income_bracket_normalized_on4fascia bassa  3.738e-02  -0.491
    ## exa_grade_average:income_bracket_normalized_on4fascia media  3.592e-02  -0.471
    ## exa_grade_average:income_bracket_normalized_on4LS            5.764e-02  -0.817
    ## highschool_grade:income_bracket_normalized_on4fascia bassa   2.348e-02  -0.803
    ## highschool_grade:income_bracket_normalized_on4fascia media   2.308e-02  -1.256
    ## highschool_grade:income_bracket_normalized_on4LS             4.045e-02  -1.580
    ## career_time_conv:income_bracket_normalized_on4fascia bassa   6.785e-04   1.501
    ## career_time_conv:income_bracket_normalized_on4fascia media   6.473e-04   1.513
    ## career_time_conv:income_bracket_normalized_on4LS             1.300e-03  -1.405
    ## exa_grade_average:dropped_on_180Y                            3.745e+02   0.000
    ## highschool_grade:dropped_on_180Y                             6.187e+01   0.000
    ## career_time_conv:dropped_on_180Y                             1.386e+01   0.000
    ##                                                             Pr(>|z|)    
    ## (Intercept)                                                 8.64e-07 ***
    ## exa_cfu_pass                                                1.00e-09 ***
    ## exa_grade_average                                             0.2585    
    ## highschool_grade                                              0.1542    
    ## career_time_conv                                             < 2e-16 ***
    ## stud_genderM                                                  0.0437 *  
    ## previousStudiesOthers                                         0.2636    
    ## previousStudiesScientifica                                    0.5786    
    ## previousStudiesTecnica                                        0.0940 .  
    ## originsForeigner                                              0.3855    
    ## originsMilanese                                               0.9059    
    ## originsOffsite                                                0.2495    
    ## income_bracket_normalized_on4fascia bassa                     0.8506    
    ## income_bracket_normalized_on4fascia media                     0.5047    
    ## income_bracket_normalized_on4LS                               0.0685 .  
    ## dropped_on_180Y                                               0.9987    
    ## exa_cfu_pass:income_bracket_normalized_on4fascia bassa        0.2959    
    ## exa_cfu_pass:income_bracket_normalized_on4fascia media        0.3921    
    ## exa_cfu_pass:income_bracket_normalized_on4LS                  0.9549    
    ## exa_grade_average:income_bracket_normalized_on4fascia bassa   0.6233    
    ## exa_grade_average:income_bracket_normalized_on4fascia media   0.6374    
    ## exa_grade_average:income_bracket_normalized_on4LS             0.4141    
    ## highschool_grade:income_bracket_normalized_on4fascia bassa    0.4218    
    ## highschool_grade:income_bracket_normalized_on4fascia media    0.2091    
    ## highschool_grade:income_bracket_normalized_on4LS              0.1141    
    ## career_time_conv:income_bracket_normalized_on4fascia bassa    0.1333    
    ## career_time_conv:income_bracket_normalized_on4fascia media    0.1304    
    ## career_time_conv:income_bracket_normalized_on4LS              0.1599    
    ## exa_grade_average:dropped_on_180Y                             0.9998    
    ## highschool_grade:dropped_on_180Y                              0.9997    
    ## career_time_conv:dropped_on_180Y                              0.9998    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  899.81  on 2432  degrees of freedom
    ## AIC: 961.81
    ## 
    ## Number of Fisher Scoring iterations: 17

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6520112

``` r
x = model.matrix( model_opt ) [ , -1 ]
y = new_df$dropout

detect.lindep(x) #"Suspicious column name(s):   exa_cfu_pass:dropped_on_180Y, exa_grade_average:dropped_on_180Y" (Cancellata la prima)
```

    ## [1] "No linear dependent column(s) detected."

``` r
adjr = leaps( x, y, method = "adjr2" )
names(adjr)
```

    ## [1] "which" "label" "size"  "adjr2"

``` r
bestmodel_adjr2_ind = which.max( adjr$adjr2 )
adjr$which[ bestmodel_adjr2_ind, ] 
```

    ##     1     2     3     4     5     6     7     8     9     A     B     C     D 
    ##  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE  TRUE 
    ##     E     F     G     H     I     J     K     L     M     N     O     P     Q 
    ##  TRUE  TRUE FALSE FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE 
    ##     R     S     T     U 
    ##  TRUE  TRUE FALSE FALSE

``` r
maxadjr(adjr,15)
```

    ##         1,2,3,4,5,6,8,12,13,14,15,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##            1,2,3,4,5,6,8,12,13,14,15,22,23,24,25,26,27 
    ##                                                   0.65 
    ##       1,2,3,4,5,6,8,9,12,13,14,15,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##   1,2,3,4,5,6,8,12,13,14,15,18,21,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##       1,2,3,4,5,6,7,8,12,13,14,15,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##         1,2,3,4,5,6,8,12,13,14,15,21,22,23,25,26,27,28 
    ##                                                   0.65 
    ##      1,2,3,4,5,6,8,12,13,14,15,18,21,22,23,24,25,26,27 
    ##                                                   0.65 
    ##          1,2,3,4,5,6,8,9,12,13,14,15,22,23,24,25,26,27 
    ##                                                   0.65 
    ##      1,2,3,4,5,6,8,12,13,14,15,22,23,24,25,26,27,28,29 
    ##                                                   0.65 
    ## 1,2,3,4,5,6,8,9,12,13,14,15,18,21,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##          1,2,3,4,5,6,7,8,12,13,14,15,22,23,24,25,26,27 
    ##                                                   0.65 
    ##            1,2,3,4,5,6,8,12,13,14,15,21,22,23,25,26,27 
    ##                                                   0.65 
    ##     1,2,3,4,5,6,7,8,9,12,13,14,15,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##      1,2,3,4,5,6,8,12,13,14,15,16,22,23,24,25,26,27,28 
    ##                                                   0.65 
    ##      1,2,3,4,5,6,8,12,13,14,15,21,22,23,25,26,27,28,29 
    ##                                                   0.65

``` r
leap_matrix = data.frame(x[, c(1,2,3,4,5,6,8,12,13,14,15,22,23,24,25,26,27,28)])

leap_matrix$dropout = new_df$dropout

covariate <- paste("dropout ~", paste(names(leap_matrix[,-which(names(leap_matrix) == "dropout")]), collapse = " + "))

formula = as.formula(paste(covariate))

# Fit the model
model_leap <- glm(formula, data = leap_matrix, family = binomial)

# Print the summary of the model
summary(model_leap)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = leap_matrix)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.0506  -0.2933  -0.2089  -0.0918   4.4675  
    ## 
    ## Coefficients:
    ##                                                              Estimate
    ## (Intercept)                                                 8.477e+00
    ## exa_cfu_pass                                               -1.304e-01
    ## exa_grade_average                                          -4.279e-02
    ## highschool_grade                                           -2.633e-02
    ## career_time_conv                                           -4.705e-03
    ## stud_genderM                                                4.080e-01
    ## previousStudiesOthers                                       7.278e-01
    ## previousStudiesTecnica                                      7.229e-01
    ## income_bracket_normalized_on4fascia.bassa                   2.099e-01
    ## income_bracket_normalized_on4fascia.media                   1.182e+00
    ## income_bracket_normalized_on4LS                             6.895e+00
    ## dropped_on_180Y                                             1.262e+01
    ## highschool_grade.income_bracket_normalized_on4fascia.bassa -1.141e-02
    ## highschool_grade.income_bracket_normalized_on4fascia.media -2.306e-02
    ## highschool_grade.income_bracket_normalized_on4LS           -6.491e-02
    ## career_time_conv.income_bracket_normalized_on4fascia.bassa  8.025e-04
    ## career_time_conv.income_bracket_normalized_on4fascia.media  8.704e-04
    ## career_time_conv.income_bracket_normalized_on4LS           -1.458e-03
    ## exa_grade_average.dropped_on_180Y                           1.100e-01
    ##                                                            Std. Error z value
    ## (Intercept)                                                 1.533e+00   5.529
    ## exa_cfu_pass                                                1.487e-02  -8.773
    ## exa_grade_average                                           1.435e-02  -2.982
    ## highschool_grade                                            1.477e-02  -1.783
    ## career_time_conv                                            4.603e-04 -10.221
    ## stud_genderM                                                1.974e-01   2.067
    ## previousStudiesOthers                                       5.214e-01   1.396
    ## previousStudiesTecnica                                      4.254e-01   1.699
    ## income_bracket_normalized_on4fascia.bassa                   2.266e+00   0.093
    ## income_bracket_normalized_on4fascia.media                   2.206e+00   0.536
    ## income_bracket_normalized_on4LS                             3.873e+00   1.780
    ## dropped_on_180Y                                             6.004e+02   0.021
    ## highschool_grade.income_bracket_normalized_on4fascia.bassa  2.178e-02  -0.524
    ## highschool_grade.income_bracket_normalized_on4fascia.media  2.140e-02  -1.078
    ## highschool_grade.income_bracket_normalized_on4LS            3.554e-02  -1.827
    ## career_time_conv.income_bracket_normalized_on4fascia.bassa  6.552e-04   1.225
    ## career_time_conv.income_bracket_normalized_on4fascia.media  6.240e-04   1.395
    ## career_time_conv.income_bracket_normalized_on4LS            1.134e-03  -1.286
    ## exa_grade_average.dropped_on_180Y                           3.639e+02   0.000
    ##                                                            Pr(>|z|)    
    ## (Intercept)                                                3.22e-08 ***
    ## exa_cfu_pass                                                < 2e-16 ***
    ## exa_grade_average                                           0.00286 ** 
    ## highschool_grade                                            0.07463 .  
    ## career_time_conv                                            < 2e-16 ***
    ## stud_genderM                                                0.03872 *  
    ## previousStudiesOthers                                       0.16274    
    ## previousStudiesTecnica                                      0.08924 .  
    ## income_bracket_normalized_on4fascia.bassa                   0.92617    
    ## income_bracket_normalized_on4fascia.media                   0.59208    
    ## income_bracket_normalized_on4LS                             0.07505 .  
    ## dropped_on_180Y                                             0.98323    
    ## highschool_grade.income_bracket_normalized_on4fascia.bassa  0.60044    
    ## highschool_grade.income_bracket_normalized_on4fascia.media  0.28107    
    ## highschool_grade.income_bracket_normalized_on4LS            0.06776 .  
    ## career_time_conv.income_bracket_normalized_on4fascia.bassa  0.22064    
    ## career_time_conv.income_bracket_normalized_on4fascia.media  0.16305    
    ## career_time_conv.income_bracket_normalized_on4LS            0.19841    
    ## exa_grade_average.dropped_on_180Y                           0.99976    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  906.09  on 2444  degrees of freedom
    ## AIC: 944.09
    ## 
    ## Number of Fisher Scoring iterations: 17

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_leap)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6495834

## Ultima Idea

Fra il modello con tutte le variabili numeriche e quello con solo 4,
accumuliamo un notevole AIC, proviamo a mantenere tutte le variabili
numeriche e ad aggiungere le categorie:

``` r
filtered_df <- df %>% filter(stud_career_status != 'A')
filtered_df_no_na = na.omit(filtered_df)

numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
general_df = na.omit(numerical_df)

general_df$stud_gender = factor(filtered_df_no_na$stud_gender, ordered = F)
general_df$previousStudies = factor(filtered_df_no_na$previousStudies, ordered = F)
general_df$origins = factor(filtered_df_no_na$origins, ordered = F)
general_df$income_bracket_normalized_on4 = factor(filtered_df_no_na$income_bracket_normalized_on4, ordered = F)
general_df$dropped_on_180 = factor(filtered_df_no_na$dropped_on_180, ordered = F)

covariate <- paste("dropout ~", paste(names(general_df[,-which(names(general_df) == "dropout")]), collapse = " + "))

formula = as.formula(paste(covariate))

# Fit the model
model_opt <- glm(formula, data = general_df, family = binomial)

# Print the summary of the model
summary(model_opt)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = general_df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6688  -0.2847  -0.1944  -0.0802   4.6237  
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error z value
    ## (Intercept)                                96.845513  69.524464   1.393
    ## career_start_ay                            -1.731600   0.365004  -4.744
    ## stud_admission_score                        0.010442   0.010739   0.972
    ## stud_career_admission_age                   0.380631   0.146621   2.596
    ## exa_cfu_pass                               -0.128081   0.015343  -8.348
    ## exa_grade_average                          -0.063431   0.018135  -3.498
    ## exa_avg_attempts                            0.389916   0.242207   1.610
    ## stud_career_end_ay                          1.685015   0.362983   4.642
    ## highschool_grade                           -0.037653   0.009865  -3.817
    ## career_time_conv                           -0.009222   0.001088  -8.480
    ## stud_genderM                                0.329145   0.202540   1.625
    ## previousStudiesOthers                       0.452157   0.684213   0.661
    ## previousStudiesScientifica                  0.118441   0.345147   0.343
    ## previousStudiesTecnica                      0.865259   0.527201   1.641
    ## originsForeigner                            0.147374   1.164351   0.127
    ## originsMilanese                            -0.081983   0.215454  -0.381
    ## originsOffsite                              0.496998   0.381272   1.304
    ## income_bracket_normalized_on4fascia bassa   0.088652   0.256439   0.346
    ## income_bracket_normalized_on4fascia media   0.042141   0.230971   0.182
    ## income_bracket_normalized_on4LS            -0.352839   0.322202  -1.095
    ## dropped_on_180Y                            11.809590 582.851181   0.020
    ##                                           Pr(>|z|)    
    ## (Intercept)                               0.163629    
    ## career_start_ay                           2.09e-06 ***
    ## stud_admission_score                      0.330886    
    ## stud_career_admission_age                 0.009431 ** 
    ## exa_cfu_pass                               < 2e-16 ***
    ## exa_grade_average                         0.000469 ***
    ## exa_avg_attempts                          0.107432    
    ## stud_career_end_ay                        3.45e-06 ***
    ## highschool_grade                          0.000135 ***
    ## career_time_conv                           < 2e-16 ***
    ## stud_genderM                              0.104145    
    ## previousStudiesOthers                     0.508713    
    ## previousStudiesScientifica                0.731478    
    ## previousStudiesTecnica                    0.100749    
    ## originsForeigner                          0.899279    
    ## originsMilanese                           0.703565    
    ## originsOffsite                            0.192396    
    ## income_bracket_normalized_on4fascia bassa 0.729565    
    ## income_bracket_normalized_on4fascia media 0.855228    
    ## income_bracket_normalized_on4LS           0.273479    
    ## dropped_on_180Y                           0.983835    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  877.53  on 2442  degrees of freedom
    ## AIC: 919.53
    ## 
    ## Number of Fisher Scoring iterations: 17

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_opt)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ##  McFadden 
    ## 0.6606292

``` r
x = model.matrix( model_opt ) [ , -1 ]
y = new_df$dropout

detect.lindep(x) #"Suspicious column name(s):   exa_cfu_pass:dropped_on_180Y, exa_grade_average:dropped_on_180Y" (Cancellata la prima)
```

    ## [1] "No linear dependent column(s) detected."

``` r
adjr = leaps( x, y, method = "adjr2" )
names(adjr)
```

    ## [1] "which" "label" "size"  "adjr2"

``` r
bestmodel_adjr2_ind = which.max( adjr$adjr2 )
adjr$which[ bestmodel_adjr2_ind, ] 
```

    ##     1     2     3     4     5     6     7     8     9     A     B     C     D 
    ##  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE 
    ##     E     F     G     H     I     J     K 
    ## FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE

``` r
maxadjr(adjr,15)
```

    ##       1,2,3,4,5,6,7,8,9,10,11,13,17,18,19,20 
    ##                                        0.652 
    ##          1,2,3,4,5,6,7,8,9,10,11,13,17,18,20 
    ##                                        0.651 
    ##    1,2,3,4,5,6,7,8,9,10,11,13,16,17,18,19,20 
    ##                                        0.651 
    ##    1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20 
    ##                                        0.651 
    ##       1,2,3,4,5,6,7,8,9,10,11,13,16,17,18,20 
    ##                                        0.651 
    ##    1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,19,20 
    ##                                        0.651 
    ##             1,2,3,4,5,6,7,8,9,10,11,13,19,20 
    ##                                        0.651 
    ##          1,2,3,4,5,6,7,8,9,10,11,13,18,19,20 
    ##                                        0.651 
    ##    1,2,3,4,5,6,7,8,9,10,11,13,15,17,18,19,20 
    ##                                        0.651 
    ##       1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,20 
    ##                                        0.651 
    ##          1,2,3,4,5,6,7,8,9,10,11,13,17,19,20 
    ##                                        0.651 
    ##       1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,20 
    ##                                        0.651 
    ##         1,2,3,4,5,7,8,9,10,11,13,17,18,19,20 
    ##                                        0.651 
    ## 1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20 
    ##                                        0.651 
    ##       1,2,3,4,5,6,7,8,9,10,11,13,15,17,18,20 
    ##                                        0.651

Siamo riusciti quindi a raggiungere un adjR2 = 0.652, meglio, per ora,
non risulta possibile.

Qui sotto il modello:

``` r
leap_matrix = data.frame(x[, c(1,2,3,4,5,6,7,8,9,10,11,13,17,18,19,20 )])
leap_matrix$dropout = new_df$dropout

covariate <- paste("dropout ~", paste(names(leap_matrix[,-which(names(leap_matrix) == "dropout")]), collapse = " + "))

formula = as.formula(paste(covariate))

# Fit the model
model_leap <- glm(formula, data = leap_matrix, family = binomial)

# Print the summary of the model
summary(model_leap)
```

    ## 
    ## Call:
    ## glm(formula = formula, family = binomial, data = leap_matrix)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7046  -0.2844  -0.1969  -0.0813   4.6166  
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error z value
    ## (Intercept)                                95.361893  69.158035   1.379
    ## career_start_ay                            -1.702157   0.362584  -4.695
    ## stud_admission_score                        0.011562   0.010564   1.095
    ## stud_career_admission_age                   0.381823   0.145176   2.630
    ## exa_cfu_pass                               -0.127469   0.015241  -8.364
    ## exa_grade_average                          -0.064271   0.018092  -3.553
    ## exa_avg_attempts                            0.385579   0.240971   1.600
    ## stud_career_end_ay                          1.656255   0.360694   4.592
    ## highschool_grade                           -0.036883   0.009505  -3.880
    ## career_time_conv                           -0.009114   0.001079  -8.444
    ## stud_genderM                                0.335473   0.202075   1.660
    ## previousStudiesOthers                       0.404179   0.538153   0.751
    ## previousStudiesTecnica                      0.762679   0.424096   1.798
    ## income_bracket_normalized_on4fascia.bassa   0.082961   0.252213   0.329
    ## income_bracket_normalized_on4fascia.media   0.037712   0.227524   0.166
    ## income_bracket_normalized_on4LS            -0.359592   0.320279  -1.123
    ## dropped_on_180Y                            11.817012 583.021802   0.020
    ##                                           Pr(>|z|)    
    ## (Intercept)                               0.167926    
    ## career_start_ay                           2.67e-06 ***
    ## stud_admission_score                      0.273727    
    ## stud_career_admission_age                 0.008537 ** 
    ## exa_cfu_pass                               < 2e-16 ***
    ## exa_grade_average                         0.000382 ***
    ## exa_avg_attempts                          0.109575    
    ## stud_career_end_ay                        4.39e-06 ***
    ## highschool_grade                          0.000104 ***
    ## career_time_conv                           < 2e-16 ***
    ## stud_genderM                              0.096886 .  
    ## previousStudiesOthers                     0.452624    
    ## previousStudiesTecnica                    0.072120 .  
    ## income_bracket_normalized_on4fascia.bassa 0.742207    
    ## income_bracket_normalized_on4fascia.media 0.868353    
    ## income_bracket_normalized_on4LS           0.261545    
    ## dropped_on_180Y                           0.983829    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2585.74  on 2462  degrees of freedom
    ## Residual deviance:  879.65  on 2446  degrees of freedom
    ## AIC: 913.65
    ## 
    ## Number of Fisher Scoring iterations: 17

``` r
# adjr2 per il modello con interazione: (Usiamo McFadden)
pseudo_r2 <- pR2(model_leap)
```

    ## fitting null model for pseudo-r2

``` r
pseudo_r2['McFadden']
```

    ## McFadden 
    ## 0.659807

## Predizione

Applichiamo adesso il modello lineare sugli studenti che non hanno
ancora completato il loro percorso universitario: (NON FUNZIONA, va
sistemato)

``` r
if(FALSE){
  # This is a block comment in R.
  # You can write multiple lines of comments here.
  # None of these lines will be executed.

filtered_df <- df %>% filter(stud_career_status == 'A')
filtered_df$dropout <-NULL
filtered_df$stud_career_end_ay <- NULL
filtered_df$stud_career_end_date <- NULL
filtered_df_no_na = na.omit(filtered_df)

new_data <- filtered_df_no_na

new_data$stud_gender = factor(filtered_df_no_na$stud_gender, ordered = F)
new_data$previousStudies = factor(filtered_df_no_na$previousStudies, ordered = F)
new_data$origins = factor(filtered_df_no_na$origins, ordered = F)
new_data$income_bracket_normalized_on4 = factor(filtered_df_no_na$income_bracket_normalized_on4, ordered = F)
new_data$dropped_on_180 = factor(filtered_df_no_na$dropped_on_180, ordered = F)

new_data$dropout <- NA
new_data <- new_data[names(new_df)]
new_data$dropout <- NULL

#new_data$dropped_on_180 <- NULL 

predicted_values <- predict(model_opt, newdata = new_data, type = "response")
plot(new_data$exa_cfu_pass, predicted_values, main="Predizione di dropout / CFU acquisiti nel primo semestre", 
     xlab="CFU sostenuti", 
     ylab="Predizione Dropout")

plot(new_data$exa_grade_average, predicted_values, main="Predizione di dropout / Media pesata nel primo semestre", 
     xlab="media pesata", 
     ylab="Predizione Dropout")
}
```
