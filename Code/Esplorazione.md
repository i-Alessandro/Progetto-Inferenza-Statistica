Esplorazione
================
Alessandro Wiget
2024-05-15

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
```

## Considerazioni Iniziali

Importiamo il Dataset, presente nella cartella `Dati/`:

``` r
setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto")
df <- read_excel("./Dati/Dropout20240226_IngMate.xlsx")
#View(df)
#summary(df)
```

Omettiamo dal dataset le colonne che contengono l’id anonimizzato degli
studenti, il dato career_tim (che è doppio) e un id che identifica
quando gli studenti si sono iscritti al Politecnico:

``` r
df$career_anonymous_id <- NULL
df$career_time <- NULL
df$stud_career_degree_start_id <- NULL
df$stud_ofa_flst <- NULL
df$stud_ofa_fltp <- NULL

filtered_df <- df %>% filter(stud_career_status != 'A')
```

Creiamo anche una versione del dataset che contiene solo le variabili
numeriche e omette quelle categoriche, abbiamo tolto tutte le righe che
contengono N.A. poichè erano legati a due fattori:  
- Persone che si sono disiscritte prima della fine del primo semestre (e
che quindi hano una media che potrebbe essere N.A.). - Persone che
possiedono un titolo di studio estero, e che quindi risulta come N.A.
nel dataset.

``` r
numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
numerical_df = na.omit(numerical_df)
```