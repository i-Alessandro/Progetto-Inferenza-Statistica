---
title: "Esplorazione"
author: "Alessandro Wiget, Sofia Sannino"
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
```

## ESPLORAZIONE E ANALISI QUALITATIVA INIZIALE

Importiamo il Dataset, presente nella cartella `Dati/`:

```{r}
setwd("C:/Users/sanni/OneDrive/Desktop/POLIMI/3 anno/inferenza/Progetto-Inferenza-Statistica-main")
df <- read_excel("./Dati/Dropout20240226_IngMate.xlsx")
View(df)
summary(df)
```

Il dataset è piuttosto complesso: vi sono diverse variabili, in particolare diverse variabili categoriche.

Omettiamo dal dataset le colonne che contengono l'id anonimizzato degli studenti, il dato career_time (che è doppio) e un id che identifica quando gli studenti si sono iscritti al Politecnico:

```{r}
 
df$career_anonymous_id <- NULL
df$career_time <- NULL
df$stud_career_degree_start_id <- NULL
df$stud_ofa_flst <- NULL
df$stud_ofa_fltp <- NULL

filtered_df <- df %>% filter(stud_career_status != 'A')
```

-   Creiamo anche una versione del dataset che contiene solo le variabili numeriche e omette quelle categoriche. Inoltre togliamo tutte le righe che contengono N.A. poichè erano legate a due fattori:\
    Persone che si sono disiscritte prima della fine del primo semestre (e che quindi hano una media che potrebbe essere N.A.).

-   Persone che possiedono un titolo di studio estero, e che quindi risulta come N.A. nel dataset.

-   Eventuali altre incongruenze trascurabili rispetto alla numerosità del campione.

```{r}
conteggio_na <- colSums(is.na(filtered_df))
conteggio_na #NA solo in exa_grade_average e highschool_grade come già detto 
```

```{r}
numerical_vars <- sapply(filtered_df, is.numeric)  # Find numeric columns
numerical_df <- filtered_df[, numerical_vars]  # Subset dataframe with numeric columns
numerical_df = na.omit(numerical_df)
```

Siccome l'obiettivo dell'analisi dati è prevedere quali studenti sono a rischio dropout tra quelli con carriera attiva e portare avanti azioni targhetizzate per prevenirlo, facciamo un'analisi qualitativa del dataset partendo dalla percentuale di dropout sul totale del campione.

```{r}
hist(numerical_df$dropout)

```

Notiamo che poco meno di un quarto del campione coinvolto ha effettivamente lasciato il Politecnico. Da qui l'importanza di analizzare il fenomeno

Visualizziamo le relazioni tra coppie di variabili, la loro correlazione nel campione e la loro funzione densità approssimata.

``` R
```

```{r}
ggpairs(data = numerical_df, title ="Relationships between variables",
        lower = list(continuous=wrap("points", alpha = 0.5, size=0.1)))
```

Le variabili sono tante e R ci indica che le correlazioni rilevanti sono diverse. E' necessario semplificare il modello per iniziare l'interpretazione.

Osserviamo se esistono correlazioni significative fra i dati numerici, in un altro modo:

```{r, echo=FALSE}
X = numerical_df[, -5]
corrplot(cor(X), method='color')
```

Sembra ci sia una correlazione significativa tra la media dei voti dello studente e il numero di tentativi che impiega in media per passare gli esami (approfondire sto dato che boh).

Data la presenza di variabili categoriche, si può iniziare l'analisi da un modello di regressione logistica.
