# Carica le librerie necessarie
library(readxl)
library(dplyr)
library(faraway)
library(leaps)
library(MASS)
library(GGally)
library(BAS)
library(rgl)
library(corrplot)
library(pscl)

# Imposta la directory di lavoro
setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto")

# Carica il dataset
df <- read_excel("./Dati/Dropout20240226_IngMate.xlsx")

# Filtra gli studenti con carriere in corso
df$career_anonymous_id <- NULL
df$career_time <- NULL
df$stud_career_degree_start_id <- NULL
filtered_df <- df %>% filter(stud_career_status != 'A')

# Seleziona le variabili numeriche
numerical_vars <- sapply(filtered_df, is.numeric)
numerical_df <- filtered_df[, numerical_vars]
numerical_df = na.omit(numerical_df)

# Controlla le correlazioni tra i dati numerici
X = numerical_df[, -4]
corrplot(cor(X), method='color')

# Esegui la regressione logistica
formula <- as.formula(paste("dropout ~", paste(names(numerical_df[,-which(names(numerical_df) == "dropout")]), collapse = " + "))) 
model <- glm(formula, data = numerical_df)
summary(model)
                      
# Trova il miglior modello usando il metodo di selezione automatica
x = model.matrix( model ) [ , -1 ]
y = numerical_df$dropout
adjr = leaps( x, y, method = "adjr2" )
bestmodel_adjr2_ind = which.max( adjr$adjr2 )
adjr$which[bestmodel_adjr2_ind, ] 
maxadjr(adjr,50)

# Seleziona il modello con le caratteristiche dalle colonne 4,5,8,9
selected_df <- numerical_df[, c(4,5,6,9,10)]
formula <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))
model_opt <- glm(formula, data = selected_df, family = binomial)
summary(model_opt)
pseudo_r2 <- pR2(model_opt)
pseudo_r2['McFadden']

# Controlla se il modello può essere migliorato introducendo l'interazione tra le covariate più correlate
covariate <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))
interazioni = " + exa_cfu_pass * exa_grade_average"
formula = as.formula(paste(covariate, interazioni))
model_opt <- glm(formula, data = selected_df, family = binomial)
summary(model_opt)
pseudo_r2 <- pR2(model_opt)
pseudo_r2['McFadden']

# Rimuovi la caratteristica superflua exa_cfu_pass
covariate <- paste("dropout ~", paste(names(selected_df[,-which(names(selected_df) == "dropout")]), collapse = " + "))
interazioni = " + exa_cfu_pass * exa_grade_average"
rimuovere = "- exa_cfu_pass"
formula = as.formula(paste(covariate, interazioni, rimuovere))
model_opt <- glm(formula, data = selected_df, family = binomial)
summary(model_opt)
pseudo_r2 <- pR2(model_opt)
pseudo_r2['McFadden']

# Applica il modello lineare agli studenti che non hanno ancora completato il loro corso universitario
new_data = filtered_df <- df %>% filter(stud_career_status == 'A')
new_data_vars <- sapply(new_data, is.numeric)
new_data <- new_data[, numerical_vars]
new_data = new_data[,c(-4,-8)]
new_data = na.omit(new_data)
predicted_values <- predict(model_opt, newdata = new_data, type = "response")
plot(new_data$exa_cfu_pass, predicted_values, main="Predizione di dropout / CFU acquisiti nel primo semestre", 
     xlab="CFU sostenuti", 
     ylab="Predizione Dropout")
plot(new_data$exa_grade_average, predicted_values, main="Predizione di dropout / Media pesata nel primo semestre", 
     xlab="media pesata", 
     ylab="Predizione Dropout")