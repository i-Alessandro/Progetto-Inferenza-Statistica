library(readxl)

################################################################################
#IMPORTANTE!! Cambiare la working directory con quella del proprio computer!
setwd("C:/Users/alewi/Documents/University/HKUST & PoliMi/II Semestre/Inferenza Statistica/Progetto")

################################################################################

#Creazione del Dataframe
Data <- read_excel("./Dati/ESPERIMENTI.xlsx")
View(Data)

#Info generali sul Dataframe
dim(Data)
head(Data)
summary(Data)
sum(is.na(Data))
print(sapply(Data, function(x) any(is.na(x))))

#Riduzione del Dataframe per togliere le persone che si sono iscritte ma non 
#hanno mai dato (o passato) nemmeno un esame.
test = Data[Data$exa_grade_average != 0, ]
attach(test)

#Plot della correlazione fra varie Features
dev.new()
pairs(test[, c('stud_admission_score', 'exa_grade_average')], pch=16)

#Un Primo Modello Lineare per il lavoro. 
g = lm( exa_grade_average ~ stud_admission_score + exa_cfu_pass + exa_avg_attempts + highschool_grade + gender_binary*stud_admission_score + gender_binary*exa_cfu_pass + gender_binary*exa_avg_attempts + gender_binary*highschool_grade)
summary( g )

h = lm( exa_grade_average ~ stud_admission_score + highschool_grade + check_fascia_alta +  I(stud_admission_score * check_fascia_alta) + I(highschool_grade * check_fascia_alta))
summary(h)

      