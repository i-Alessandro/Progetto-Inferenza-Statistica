import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.linear_model import LogisticRegression

# Creiamo un dataset di esempio con CFU, voto medio e una variabile target binaria
np.random.seed(42)
data = pd.DataFrame({
    'cfu': np.random.randint(0, 27, 100),
    'voto_medio': np.random.randint(18, 31, 100),
    'esito': np.random.choice([0, 1], 100)  # 0 = non pass, 1 = pass
})

# Addestriamo un modello di regressione logistica
X = data[['cfu', 'voto_medio']]
y = data['esito']
model = LogisticRegression()
model.fit(X, y)

# Funzione per calcolare il logit
def calculate_logit(cfu, voto_medio, model):
    logit = model.predict_proba([[cfu, voto_medio]])[0][1]
    return logit

# Funzione di aggiornamento dell'animazione
def update_plot(cfu, voto_medio):
    logit = calculate_logit(cfu, voto_medio, model)
    ax.clear()
    ax.bar(['Probability of Passing'], [logit], color='blue')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(f'CFU: {cfu}, Voto Medio: {voto_medio}, Logit: {logit:.2f}')
    plt.draw()

# Creiamo la figura
fig, ax = plt.subplots()
update_plot(data['cfu'].mean(), data['voto_medio'].mean())

# Creiamo i widget per i cursori
cfu_slider = widgets.IntSlider(min=data['cfu'].min(), max=data['cfu'].max(), step=1, value=int(data['cfu'].mean()), description='CFU')
voto_slider = widgets.IntSlider(min=data['voto_medio'].min(), max=data['voto_medio'].max(), step=1, value=int(data['voto_medio'].mean()), description='Voto Medio')

# Colleghiamo i cursori alla funzione di aggiornamento
widgets.interactive(update_plot, cfu=cfu_slider, voto_medio=voto_slider)

# Visualizziamo i cursori
display(cfu_slider, voto_slider)
plt.show()