import csv
import random
from sys import argv

# Definizione delle opzioni per le preferenze
locations = ["Mare", "Montagna", "Città", "Nessuna preferenza"]
accommodations = ["Hotel", "Bed & Breakfast", "Villaggio turistico", "Ostello", "Nessuna preferenza"]
dietary_preferences = ["Vegan", "Vegetarian", "Gluten-free", "Nessuna preferenza"]
activity_types = ["All'aperto", "Visite storico-culturali", "Gastronomia", "Nessuna preferenza"]
pet_preferences = ["Sì", "No"]
budgets = ["Basso", "Medio", "Alto", "Flessibile"]
souvenir_interests = ["Sì", "No"]
seasons = ["Autunno-inverno", "Primavera-estate", "Nessuna preferenza"]

# Generazione dei dati dei visitatori
num_visitors = int(argv[1]) if len(argv) > 1 else 10
data = []
for i in range(num_visitors):
    visitor = {
        "ID": i+1,
        "Località preferita": random.choice(locations),
        "Tipologia di alloggio preferita": random.choice(accommodations),
        "Preferenze alimentari": random.choice(dietary_preferences),
        "Tipologia di attività preferita": random.choice(activity_types),
        "Animale domestico": random.choice(pet_preferences),
        "Budget": random.choice(budgets),
        "Souvenir locali": random.choice(souvenir_interests),
        "Stagione preferita": random.choice(seasons)
    }
    data.append(visitor)

# Scrittura del CSV
with open("../datasets/visitatori.csv", "w", newline="") as csvfile:
    fieldnames = list(data[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for visitor in data:
        writer.writerow(visitor)

print("CSV generato: visitatori.csv")
