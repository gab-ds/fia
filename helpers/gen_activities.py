import csv
import random
import pycristoforo as pyc
from sys import argv

# Definizione delle possibili valori per le colonne
localita = ["Mare", "Montagna", "Città"]
is_alloggio = ["Sì", "No"]
tipologie_alloggio = ["Hotel", "Bed & Breakfast", "Villaggio turistico", "Ostello"]
tipologie_attivita_turistica = ["All'aperto", "Visite storico-culturali", "Gastronomia"]
cibo = ["Vegan", "Vegetarian", "Gluten-free", "Nessuna informazione"]
animali_ammessi = ["Sì", "No"]
souvenir_locali = ["Sì", "No"]
stagione_rilevante = ["Autunno-inverno", "Primavera-estate", "Nessuna informazione"]

# Funzione per generare una riga del dataset
def generate_row(latitude, longitude, id):
    row = []
    row.append(id)
    row.append(latitude)
    row.append(longitude)

    # Genera i valori delle colonne opzionali
    is_alloggio_value = random.choice(is_alloggio)
    row.append(is_alloggio_value)
    tipologie = tipologie_alloggio if is_alloggio_value == "Sì" else tipologie_attivita_turistica
    row.append(random.choice(tipologie))
    row.append(random.randint(20, 500))
    row.append(random.choice(cibo))
    row.append(random.choice(animali_ammessi))
    row.append(random.choice(stagione_rilevante))
    row.append(random.choice(souvenir_locali))
    row.append(random.choice(stagione_rilevante))
    return row


num_activities = int(argv[1]) if len(argv) > 1 else 10
id = 1
dataset = []
country = pyc.get_shape("Italy")
points = pyc.geoloc_generation(country, num_activities, "Italy")

for point in points:
    lon, lat = point['geometry']['coordinates']
    dataset.append(generate_row(lat, lon, id))
    id += 1

# Salva il dataset in un file CSV
with open("../datasets/attivita.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Latitudine", "Longitudine", "Alloggio", "Tipologia", "Prezzo", "Cibo", "Animali ammessi", "Stagione rilevante", "Souvenir locali", "Località"])
    writer.writerows(dataset)

print("CSV generato: attivita.csv")