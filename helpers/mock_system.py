import json

import pandas
import requests
from geopy.distance import geodesic

dataset_visitors = pandas.read_csv('../datasets/visitatori.csv')
dataset_activities = pandas.read_csv('../datasets/attivita.csv')

field_mapping = {
    'ID': 'id',
    'Latitudine': 'lat',
    'Longitudine': 'lon',
    'Alloggio': 'is_accomodation',
    'Tipologia': 'category',
    'Cibo': 'food_type',
    'Stagione rilevante': 'season',
    'Località': 'location',
    'Località preferita': 'location',
    'Tipologia di alloggio preferita': 'accomodation_type',
    'Preferenze alimentari': 'food_type',
    'Tipologia di attività preferita': 'activity_type',
    'Animali domestici': 'pets_allowed',
    'Budget': 'budget',
    'Prezzo': 'price',
    'Souvenir locali': 'souvenirs_available',
    'Stagione preferita': 'season'
}

enum_mappings = {
    'location': {
        'Mare': 1,
        'Montagna': 2,
        'Città': 3,
        'Nessuna preferenza': 4
    },
    'accomodation_type': {
        'Hotel': 1,
        'Bed & Breakfast': 2,
        'Villaggio turistico': 3,
        'Ostello': 4,
        'Nessuna preferenza': 5,
    },
    'food_type': {
        'Vegan': 1,
        'Vegetarian': 2,
        'Gluten-free': 3,
        'Nessuna preferenza': 4,
        'Nessuna informazione': 4
    },
    'activity_type': {
        'All\'aperto': 1,
        'Visite storico-culturali': 2,
        'Gastronomia': 3,
        'Nessuna preferenza': 4
    },
    'budget': {
        'Basso': 1,
        'Medio': 2,
        'Alto': 3,
        'Flessibile': 4
    },
    'season': {
        'Autunno-inverno': 1,
        'Primavera-estate': 2,
        'Nessuna preferenza': 3,
        'Nessuna informazione': 3
    },
    'pets_allowed': {
        'Sì': True,
        'No': False
    }
}

enum_mappings['food'] = enum_mappings['food_type']
enum_mappings['relevant_season'] = enum_mappings['season']
enum_mappings['souvenirs_available'] = enum_mappings['pets_allowed']
enum_mappings['is_accomodation'] = enum_mappings['pets_allowed']

# Rinomina i campi
dataset_visitors = dataset_visitors.rename(columns=field_mapping)
dataset_activities = dataset_activities.rename(columns=field_mapping)

# Converti i valori in enum
for field, mapping in enum_mappings.items():
    if field in dataset_visitors:
        dataset_visitors[field] = dataset_visitors[field].map(mapping)

for field, mapping in enum_mappings.items():
    if field in dataset_activities:
        dataset_activities[field] = dataset_activities[field].map(mapping)

random_visitor = dataset_visitors.sample(n=1).to_dict('records')[0]
del random_visitor['id']

dataset_accomodations = dataset_activities[dataset_activities['is_accomodation'] == True]

random_activities = []

while len(random_activities) == 0:
    tourist_activities = []
    accomodations = []
    random_accomodation = dataset_accomodations.sample(n=1).to_dict('records')[0]
    lat = random_accomodation['lat']
    lon = random_accomodation['lon']
    # Prima scrematura: selezioniamo una struttura ricettiva con almeno 5 attività in un raggio tra ~ 10 km e ~ 20 km
    activities = (dataset_activities
                  .query(f'{lat - 0.1} < lat < {lat + 0.1}')
                  .query(f'{lon - 0.1} < lon < {lon + 0.1}')
                  .to_dict('records'))
    if len(activities) < 5:
        continue
    for a in activities:
        distance = geodesic((lat, lon), (a['lat'], a['lon'])).meters
        if distance > 5000:
            continue
        if a['is_accomodation']:
            accomodations.append(a)
            continue
        tourist_activities.append(a)

    # Se ci sono meno di 5 attività turistiche in zona, preferiamo un altro gruppo
    # Seppur siano sufficienti 2 attività turistiche per una soluzione ammissibile, in tal caso non vi sarebbe necessità
    # di utilizzare l'algoritmo genetico per calcolare il percorso
    # Inoltre, in questo modo ci concentriamo sui casi più interessanti su cui provare l'algoritmo
    if len(tourist_activities) < 5:
        continue
    # Facciamo lo stesso con le strutture ricettive: ne basterebbe una sola, ma in questo modo abbiamo un dataset
    # di partenza più variegato su cui provare l'algoritmo
    if len(accomodations) < 5:
        continue
    random_activities = activities

for index, activity in enumerate(random_activities):
    category_key = 'accomodation_type' if activity['is_accomodation'] else 'activity_type'
    activity['category'] = enum_mappings[category_key][activity['category']]
    random_activities[index] = activity

r = requests.post('http://127.0.0.1:8000/', json={'preferences': random_visitor, 'activities': random_activities})

print(json.dumps(json.loads(r.text), indent=4))
