from scipy.spatial import KDTree
import geopy.distance

from prefs import Preferences

activity_weight: int = 1
distance_weight: int = 1


class GeneticAlgorithm:
    __tourist_activities: dict[int, dict] = {}
    __hotels: dict[int, dict] = {}

    __ta_points: list[float, float] = []
    __ta_map: list[int] = []
    __ta_tree: KDTree
    __preferences: Preferences

    def __init__(self, activities: list[dict], preferences: Preferences):
        for activity in activities:
            if activity['is_alloggio']:
                self.__hotels[activity['id']] = activity
                continue
            self.__tourist_activities[activity['id']] = activity
            self.__ta_map.append(activity['id'])
            self.__ta_points.append([activity['lat'], activity['lon']])
        self.__ta_tree = KDTree(self.__ta_points)
        self.__preferences = preferences

    def find_neighbors(self, activity: dict, n: int):
        _, indices = self.__ta_tree.query([activity['lat'], activity['lon']], n)
        return [self.__tourist_activities[self.__ta_map[i]] for i in indices]

    def fitness_activity(self, activity: dict) -> int:
        total = 0
        preferences = self.__preferences

        # corrispondenza località
        if preferences.favourite_location == Preferences.Location.NO_PREFERENCE or preferences.favourite_location == \
                activity['localita']:
            total += 2

        # corrispondenza tipologia alloggio/attività turistica
        prefs_type = preferences.favourite_activity_type
        a_type = activity['tipologia']
        if activity['is_alloggio']:
            prefs_type = Preferences.favourite_activity_type

        if prefs_type == type(prefs_type).NO_PREFERENCE or prefs_type == a_type:
            total += 2

        # corrispondenza tipologia cibo
        if preferences.favourite_food_type == Preferences.Food.NO_PREFERENCE or preferences.favourite_food_type == \
                activity['cibo']:
            total += 2

        # corrispondenza animali domestici
        if not preferences.pets_allowed or activity['animali_domestici']:
            total += 2

        # corrispondenza budget
        budget = Preferences.Budget.HIGH
        if activity['prezzo'] <= 50:
            budget = Preferences.Budget.LOW
        elif activity['prezzo'] <= 150:
            budget = Preferences.Budget.MEDIUM

        if preferences.favourite_budget == Preferences.Budget.FLEXIBLE or preferences.favourite_budget == budget:
            total += 2
        else:
            total += 2 * (
                    preferences.favourite_budget.value - budget.value)  # bonus (o malus) in base alla differenza di budget

        # corrispondenza souvenir locali
        if not preferences.local_souvenirs_available or activity['souvenir']:
            total += 2

        # corrispondenza stagione
        if preferences.favourite_season == Preferences.Season.NO_PREFERENCE:
            total += 2
        elif activity['stagione_rilevante'] is None:  # punteggio dimezzato per le attività senza stagione rilevante
            total += 1
        elif preferences.favourite_season == activity['stagione_rilevante']:
            total += 2

        return total * activity_weight

    @staticmethod
    def distance_between(a: dict, b: dict) -> int:
        return int(geopy.distance.geodesic((a['lat'], a['lon']), (b['lat'], b['lon'])).meters)

    @staticmethod
    def fitness_distance(a: dict, b: dict) -> int:
        total = 5000 - GeneticAlgorithm.distance_between(a, b)
        if total < 0:
            return 0
        return total * distance_weight

    def fitness_function(self, trail: list[int]):
        if len(trail) != 5:  # Rifiutiamo tutti gli individui di dimensione diversa da 5
            raise ValueError('L\'individuo non è di dimensione 5')

        explored = []
        total = 0
        for i in range(4, 0, -1):
            curr = trail[i]
            prev = trail[i - 1]

            if curr in explored:  # Evitiamo gli individui con attività duplicate all'interno del percorso
                return 0
            explored.append(curr)
            a = self.__hotels[curr] if i == 4 else self.__tourist_activities[curr]
            total += self.fitness_activity(a)
            if prev == 0:
                break
            b = self.__tourist_activities[prev]
            if i == 1:
                total += self.fitness_activity(b)
            distance_fitness = self.fitness_distance(a, b)
            if distance_fitness <= 0:  # Evitiamo gli individui con attività troppo distanti tra loro (> 5 km)
                return 0
            total += distance_fitness

        return total
