import random
import time

import geopy.distance
import numpy
from scipy.spatial import KDTree

from prefs import Preferences


class GeneticAlgorithm:
    __population_size: int
    __activity_weight: int
    __distance_weight: int
    __max_distance: int
    __elite_size: int
    __crossover_rate: float
    __mutation_rate: float

    __generation: int = 1
    __population: list[list[int]] = []
    __best_population: tuple[int, list[list[int]]] = (0, [])
    __best_individual: tuple[int, list[int]] = (0, [])

    __tourist_activities: dict[int, dict] = {}
    __accomodations: dict[int, dict] = {}

    __ta_points: list[float, float] = []
    __ta_map: list[int] = []
    __ta_tree: KDTree
    __preferences: Preferences

    def __init__(self, activities: list[dict], preferences: Preferences, population_size: int = 10,
                 activity_weight: int = 1, distance_weight: int = 1, elite_size: int = 2, max_distance: int = 5000,
                 crossover_rate: float = 0.5, mutation_rate: float = 0.1):
        self.__population_size = population_size
        self.__activity_weight = activity_weight
        self.__distance_weight = distance_weight
        self.__max_distance = max_distance
        self.__elite_size = elite_size
        self.__crossover_rate = crossover_rate
        self.__mutation_rate = mutation_rate
        for activity in activities:
            if activity['is_alloggio']:
                self.__accomodations[activity['id']] = activity
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

        return total * self.__activity_weight

    @staticmethod
    def distance_between(a: dict, b: dict) -> int:
        return int(geopy.distance.geodesic((a['lat'], a['lon']), (b['lat'], b['lon'])).meters)

    def fitness_distance(self, a: dict, b: dict) -> int:
        total = self.__max_distance - GeneticAlgorithm.distance_between(a, b)
        if total < 0:
            return 0
        return total * self.__distance_weight

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
            a = self.__accomodations[curr] if i == 4 else self.__tourist_activities[curr]
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

    def __generate_individual(self) -> list[int] | None:
        individual = [0, 0, 0, 0, 0]
        accomodation = random.choice(self.__accomodations)
        self.__accomodations.pop(accomodation['id'])
        individual[4] = accomodation['id']
        indices = self.__ta_tree.query([accomodation['lat'], accomodation['lon']], 4)
        activity_count = 0
        last_activity = None
        for j in indices:
            activity = self.__tourist_activities[self.__ta_map[j]]
            comparator = accomodation if last_activity is None else last_activity
            if self.distance_between(activity, comparator) > self.__max_distance:
                if activity_count < 2:
                    return None
                break
            individual[3 - activity_count] = activity['id']
            last_activity = activity
            activity_count += 1
        return individual

    def init_population(self) -> list[list[int]]:
        population = []
        for i in range(self.__population_size):
            individual = None
            while individual is None:
                individual = self.__generate_individual()
            population.append(individual)
        return population

    def selection(self, population: list[list[int]]) -> list[list[int]]:
        new_population = []
        p = population.copy()
        fitness = [self.fitness_function(individual) for individual in p]
        p.sort(key=lambda i: fitness[i], reverse=True)
        new_population.extend(p[:self.__elite_size])

        # riduciamo le chance di vincita nella roulette wheel degli elementi appartenenti all'élite
        for i in range(self.__elite_size):
            fitness[i] *= 0.8

        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        selected = numpy.random.choice(p, size=len(p) - len(new_population), p=probabilities)
        new_population.extend(selected)
        return new_population

    def crossover(self, population: [list[list[int]]]) -> list[list[int]]:
        new_population = population.copy()
        numpy.random.shuffle(new_population)

        for i in range(0, len(new_population), 2):
            if numpy.random.rand() < self.__crossover_rate:
                parent_a = new_population[i]
                parent_b = new_population[i + 1]

                child_a = parent_a[:2] + parent_b[2:]
                child_b = parent_b[:2] + parent_a[2:]

                if self.distance_between(parent_a[1], parent_b[2]) <= self.__max_distance \
                        and len(child_a) == len(set(child_a)):
                    new_population[i] = child_a

                if self.distance_between(parent_b[1], parent_a[2]) <= self.__max_distance \
                        and len(child_b) == len(set(child_b)):
                    new_population[i + 1] = child_b

        return new_population

    def mutate(self, population: [list[list[int]]]) -> list[list[int]]:
        new_population = population.copy()
        for i, individual in enumerate(new_population):
            if numpy.random.rand() < self.__mutation_rate:
                j1, j2 = numpy.random.choice(len(individual) - 1, 2, replace=False)
                individual[j1], individual[j2] = individual[j2], individual[j1]
                if self.fitness_activity(individual) > 0:
                    new_population[i] = individual
        return new_population

    def evolve(self, population: [list[list[int]]]) -> list[list[int]]:
        new_population = population.copy()
        new_population = self.selection(new_population)
        new_population = self.crossover(new_population)
        new_population = self.mutate(new_population)
        return new_population

    def execute_ga(self) -> (list[int], (list[list[int]], list[list[int]])):
        start_time = time.time()

        self.__population = self.init_population()

        while time.time() - start_time < 2:
            total_fitness = 0
            for individual in self.__population:
                fitness = self.fitness_function(individual)
                if fitness > self.__best_individual[0]:
                    self.__best_individual = fitness, individual
                    print(f"Nuovo individuo migliore -> Fitness: {fitness}")
                total_fitness += fitness
            print(f"Generazione {self.__generation} -> Fitness totale: {total_fitness}")
            if total_fitness >= self.__best_population[0]:
                self.__best_population = total_fitness, self.__population
                print("-> Questa generazione è la migliore finora, aggiorno la migliore generazione.")
            self.__population = self.evolve(self.__population)
            self.__generation += 1

        return self.__best_individual, (self.__best_population, self.__population)