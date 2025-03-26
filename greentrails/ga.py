import random
import time
from collections import Counter
from functools import cache

import numpy
from matplotlib import pyplot
from scipy.spatial import KDTree

from .activities import Activity
from .options import Options


class GeneticAlgorithm:
    population_size: int
    activity_weight: int
    distance_weight: int
    completeness_weight: int
    max_distance: int
    elite_size: int
    crossover_rate: float
    mutation_rate: float

    __last_best_generation: int
    __generation: int = 1
    __population: list[list[int]] = []
    __best_population: tuple[list[list[int]], int] = ([], 0)
    __best_individual: tuple[list[int], int, int, float] = ([], 0, 1, 0)

    __activities: dict[int, Activity] = {}
    __tourist_activities: dict[int, Activity] = {}
    __accomodations: dict[int, Activity] = {}

    __ta_points: list[float, float] = []
    __ta_map: list[int] = []
    __ta_tree: KDTree

    __accomodations_points: list[float, float] = []
    __accomodations_map: list[int] = []
    __accomodations_tree: KDTree

    __preferences: Options

    def __init__(self, activities: list[Activity], preferences: Options, population_size: int = 10,
                 activity_weight: int = 200, distance_weight: int = 1, completeness_weight: int = 1000,
                 elite_size: int = 2, max_distance: int = 5000,
                 crossover_rate: float = 0.5, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.activity_weight = activity_weight
        self.distance_weight = distance_weight
        self.completeness_weight = completeness_weight
        self.max_distance = max_distance
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        for activity in activities:
            self.__activities.update({activity.id: activity})
            target_dict = self.__tourist_activities
            target_map = self.__ta_map
            target_points = self.__ta_points
            if activity.is_accomodation:
                target_dict = self.__accomodations
                target_map = self.__accomodations_map
                target_points = self.__accomodations_points
            target_dict.update({activity.id: activity})
            target_map.append(activity.id)
            target_points.append([activity.latitude, activity.longitude])
        if len(self.__accomodations) == 0:
            raise ValueError('Non ci sono strutture ricettive')
        if len(self.__tourist_activities) < 5:
            raise ValueError('Non ci sono abbastanza attività turistiche')
        self.__ta_tree = KDTree(self.__ta_points)
        self.__accomodations_tree = KDTree(self.__accomodations_points)
        self.__preferences = preferences

    def __distance_ids(self, a: int, b: int) -> float:
        if a not in self.__activities or b not in self.__activities:
            raise ValueError("Attività non presenti")
        activity_a = self.__activities[a]
        activity_b = self.__activities[b]
        return activity_a.distance_from(activity_b)

    def fitness_distance(self, individual: tuple[int]):
        total_distance = 0
        for idx, i in enumerate(individual):
            if i == 0:
                if idx > 1:
                    return 0
                continue
            next_idx = idx + 1
            if next_idx > len(individual) - 1:
                break

            total_distance += int(self.__distance_ids(i, individual[next_idx]))
            if total_distance > self.max_distance:
                return 0
        return self.max_distance - total_distance

    def find_neighbors(self, activity: Activity, n: int, accomodations: bool = False) -> list[Activity]:
        target_dict = self.__tourist_activities
        target_map = self.__ta_map
        target_tree = self.__ta_tree
        if accomodations:
            target_dict = self.__accomodations
            target_map = self.__accomodations_map
            target_tree = self.__accomodations_tree
        _, indices = target_tree.query([activity.latitude, activity.longitude], n)
        result = []
        for i in indices:
            # In questo caso, vuol dire che il numero di punti presenti nel K-D Tree è minore del numero di punti richiesti
            if i == len(target_map):
                break
            index = target_map[i]
            # Se non fa parte del dizionario, allora si tratta una struttura ricettiva che è stata rimossa in precedenza
            if index not in target_dict:
                continue
            result.append(target_dict[index])
        return result

    def find_neighbors_in_range(self, activity: Activity, distance: int, accomodations: bool = False) -> list[Activity]:
        neighbors = self.find_neighbors(activity, 20, accomodations)
        result = []
        for neighbor in neighbors:
            if activity.distance_from(neighbor) > distance:
                continue
            if neighbor not in result:
                result.append(neighbor)
        return result

    def find_acceptable_neighbors(self, activity: Activity, accomodations: bool = False) -> list[Activity]:
        return self.find_neighbors_in_range(activity, self.max_distance, accomodations)

    def fitness_activity(self, activity: Activity) -> int:
        total = 0
        preferences = self.__preferences
        options = activity.options

        # corrispondenza località
        if preferences.location is None or preferences.location == options.location:
            total += 2

        # corrispondenza tipologia alloggio/attività turistica
        activity_preference = preferences.activity_type
        if activity.is_accomodation:
            activity_preference = preferences.accomodation_type

        if activity_preference is None or activity_preference == options.category:
            total += 2

        # corrispondenza tipologia cibo
        if preferences.food_type is None or preferences.food_type == options.food_type:
            total += 2

        # corrispondenza animali domestici
        if preferences.pets_allowed is None or preferences.pets_allowed == options.pets_allowed:
            total += 2

        # corrispondenza budget
        if preferences.budget == Options.Budget.FLEXIBLE or preferences.budget == options.budget:
            total += 2
        else:  # bonus (o malus) in base alla differenza di budget
            total += 2 * (preferences.budget.value - options.budget.value)

        # corrispondenza souvenir locali
        if not preferences.souvenirs_available or options.souvenirs_available:
            total += 2

        # corrispondenza stagione
        if preferences.season is None or preferences.season == options.season:
            total += 2
        elif options.season is None:  # punteggio dimezzato per le attività senza stagione rilevante
            total += 1

        return total * self.activity_weight

    @cache
    def fitness_function(self, individual: tuple[int]):

        activity_fitness = 0
        activity_ids = []
        last_activity_id = None
        accomodation = None

        for idx, activity_id in enumerate(individual):
            if activity_id == 0:
                # Scartiamo tutti gli individui con valori non-nulli prima dei valori nulli (es. [1 0 2 3 4])
                # Inoltre, scartiamo tutti gli individui con valori nulli dopo la seconda posizione (es. [0 0 0 1 2])
                if idx > 1 or last_activity_id is not None and last_activity_id != 0:
                    return 0
                last_activity_id = 0
                continue

            last_activity_id = activity_id

            # Scartiamo tutti gli individui con attività duplicate (es. [1 1 2 3 4])
            if activity_id in activity_ids:
                return 0
            activity_ids.append(activity_id)

            if activity_id not in self.__activities:
                raise ValueError(f"Individuo con attività non valide: {activity_id}")

            activity = self.__activities[activity_id]

            if activity.is_accomodation:
                # Scartiamo tutti gli individui con una struttura ricettiva in posizione non finale
                if idx < len(individual) - 1:
                    return 0
                accomodation = activity
            activity_fitness += self.fitness_activity(activity)

        # Dovrebbe essere impossibile avere individui con l'ultima attività che non è una struttura ricettiva
        # Ma non si sa mai, meglio controllare
        if accomodation is None:
            return 0

        distance_fitness = self.fitness_distance(individual)
        if distance_fitness <= 0:
            return 0
        completeness_fitness = (2 - (5 - len(activity_ids))) * self.completeness_weight
        total_fitness = activity_fitness + distance_fitness + completeness_fitness
        if total_fitness <= 0:
            return 0
        return total_fitness

    def __generate_individual(self) -> list[int] | None:
        individual = [0, 0, 0, 0, 0]
        accomodation: Activity = random.choice(list(self.__accomodations.values()))

        individual[4] = accomodation.id

        total_distance = 0
        curr = accomodation
        i = 0
        while i <= 3:
            neighbors = self.find_neighbors(curr, 5)
            neighbor = None
            for n in neighbors:
                if n.id not in individual:
                    neighbor = n
                    break
            if neighbor is None:
                if i >= 2:
                    break
                self.__accomodations.pop(accomodation.id)
                return None
            total_distance += int(curr.distance_from(neighbor))
            if total_distance > self.max_distance:
                if i >= 2:
                    break
                self.__accomodations.pop(accomodation.id)
                return None
            curr = neighbor
            individual[3 - i] = neighbor.id
            i += 1
        return individual

    def init_population(self) -> list[list[int]]:
        population = []
        for i in range(self.population_size):
            individual = None
            while individual is None:
                individual = self.__generate_individual()
            population.append(individual)
        return population

    def selection(self, population: list[list[int]]) -> list[list[int]]:
        new_population = []
        fitness = [(individual, self.fitness_function(tuple(individual))) for individual in population]
        fitness.sort(key=lambda x: x[1], reverse=True)

        new_population.extend([x[0] for x in fitness[:self.elite_size]])

        # riduciamo del 20% le chance di vincita nella roulette wheel degli elementi appartenenti all'élite
        for i in range(self.elite_size):
            fitness[i] = (fitness[i][0], fitness[i][1] * 0.8)

        total_fitness = sum(j for i, j in fitness)
        probabilities = [f[1] / total_fitness for f in fitness]
        for f in fitness:
            if f[1] < 0:
                print(f)
                print(population)
                print(fitness)
                raise ValueError("COME CAZZO È POSSIBILE?")
        selected_indices = numpy.random.choice(len(fitness), size=len(fitness) - len(new_population), p=probabilities)
        selected = [fitness[i][0] for i in selected_indices]
        new_population.extend(selected)
        return new_population

    @staticmethod
    def __crossover_individuals(parent_a: list[int], parent_b: list[int]) -> tuple[list[int], list[int]]:
        child_a = parent_a[:2] + parent_b[2:]
        child_b = parent_b[:2] + parent_a[2:]
        return child_a, child_b

    def crossover(self, population: [list[list[int]]]) -> list[list[int]]:
        new_population = []

        p = population.copy()
        numpy.random.shuffle(p)

        while len(p) > 0:
            parent_a = p.pop()
            inserted = False
            # Se è più alto della probabilità di crossover, per quest'individuo la riproduzione non avviene
            if numpy.random.random() < self.crossover_rate:
                pool = []
                for i in p:
                    if i != parent_a:
                        pool.append(i)
                for parent_b in pool:
                    child_a, child_b = self.__crossover_individuals(parent_a, parent_b)
                    if self.fitness_function(tuple(child_a)) > 0 and self.fitness_function(tuple(child_b)) > 0:
                        new_population.append(child_a)
                        new_population.append(child_b)
                        p.remove(parent_b)
                        inserted = True
                        break
            if not inserted:
                new_population.append(parent_a)

        return new_population

    def scramble(self, individual: list[int]) -> list[int]:
        start = 0
        end = 3
        for idx, i in enumerate(individual):
            if i == 0:
                continue
            start = idx
            break

        new_individual = individual[:start]
        subset = individual[start:end + 1]
        numpy.random.shuffle(subset)
        new_individual.extend(subset)
        new_individual.extend(individual[end + 1:])
        return new_individual

    def random_reset(self, individual: list[int]) -> list[int]:
        random_index = numpy.random.choice(len(individual))
        # Si pensi alla casistica in cui viene selezionato il primo elemento, ma l'individuo ha codifica [0 0 1 2 3]
        # Chiaramente, in questo caso preferiamo selezionare un altro elemento
        # In questo modo, dovremmo evitare codifiche non valide (es. [7 0 1 2 3])
        if random_index == 0 and individual[random_index + 1] == 0:
            random_index += 1
        #while random_index == 0 and individual[random_index + 1] == 0:
        #    random_index = numpy.random.choice(len(individual))
        is_accomodation = random_index + 1 == len(individual)
        target_dict = self.__accomodations if is_accomodation else self.__tourist_activities
        activity_id = individual[random_index]
        if activity_id == 0:
            activity_id = individual[random_index + 1]
        activity = target_dict[activity_id]
        neighbors = self.find_acceptable_neighbors(activity, is_accomodation)
        numpy.random.shuffle(neighbors)
        new_individual = individual.copy()
        for neighbor in neighbors:
            if neighbor.id in new_individual:
                continue
            tmp = individual.copy()
            tmp[random_index] = neighbor.id
            fitness = self.fitness_function(tuple(tmp))
            if fitness <= 0:
                continue
            new_individual = tmp
            break
        return new_individual.copy()

    def mutate(self, population: list[list[int]]) -> list[list[int]]:
        new_population = population.copy()
        for i, individual in enumerate(new_population):
            if numpy.random.random() < self.mutation_rate:
                new_individual = self.random_reset(individual)
                new_individual = self.scramble(new_individual)
                fitness = self.fitness_function(tuple(new_individual))
                if fitness > 0:
                    new_population[i] = new_individual

        return new_population

    def evolve(self, population: list[list[int]]) -> list[list[int]]:
        new_population = population.copy()
        new_population = self.selection(new_population)
        new_population = self.crossover(new_population)
        new_population = self.mutate(new_population)
        return new_population

    def execute_ga(self) -> (list[int], float):
        pyplot.xlabel('Generazione')
        pyplot.ylabel('Fitness')

        start_time = time.time()
        avg_fitnesses = [0]
        best_fitnesses = [0]

        self.__population = self.init_population()
        self.__last_best_generation = self.__generation

        while time.time() - start_time < 1.9:
            avgs = 0
            fitnesses = []
            best_individual_fitness = 0
            for individual in self.__population:
                fitness = self.fitness_function(tuple(individual))
                if fitness > best_individual_fitness:
                    best_individual_fitness = fitness
                if fitness > self.__best_individual[1]:
                    self.__best_individual = (individual.copy(), fitness, self.__generation, time.time() - start_time)
                    print(f"Nuovo individuo migliore {individual} -> Fitness: {fitness}")
                    print("-> Questa generazione contiene un nuovo individuo migliore, aggiorno la migliore generazione.")
                    self.__last_best_generation = self.__generation
                avgs += fitness
                fitnesses.append(fitness)
            best_fitnesses.append(best_individual_fitness)
            avgs /= self.population_size
            print(f"Generazione {self.__generation} -> Fitness medio: {avgs}")
            if avgs == 0:
                print("Generazione di individui non ammissibili, interrompo l'esecuzione.")
                break
            if avgs > self.__best_population[1]:
                self.__last_best_generation = self.__generation
                self.__best_population = (self.__population, avgs)
                print("-> Questa generazione ha il miglior valore di fitness medio finora, aggiorno la migliore generazione.")
            _, occurrences = Counter(fitnesses).most_common(1)[0]
            if occurrences == self.population_size:
                if self.mutation_rate >= 1:
                    print("Convergenza raggiunta, interrompo l'esecuzione.")
                    break
                print("Convergenza raggiunta, raddoppio la probabilità di mutazioni.")
                self.mutation_rate *= 2
            elif self.mutation_rate >= 0.5 and occurrences - self.population_size < 2 and self.__generation > 20:
                print("Generazione troppo eterogenea, dimezzo la probabilità di mutazioni.")
                self.mutation_rate /= 2
            if self.__generation - self.__last_best_generation > 50:
                print("Non ci sono stati miglioramenti per almeno 50 generazioni, interrompo l'esecuzione.")
                break
            avg_fitnesses.append(avgs)
            self.__population = self.evolve(self.__population)
            self.__generation += 1

        end_time = time.time() - start_time
        print(f"Soluzione trovata in {self.__best_individual[3]} secondi, esecuzione terminata dopo {end_time} secondi.")
        print(f"Individuo migliore: {self.__best_individual[0]}, con fitness {self.__best_individual[1]}")

        pyplot.plot(avg_fitnesses, label="Media")
        pyplot.plot(best_fitnesses, label="Miglior individuo")
        pyplot.plot(self.__best_individual[2], self.__best_individual[1], 'g.', label="Soluzione")
        pyplot.legend()
        pyplot.show()


        return self.__best_individual, end_time
