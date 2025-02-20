import random
import time

import numpy
from scipy.spatial import KDTree

from .activities import Activity
from .options import Options


class GeneticAlgorithm:
    population_size: int
    activity_weight: int
    distance_weight: int
    max_distance: int
    elite_size: int
    crossover_rate: float
    mutation_rate: float

    __last_best_generation: int
    __generation: int = 1
    __population: list[list[int]] = []
    __best_population: tuple[int, list[list[int]]] = (0, [])
    __best_individual: tuple[int, list[int]] = (0, [])

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
                 activity_weight: int = 200, distance_weight: int = 1, elite_size: int = 2, max_distance: int = 5000,
                 crossover_rate: float = 0.5, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.activity_weight = activity_weight
        self.distance_weight = distance_weight
        self.max_distance = max_distance
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        for activity in activities:
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
            # In questo caso, vuol dire che il numero di punti totali è minore del numero di punti richiesti
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

    def fitness_function(self, individual: list[int]):
        if len(individual) != 5:  # Rifiutiamo tutti gli individui di dimensione diversa da 5
            raise ValueError('L\'individuo non è di dimensione 5')

        total_fitness = 0
        avg_lat, avg_lon = 0, 0
        activity_ids = []
        last_activity_id = None
        accomodation = None
        for activity_id in reversed(individual):
            # Se l'ID corrente è 0, continuiamo con la prossima iterazione
            if activity_id == 0:
                last_activity_id = 0
                continue

            # Scartiamo tutti gli individui con valori non-nulli prima dei valori nulli (es. [1 0 2 3 4])
            if last_activity_id == 0:
                return 0

            last_activity_id = activity_id

            # Scartiamo tutti gli individui con attività duplicate (es. [1 1 2 3 4])
            if activity_id in activity_ids:
                return 0
            activity_ids.append(activity_id)

            if accomodation is None:
                # Dovrebbe essere impossibile avere individui con l'ultima attività che non è una struttura ricettiva
                # Ma non si sa mai, meglio controllare
                if activity_id not in self.__accomodations:
                    return 0
                activity = self.__accomodations[activity_id]
                accomodation = activity
            else:
                activity = self.__tourist_activities[activity_id]

            total_fitness += self.fitness_activity(activity)

            avg_lat += activity.latitude
            avg_lon += activity.longitude

        # Scartiamo tutti gli individui con meno di 3 attività nel percorso (es. [0 0 0 3 4])
        if len(activity_ids) < 3:
            return 0

        avg_lat /= len(activity_ids)
        avg_lon /= len(activity_ids)

        # Calcoliamo il fitness tra un'attività e il punto medio calcolato tra tutte le attività non nulle
        distance_fitness = int(self.max_distance - accomodation.distance_from_coords(avg_lat, avg_lon))

        # Scartiamo tutti gli individui con attività troppo distanti tra loro (non incluse in un raggio di MAX_DISTANCE km)
        if distance_fitness <= 0:
            return 0

        total_fitness += distance_fitness
        return total_fitness

    def __generate_individual(self) -> list[int] | None:
        individual = [0, 0, 0, 0, 0]
        accomodation = random.choice(list(self.__accomodations.values()))

        individual[4] = accomodation.id
        neighbors = self.find_acceptable_neighbors(accomodation)
        # neighbors = self.find_neighbors(accomodation, 10)

        acceptable_neighbors = neighbors.copy()
        """acceptable_neighbors = []
        for neighbor in neighbors:
            if accomodation.distance_from(neighbor) > self.max_distance:
                break
            if neighbor not in acceptable_neighbors:
                acceptable_neighbors.append(neighbor)"""

        # Se la struttura ricettiva non ha abbastanza attività vicine in un raggio di 5 km, allora scartala dalla lista di candidati
        if len(acceptable_neighbors) < 2:
            self.__accomodations.pop(accomodation.id)
            return None

        numpy.random.shuffle(acceptable_neighbors)
        for i in range(4):
            if len(acceptable_neighbors) == 0:
                break
            selected_neighbor = acceptable_neighbors.pop(0)
            individual[3 - i] = selected_neighbor.id

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
        fitness = [(individual, self.fitness_function(individual)) for individual in population]
        fitness.sort(key=lambda x: x[1], reverse=True)

        new_population.extend([x[0] for x in fitness[:self.elite_size]])

        # riduciamo le chance di vincita nella roulette wheel degli elementi appartenenti all'élite
        for i in range(self.elite_size):
            fitness[i] = (fitness[i][0], fitness[i][1] * 0.8)

        total_fitness = sum(j for i, j in fitness)
        probabilities = [f[1] / total_fitness for f in fitness]
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
                for parent_b in p:
                    child_a, child_b = self.__crossover_individuals(parent_a, parent_b)
                    if self.fitness_function(child_a) > 0 and self.fitness_function(child_b) > 0:
                        new_population.append(child_a)
                        new_population.append(child_b)
                        p.remove(parent_b)
                        inserted = True
                        break
            if not inserted:
                new_population.append(parent_a)

        return new_population

    def mutate(self, population: [list[list[int]]]) -> list[list[int]]:
        new_population = population.copy()
        for i, individual in enumerate(new_population):
            if numpy.random.random() < self.mutation_rate:
                random_index = numpy.random.choice(len(individual))
                # Si pensi alla casistica in cui viene selezionato il primo elemento, ma l'individuo ha codifica [0 0 1 2 3]
                # Chiaramente, in questo caso preferiamo selezionare un altro elemento
                # In questo modo, dovremmo evitare codifiche non valide (es. [7 0 1 2 3])
                while random_index + 1 < len(individual) and individual[random_index + 1] == 0:
                    random_index = numpy.random.choice(len(individual))
                is_accomodation = random_index + 1 == len(individual)
                target_dict = self.__accomodations if is_accomodation else self.__tourist_activities
                activity = target_dict[individual[random_index]] if individual[random_index] != 0 else target_dict[
                    individual[random_index + 1]]
                neighbors = self.find_acceptable_neighbors(activity, is_accomodation)
                numpy.random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor.id in individual:
                        continue
                    tmp = individual.copy()
                    tmp[random_index] = neighbor.id
                    if self.fitness_function(tmp) <= 0:
                        continue
                    new_population[i] = tmp.copy()

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

        while time.time() - start_time < 1.99:
            total_fitness = 0
            for individual in self.__population:
                fitness = self.fitness_function(individual)
                if fitness > self.__best_individual[0]:
                    self.__best_individual = fitness, individual.copy()
                    print(f"Nuovo individuo migliore {individual} -> Fitness: {fitness}")
                total_fitness += fitness
            print(f"Generazione {self.__generation} -> Fitness totale: {total_fitness}")
            if total_fitness == 0:
                print("Generazione di individui non ammissibili, interrompo l'esecuzione.")
                break
            if total_fitness > self.__best_population[0]:
                self.__last_best_generation = self.__generation
                self.__best_population = total_fitness, self.__population.copy()
                print("-> Questa generazione è la migliore finora, aggiorno la migliore generazione.")
            if self.__generation - self.__last_best_generation > 50:
                print("Non ci sono stati miglioramenti per almeno 50 generazioni, interrompo l'esecuzione.")
                break
            self.__population = self.evolve(self.__population)
            self.__generation += 1

        print(f"Soluzione trovata in {time.time() - start_time} secondi.")
        print(f"Individuo migliore: {self.__best_individual[1]}, con fitness {self.__best_individual[0]}")

        return self.__best_individual, (self.__best_population, self.__population)
