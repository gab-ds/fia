from scipy.spatial import KDTree

class GeneticAlgorithm:

    __tourist_activities: list[dict] = []
    __hotels: list[dict] = []

    __ta_points: list[float, float] = []
    __ta_tree: KDTree

    def __init__(self, activities: list[dict]):
        for activity in activities:
            if activity['is_alloggio']:
                self.__hotels.append(activity)
                continue
            self.__tourist_activities.append(activity)
            self.__ta_points.append([activity['lat'], activity['lon']])
        self.__ta_tree = KDTree(self.__ta_points)
        print(self.__ta_points)

    def find_neighbors(self, activity: dict, n: int):
        _, indices = self.__ta_tree.query([activity['lat'], activity['lon']], n)
        return [self.__tourist_activities[i] for i in indices]