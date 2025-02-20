from enum import Enum


class Options:
    class Location(Enum):
        SEA = 1
        MOUNTAIN = 2
        CITY = 3

    class Accomodation(Enum):
        HOTEL = 1
        BED_AND_BREAKFAST = 2
        TOURISTIC_VILLAGE = 3
        HOSTEL = 4

    class Food(Enum):
        VEGAN = 1
        VEGETARIAN = 2
        GLUTEN_FREE = 3

    class Activity(Enum):
        OUTDOOR = 1
        HISTORIC_CULTURAL_VISITS = 2
        GASTRONOMY = 3

    class Budget(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        FLEXIBLE = 4

    class Season(Enum):
        AUTUMN_WINTER = 1
        SPRING_SUMMER = 2

    location: Location = None
    accomodation_type: Accomodation = None
    food_type: Food = None
    activity_type: Activity = None
    pets_allowed: bool = None
    budget: Budget = None
    souvenirs_available: bool = None
    season: Season = None

    def __init__(self, location: Location = None, accomodation_type: Accomodation = None, food_type: Food = None,
                 activity_type: Activity = None, pets_allowed: bool = None, budget: Budget = None,
                 souvenirs_available: bool = None, season: Season = None):
        self.location = location
        self.accomodation_type = accomodation_type
        self.food_type = food_type
        self.activity_type = activity_type
        self.pets_allowed = pets_allowed
        self.budget = Options.Budget(budget)
        self.souvenirs_available = souvenirs_available
        self.season = season

    @classmethod
    def calculate_budget(cls, price: float):
        if price <= 0:
            raise ValueError("Il budget non può essere minore o uguale a zero")
        if price <= 50:
            return Options.Budget.LOW
        if price <= 150:
            return Options.Budget.MEDIUM
        return Options.Budget.HIGH


class ActivityOptions(Options):
    category: Options.Activity

    def __init__(self, location: Options.Location, activity_type: Options.Activity, food_type: Options.Food | None,
                 pets_allowed: bool, price: float, souvenirs_available: bool, season: Options.Season | None):
        super().__init__(location, None, food_type, activity_type, pets_allowed, self.calculate_budget(price),
                         souvenirs_available, season)
        self.category = activity_type


class AccomodationOptions(Options):
    category: Options.Accomodation

    def __init__(self, location: Options.Location, accomodation_type: Options.Accomodation,
                 food_type: Options.Food | None,
                 pets_allowed: bool, price: float, souvenirs_available: bool, season: Options.Season | None):
        super().__init__(location, accomodation_type, food_type, None, pets_allowed, self.calculate_budget(price),
                         souvenirs_available, season)
        self.category = accomodation_type
