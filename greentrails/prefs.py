from enum import Enum

class Preferences:
    class Location(Enum):
        SEA = 1
        MOUNTAIN = 2
        CITY = 3
        NO_PREFERENCE = 4

    class Accomodation(Enum):
        HOTEL = 1
        BED_AND_BREAKFAST = 2
        TOURISTIC_VILLAGE = 3
        HOSTEL = 4
        NO_PREFERENCE = 5

    class Food(Enum):
        VEGAN = 1
        VEGETARIAN = 2
        GLUTEN_FREE = 3
        NO_PREFERENCE = 4

    class Activity(Enum):
        OUTDOOR = 1
        HISTORIC_CULTURAL_VISITS = 2
        GASTRONOMY = 3
        NO_PREFERENCE = 4

    class Budget(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        FLEXIBLE = 4

    class Season(Enum):
        AUTUMN_WINTER = 1
        SPRING_SUMMER = 2
        NO_PREFERENCE = 3

    favourite_location: Location = Location.NO_PREFERENCE
    favourite_accomodation_type: Accomodation = Accomodation.NO_PREFERENCE
    favourite_food_type: Food = Food.NO_PREFERENCE
    favourite_activity_type: Activity = Activity.NO_PREFERENCE
    pets_allowed: bool
    favourite_budget: Budget = Budget.FLEXIBLE
    local_souvenirs_available: bool
    favourite_season: Season = Season.NO_PREFERENCE

    def __init__(self, favourite_location: Location, favourite_accomodation_type: Accomodation, favourite_food_type: Food,
                 favourite_activity_type: Activity, pets_allowed: bool, favourite_budget: Budget,
                 local_souvenirs_available: bool, favourite_season: Season):
        self.favourite_location = favourite_location
        self.favourite_accomodation_type = favourite_accomodation_type
        self.favourite_food_type = favourite_food_type
        self.favourite_activity_type = favourite_activity_type
        self.pets_allowed = pets_allowed
        self.favourite_budget = favourite_budget
        self.local_souvenirs_available = local_souvenirs_available
        self.favourite_season = favourite_season
