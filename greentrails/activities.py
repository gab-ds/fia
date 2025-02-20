from geopy import distance

from greentrails.options import ActivityOptions, AccomodationOptions


class Activity:
    id: int
    is_accomodation: bool
    latitude: float
    longitude: float
    options: ActivityOptions | AccomodationOptions

    def __init__(self, activity_id: int, is_accomodation: bool, latitude: float, longitude: float,
                 options: ActivityOptions | AccomodationOptions):
        if is_accomodation is None or latitude is None or longitude is None or options is None:
            raise ValueError('Parametri non validi')

        if is_accomodation and not isinstance(options, AccomodationOptions) \
                or not is_accomodation and not isinstance(options, ActivityOptions):
            raise ValueError('Opzioni non corrispondenti al tipo di attività')

        self.id = activity_id
        self.is_accomodation = is_accomodation
        self.latitude = latitude
        self.longitude = longitude
        self.options = options

    def distance_from_coords(self, other_latitude: float, other_longitude: float) -> float:
        return distance.geodesic((self.latitude, self.longitude), (other_latitude, other_longitude)).meters

    def distance_from(self, other: "Activity") -> float:
        return self.distance_from_coords(other.latitude, other.longitude)
