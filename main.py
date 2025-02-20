import json

from fastapi import FastAPI, Request
from pydantic import BaseModel

from greentrails.activities import Activity
from greentrails.ga import GeneticAlgorithm
from greentrails.options import AccomodationOptions, ActivityOptions, Options

app = FastAPI()


class Item(BaseModel):
    preferences: dict
    activities: list


@app.post("/")
async def main(request: Request):
    data = await request.json()
    preferences = Options(**data['preferences'])
    activities_dict = data['activities']
    activities = []
    for activity_dict in activities_dict:
        location = activity_dict['location']
        category = activity_dict['category']
        pets_allowed = activity_dict['pets_allowed']
        price = activity_dict['price']
        souvenirs_available = activity_dict['souvenirs_available']
        food = activity_dict['food_type'] if 'food_type' in activity_dict else None
        season = activity_dict['season'] if 'season' in activity_dict else None
        if activity_dict['is_accomodation']:
            options = AccomodationOptions(location, category, food, pets_allowed, price, souvenirs_available, season)
        else:
            options = ActivityOptions(location, category, food, pets_allowed, price, souvenirs_available, season)
        activity = Activity(activity_dict['id'], activity_dict['is_accomodation'], activity_dict['lat'],
                            activity_dict['lon'], options)
        activities.append(activity)
    genetic = GeneticAlgorithm(activities=activities, preferences=preferences)
    best_individual, (best_generation, current_generation) = genetic.execute_ga()
    return {"best_individual": best_individual, "best_generation": best_generation,
            "current_generation": current_generation}
