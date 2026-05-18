from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["predictions_db"]

predictions_collection = db["predictions"]