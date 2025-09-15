import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

class MongoDBHelper:
    def __init__(self, max_retries: int = 2):
        MONGODB_URI = os.getenv("MONGODB_URI")
        retries = 0
        while retries < max_retries:
            try:
                self.client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
                self.check_database_connectivity()
                break
            except (ConfigurationError, ServerSelectionTimeoutError) as e:
                print("Failed to connect to MongoDB cluster, retrying again")
                retries += 1
                print(f"Tries left: {max_retries-retries}")
        else: 
            raise RuntimeError(f"Failed to connect to MongoDB cluster after {max_retries} attempts.")
        


    def check_database_connectivity(self):
        if self.client:             # checks if the mongodb uri was extracted from .env file
            try:
                self.client.admin.command('ping')
                return True
            except Exception as e:
                print("Failed to connect to MongoDB due to the following error: {error}".format(error = e))
                return False
        else:
            print("Unable to extract any MongoDB URI from .env file.")
            return False
        
if __name__ == "__main__":
    db = MongoDBHelper()
    if db.check_database_connectivity():
        print("MongoDB connected")
    base_qns_db = db.client["Base_Questions_DB"]
    question_database = base_qns_db["HumanEval_Open_Ended"]
