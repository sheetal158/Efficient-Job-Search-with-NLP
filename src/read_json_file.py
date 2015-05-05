import json
from pprint import pprint

with open('annotation_train.json') as data_file:    
    data = json.load(data_file)

pprint(data)