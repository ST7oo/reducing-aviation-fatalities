import googleapiclient.discovery
import json
import ast

PROJECT = 'kaggle-avation-fatalities'
MODEL = 't2'

def predict_json(project, model, instances, version=None):
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    
    return response['predictions']


with open('../tester/input.json') as input_file:
    content = input_file.readlines()

instances = [ast.literal_eval(x.strip()) for x in content]
# print(instances)
print(predict_json(PROJECT, MODEL, instances))