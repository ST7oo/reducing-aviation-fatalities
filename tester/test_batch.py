import time
import re
import googleapiclient.discovery as discovery
from googleapiclient.errors import HttpError


def make_batch_job_body(project_name, input_paths, output_path,
        model_name, region, data_format='TEXT',
        version_name=None, max_worker_count=None,
        runtime_version=None):

    project_id = 'projects/{}'.format(project_name)
    model_id = '{}/models/{}'.format(project_id, model_name)
    if version_name:
        version_id = '{}/versions/{}'.format(model_id, version_name)

    # Make a jobName of the format "model_name_batch_predict_YYYYMMDD_HHMMSS"
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())

    # Make sure the project name is formatted correctly to work as the basis
    # of a valid job name.
    clean_project_name = re.sub(r'\W+', '_', project_name)

    job_id = '{}_{}_{}'.format(clean_project_name, model_name,
                           timestamp)

    # Start building the request dictionary with required information.
    body = {'jobId': job_id,
            'predictionInput': {
                'dataFormat': data_format,
                'inputPaths': input_paths,
                'outputPath': output_path,
                'region': region}}

    # Use the version if present, the model (its default version) if not.
    if version_name:
        body['predictionInput']['versionName'] = version_id
    else:
        body['predictionInput']['modelName'] = model_id

    # Only include a maximum number of workers or a runtime version if specified.
    # Otherwise let the service use its defaults.
    if max_worker_count:
        body['predictionInput']['maxWorkerCount'] = max_worker_count

    if runtime_version:
        body['predictionInput']['runtimeVersion'] = runtime_version

    return body


project_name = 'kaggle-avation-fatalities'
input_files = 'gs://reducing-commercial-aviation-fatalities/dataset/test2.json'
output_path = 'gs://reducing-commercial-aviation-fatalities/predictions'
model_name = 't1'
region = 'europe-west1'

project_id = 'projects/{}'.format(project_name)
batch_predict_body = make_batch_job_body(project_name, input_files, output_path, model_name, region)

ml = discovery.build('ml', 'v1')
request = ml.projects().jobs().create(parent=project_id, body=batch_predict_body)

try:
    response = request.execute()

    print('Job requested.')

    # The state returned will almost always be QUEUED.
    print('state : {}'.format(response['state']))

except HttpError as err:
    # Something went wrong, print out some information.
    print('There was an error getting the prediction results.' +
          'Check the details:')
    print(err._get_reason())