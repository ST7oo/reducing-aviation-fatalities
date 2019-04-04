set JOB_NAME=keras_batch_prediction_%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set MODEL_NAME=keras1
set JOB_DIR="gs://reducing-commercial-aviation-fatalities/jobs_dir"
set INPUT_PATH="gs://reducing-commercial-aviation-fatalities/dataset/prediction_input.json"

gcloud ml-engine jobs submit prediction %JOB_NAME% ^
    --model %MODEL_NAME% ^
    --data-format TEXT ^
    --region europe-west1 ^
    --input-paths %INPUT_PATH% ^
    --output-path %JOB_DIR%/predictions