set JOB_NAME=keras_training_%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set JOB_DIR="gs://reducing-commercial-aviation-fatalities/jobs_dir"
set INPUT_DIR="https://storage.googleapis.com/reducing-commercial-aviation-fatalities/dataset"

gcloud ai-platform jobs submit training %JOB_NAME% ^
    --package-path trainer ^
    --module-name trainer.task ^
    --region europe-west1 ^
    --python-version 3.5 ^
    --runtime-version 1.13 ^
    --job-dir %JOB_DIR% ^
    --stream-logs ^
    -- ^
    --input-dir %INPUT_DIR% ^
    --train-file train2.csv ^
    --num-epochs 20