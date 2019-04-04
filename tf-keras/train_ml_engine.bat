set JOB_NAME=keras_job1
set JOB_DIR="gs://reducing-commercial-aviation-fatalities/jobs_dir"

gcloud ml-engine jobs submit training %JOB_NAME% ^
    --package-path trainer ^
    --module-name trainer.task ^
    --region europe-west1 ^
    --python-version 3.5 ^
    --runtime-version 1.13 ^
    --job-dir %JOB_DIR%