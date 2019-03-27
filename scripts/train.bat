set JOB_NAME=t1_%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set JOB_DIR=gs://reducing-commercial-aviation-fatalities/jobs_dir
set TRAINING_PACKAGE_PATH=trainer
set MAIN_TRAINER_MODULE=trainer.t1
set REGION=europe-west1
set RUNTIME_VERSION=1.10
set PYTHON_VERSION=3.5
set SCALE_TIER=BASIC

gcloud ml-engine jobs submit training %JOB_NAME% \
  --job-dir %JOB_DIR% \
  --package-path %TRAINING_PACKAGE_PATH% \
  --module-name %MAIN_TRAINER_MODULE% \
  --region %REGION% \
  --runtime-version=%RUNTIME_VERSION% \
  --python-version=%PYTHON_VERSION% \
  --scale-tier %SCALE_TIER%