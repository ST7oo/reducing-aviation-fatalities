set MODEL_VERSION="v1"
set MODEL_NAME="t2"
set SAVED_MODEL_PATH="gs://reducing-commercial-aviation-fatalities/jobs_dir/keras_export/1556809260/"

gcloud ai-platform versions create %MODEL_VERSION% ^
      --model %MODEL_NAME% ^
      --origin %SAVED_MODEL_PATH% ^
      --runtime-version 1.13 ^
      --framework tensorflow ^
      --python-version 3.5