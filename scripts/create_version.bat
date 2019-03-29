set MODEL_DIR="gs://reducing-commercial-aviation-fatalities/rf_20190326_160008/"
set VERSION_NAME="t1_2"
set MODEL_NAME="t1"
set FRAMEWORK="SCIKIT_LEARN"

gcloud ml-engine versions create $VERSION_NAME ^
      --model $MODEL_NAME ^
      --origin $MODEL_DIR ^
      --runtime-version=1.12 ^
      --framework $FRAMEWORK ^
      --python-version=3.5