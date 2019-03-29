:: Error: can't find python

set MODEL_DIR="gs://reducing-commercial-aviation-fatalities/rf_20190326_160008/"
set INPUT_FILE="tester\input.json"
set FRAMEWORK="SCIKIT_LEARN"

gcloud ml-engine local predict --model-dir=%MODEL_DIR% ^
    --json-instances %INPUT_FILE% ^
    --framework %FRAMEWORK%