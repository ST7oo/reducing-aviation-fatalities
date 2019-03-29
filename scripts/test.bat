set MODEL_NAME="t1"
set VERSION_NAME="t1_1"
set INPUT_FILE="tester\input.json"

gcloud ml-engine predict --model %MODEL_NAME% ^
    --version %VERSION_NAME% ^
    --json-instances %INPUT_FILE%