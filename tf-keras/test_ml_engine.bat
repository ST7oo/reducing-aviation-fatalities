set MODEL_NAME="keras1"

gcloud ml-engine predict ^
    --model %MODEL_NAME% ^
    --json-instances prediction_input.json
