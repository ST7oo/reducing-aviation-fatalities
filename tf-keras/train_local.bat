gcloud ml-engine local train ^
    --package-path trainer ^
    --module-name trainer.task ^
    --job-dir local-train-output