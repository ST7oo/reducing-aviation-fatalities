if exist local-train-output rd /q /s local-train-output

gcloud ai-platform local train ^
    --package-path trainer ^
    --module-name trainer.task ^
    --job-dir local-train-output