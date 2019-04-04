echo "Training local ML model"

set MODEL_NAME=tf1

set PACKAGE_PATH=trainer
set TRAIN_FILES=data/train-data-*.csv
set VALID_FILES=data/eval-data-*.csv
set MODEL_DIR=trained_models/%MODEL_NAME%


REM gcloud ml-engine local train ^
REM         --module-name=trainer.task ^
REM         --package-path=%PACKAGE_PATH% ^
REM         -- ^
python -m trainer.task ^
        --train-files=%TRAIN_FILES% ^
        --num-epochs=10 ^
        --train-batch-size=500 ^
        --eval-files=%VALID_FILES% ^
        --eval-batch-size=500 ^
        --learning-rate=0.001 ^
        --hidden-units="128,40,40" ^
        --layer-sizes-scale-factor=0.5 ^
        --num-layers=3 ^
        --job-dir=%MODEL_DIR% 
        REM --remove-model-dir=True


REM ls %MODEL_DIR%/export/estimator
REM MODEL_LOCATION=%MODEL_DIR%/export/estimator/$(ls %MODEL_DIR%/export/estimator | tail -1)
REM echo %MODEL_LOCATION%
REM ls %MODEL_LOCATION%

REM # invoke trained model to make prediction given new data instances
REM gcloud ml-engine local predict --model-dir=%MODEL_LOCATION% --json-instances=data/new-data.json