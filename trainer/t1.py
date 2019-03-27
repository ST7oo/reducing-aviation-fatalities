import pandas as pd
import os
import datetime
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

BUCKET_ID = 'reducing-commercial-aviation-fatalities'
train_filename = 'train.csv'
data_dir = 'gs://' + BUCKET_ID + '/dataset'

subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir, train_filename), train_filename], stderr=sys.stdout)

train_file = pd.read_csv(train_filename)
# train_file = pd.read_csv('../reducing-commercial-aviation-fatalities/train.csv')

labels = train_file.event
features = train_file.drop(['event','crew','experiment','seat'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

random_forest = RandomForestClassifier(n_estimators=10, n_jobs=-1)
random_forest.fit(train_features, train_labels)

model_filename = 'model.joblib'
joblib.dump(random_forest, model_filename)

model_path = os.path.join('gs://', BUCKET_ID, datetime.datetime.now().strftime('rf_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, model_path], stderr=sys.stdout)
