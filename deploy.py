!gcloud beta ai-plateform versions create asp --model stackoverflow_model\
--origin=gs://stackoverflow_model/ \
--python-version=3.5 \
--runtime-version=1.13 \
--framework='TENSORFLOW' \
--config=config.yaml \
--package-uris=gs://stackoverflow_model/packages/stackoverflow_predict-1.0.tar.gz \
--prediction-class=prediction/predictionclass.prediction
