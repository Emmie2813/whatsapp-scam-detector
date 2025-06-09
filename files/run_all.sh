#!/bin/bash
pip install -r requirements.txt
python extract_and_transcribe.py
python data_preprocessing.py
python train_model.py
python create_wordclouds.py
python app.py