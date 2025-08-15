#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog, TCN
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Sample
options['sample'] = "session_window_PCA_PPA"
options['window_size'] = -1  # if fix_window

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 20
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 2

# Train
options['batch_size'] = 512
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 100
options['lr_step'] = (30, 35)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "lightlog"
options['save_dir'] = "../result/lightlog/"

# Predict
options['model_path'] = "../result/lightlog/lightlog_bestloss.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def train():
    Model = TCN(input_dim=options['input_size'],
                num_classes=options['num_classes']
                )
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = TCN(input_dim=options['input_size'],
                num_classes=options['num_classes']
                )
    predicter = Predicter(Model, options)
    predicter.predict_lightlog()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
