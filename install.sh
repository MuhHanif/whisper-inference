#!/bin/bash

virtualenv -p python3 whisper_env
whisper_env/bin/python3 -m pip install -r requirements.txt