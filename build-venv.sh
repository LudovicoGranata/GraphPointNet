#!/bin/bash

python -m venv .GPN

.GPN/bin/python3 -m pip install -U pip
.GPN/bin/pip install -U wheel setuptools
.GPN/bin/pip install -r ./requirements.txt