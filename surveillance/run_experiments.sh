#!/usr/bin/env bash
# Reproduces every surveillance experiment (detection, segmentation, generative, sequence) in order.
# Run from the repository root with the training environment:  bash surveillance/run_experiments.sh
# Seeds are fixed inside each script; logs go to surveillance/runs/<experiment>/train.log.
set -u
PY=${PY:-.venv-train/Scripts/python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run() {  # run <name> <command...>: log to surveillance/runs/<name>/train.log, keep going on failure
  local name=$1; shift
  mkdir -p "surveillance/runs/$name"
  echo "=== $(date '+%H:%M') $name: $*"
  "$PY" "$@" > "surveillance/runs/$name/train.log" 2>&1 || echo "!!! $name failed (see its train.log)"
}

# Damaged-tissue segmentation (semantic + instance) and severity
run leafseg_deeplab  -m surveillance.segmentation train --model deeplab --epochs 40
run leafseg_deeplab_eval -m surveillance.segmentation eval --model deeplab
run leafseg_fcn      -m surveillance.segmentation train --model fcn --epochs 40
run leafseg_fcn_eval -m surveillance.segmentation eval --model fcn
run leafseg_maskrcnn -m surveillance.segmentation train --model maskrcnn --epochs 30
run leafseg_maskrcnn_eval -m surveillance.segmentation eval --model maskrcnn

# Pest + disease detection on tea field photos
run tea_retinanet  -m surveillance.detection train --data tea --model retinanet --epochs 16
run tea_retinanet_eval -m surveillance.detection eval --data tea --model retinanet
run tea_fasterrcnn -m surveillance.detection train --data tea --model fasterrcnn --epochs 16
run tea_fasterrcnn_eval -m surveillance.detection eval --data tea --model fasterrcnn
run tea_yolo       -m surveillance.detection yolo --data tea --epochs 50

# Disease detection on PlantDoc (13 crops)
run plantdoc_retinanet  -m surveillance.detection train --data plantdoc --model retinanet --epochs 16
run plantdoc_retinanet_eval -m surveillance.detection eval --data plantdoc --model retinanet
run plantdoc_fasterrcnn -m surveillance.detection train --data plantdoc --model fasterrcnn --epochs 16
run plantdoc_fasterrcnn_eval -m surveillance.detection eval --data plantdoc --model fasterrcnn
run plantdoc_yolo       -m surveillance.detection yolo --data plantdoc --epochs 50

# Generative augmentation (VAE + GAN)
run generative -m surveillance.generative

echo "=== $(date '+%H:%M') all experiments finished"
