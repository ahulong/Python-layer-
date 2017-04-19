#!/usr/bin/env sh

model_dir='models/ClsRegNet/YSBox'

./build/tools/caffe train --solver $model_dir/solver.prototxt -gpu $1 1>$model_dir/batchlog0 2>$model_dir/log0 &
#./build/tools/caffe train --weights=$model_dir/vgg16.caffemodel --solver $model_dir/solver.prototxt  -gpu $1 2>&1 | tee $model_dir/log0 &
#./build/tools/caffe train   --solver $model_dir/solver.prototxt -snapshot $model_dir/8layer_morenight_iter_20000.solverstate -gpu $1 1>$model_dir/batchlog1 2>$model_dir/log1 &
#fine-tunning
#./build/tools/caffe train  --weights=$model_dir/vgg16.caffemodel --solver $model_dir/solver.prototxt -snapshot $model_dir/8layer_moredata_iter_10000.solverstate gpu $1 1>$model_dir/batchlog1 2>$model_dir/log1 &

#./build/tools/caffe train -solver $model_dir/solver.prototxt -weights $model_dir/8layer_morenight_iter_10000.caffemodel -gpu $1 2>&1 | tee $model_dir/log1 &
