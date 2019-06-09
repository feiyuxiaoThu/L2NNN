# train mnist resnet
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train --config mnist.yaml --model preact_resnet20
# CUDA_VISIBLE_DEVICES=1 python main.py --phase train --config mnist.yaml --model preact_resnet20 --MMLDA
# CUDA_VISIBLE_DEVICES=2 python main.py --phase train --config mnist.yaml --model preact_resnet20 --MMLDA --mean 1 --var 0.01

# train mnist resnet reg_data
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train --config mnist.yaml --model preact_resnet20 --reg_coef 0.0001 &
# CUDA_VISIBLE_DEVICES=1 python main.py --phase train --config mnist.yaml --model preact_resnet20 --reg_coef 0.001 &
# CUDA_VISIBLE_DEVICES=2 python main.py --phase train --config mnist.yaml --model preact_resnet20 --reg_coef 0.01 &
# CUDA_VISIBLE_DEVICES=3 python main.py --phase train --config mnist.yaml --model preact_resnet20 --reg_coef 0.1 &

# train mnist ode_resnet reg_data
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train --config mnist.yaml --model ode_resnet20 --reg_coef 0.0001 &
# CUDA_VISIBLE_DEVICES=1 python main.py --phase train --config mnist.yaml --model ode_resnet20 --reg_coef 0.001 &
# CUDA_VISIBLE_DEVICES=2 python main.py --phase train --config mnist.yaml --model ode_resnet20 --reg_coef 0.01 &
# CUDA_VISIBLE_DEVICES=3 python main.py --phase train --config mnist.yaml --model ode_resnet20 --reg_coef 0.1 &

# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0 --div_before_conv False &
# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0.1 --div_before_conv False &
# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0.5 --div_before_conv False &
# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0 --div_before_conv True &
# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0.1 --div_before_conv True &
# python main.py --phase train --config mnist.yaml --model L2NonExpaConvNet --w 0.5 --div_before_conv True &

benchmark() {
    nat $1 
    FGSM $1
    PGD $1 
}
nat() {
    python main.py \
        --phase test \
        --resume $1 \
        --resume_opt best \
}
FGSM() {
    TF_CPP_MIN_LOG_LEVEL=3 python main.py \
        --phase test \
        --resume $1 \
        --resume_opt best \
        --attack FGSM \
        --attack_params '{"eps":0.3, "clip_min":0.0, "clip_max":1.0}'
}
PGD() {
    TF_CPP_MIN_LOG_LEVEL=3 python main.py \
        --phase test \
        --resume $1 \
        --resume_opt best \
        --attack PGD \
        --attack_params '{"eps":0.3, "eps_iter":0.01, "nb_iter":40, "clip_min":0.0, "clip_max":1.0}'
}