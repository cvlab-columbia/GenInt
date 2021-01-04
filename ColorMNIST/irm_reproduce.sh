echo "IRM: "
CUDA_VISIBLE_DEVICES=0,1,2 python -u main-irm.py \
  --l2_regularizer_weight=0.00110794568 \
  --lr=0.00004898536566546834 \
  --penalty_anneal_iters=190 \
  --penalty_weight=91257.18613115903 \
  --steps=501