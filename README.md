# ReLeASE Optimizing Compiler

This repository contains the open-source contribution for the work:
"Reinforcement Learning and Adaptive Sampling for Optimized DNN Compilation"

We integrate our optimizing compiler into TVM, Open deep learning compiler stack for cpu, gpu and specialized accelerators.

__NOTE__

We make a custom implementation of PPO, making it more portable.
Internal implementation is from another repo, and the idea about the architecture is from TensorForce.
PPO Implementation: https://github.com/uidilr/ppo_tf

__HISTORY__

(5/16) initial commit
(5/29) revision of the original commit
(6/19) PPO code has been removed for cleanup

__References__

PPO Paper: https://arxiv.org/pdf/1707.06347.pdf
TVM: https://tvm.ai
