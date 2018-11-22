clear all
clc
load IRLStest.mat

Gamma = cluster_weights
Tau = tauijk
M = phiW
Winit = Wg_init

res = IRLS_MixFRHLP(Gamma,tauijk, phiW, Wg_init, verbose_IRLS);