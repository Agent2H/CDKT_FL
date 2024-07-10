# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:19:14 2017

@author: Minh
"""
import os

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Table_index1 = 40
Table_index2 = 50
Table_index3 = 190
Table_index4 = 200
Start_index = 80
def plot_final():
    # df_iters = read_files()  # 3 algorithms
    # plot_box(df_iters,1)
    # plt.savefig("figs/mnist_fixed_users_boxplot.pdf")
    # df_iters = read_files1()
    # plot_box(df_iters,2)
    # plt.savefig("figs/mnist_subset_users_boxplot.pdf")
    # df_iters = read_files2()
    # plot_box(df_iters,3)
    # plt.savefig("figs/fmnist_fixed_users_boxplot_EAAI.pdf")
    df_iters = read_files3()
    # plot_box(df_iters,4)
    # plt.savefig("figs/fmnist_subset_users_boxplot_EAAI.pdf")
    # df_iters = read_files4()
    # plot_box(df_iters, 5)
    # plt.savefig("figs/cifar10_fixed_users_boxplot_EAAI.pdf")
    df_iters = read_files5()
    # plot_box(df_iters, 6)
    # plt.savefig("figs/cifar10_subset_users_boxplot_EAAI.pdf")
    # df_iters = read_files6()
    # plot_box(df_iters, 7)
    # plt.savefig("figs/cifar100_fixed_users_boxplot_EAAI.pdf")
    df_iters = read_files7()
    # plot_box(df_iters, 8)
    # plt.savefig("figs/cifar100_subset_users_boxplot_EAAI.pdf")
    # df_iters = read_files8()
    # plot_box(df_iters, 9)
    # plt.savefig("figs/ICOIN_boxplot.pdf")

    # df_iters = read_files2_f1()
    # df_iters = read_files3_f1()
    # df_iters = read_files4_f1()
    # df_iters = read_files5_f1()
    # df_iters = read_files6_f1()
    # df_iters = read_files7_f1()
    # df_iters= scalability()
    # df_iters=scalability_f1()

    # plt.ylim(0, 100)
    # plt.savefig("figs/mnist_fixed_users_boxplot.png")
    # plt.savefig("figs/mnist_subset_users_boxplot.png")
    # plt.savefig("figs/fmnist_fixed_users_boxplot.png")
    # plt.savefig("figs/fmnist_subset_users_boxplot.png")
    plt.show()
def plot_box(df_iters,figure_index):


    # plt.figure(2, figsize=(7., 5.1))
    # plt.figure(2, figsize=(8.7, 5.8))
    plt.figure(figure_index, figsize=(35, 20))
    # plt.figure(2,figsize=(4.5,5))
    # sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=3.3)
    sns.swarmplot(x="Algorithm", y="Accuracy", data=df_iters)
    plt.tight_layout(pad=3, w_pad=3, h_pad=3)
    sns.boxplot(x="Algorithm", y="Accuracy", data=df_iters, showcaps=True, boxprops={'facecolor': 'None'},
                showfliers=False, whiskerprops={'linewidth': 2}, linewidth=2)
    plt.xlabel('Algorithm', fontsize=39)
    plt.ylabel('Testing Accuracy', fontsize=39)
    # plt.ylim(0, 100)
    # plt.savefig("figs/mnist_fixed_users_boxplot.png")
    # plt.savefig("figs/mnist_subset_users_boxplot.png")
    # plt.savefig("figs/fmnist_fixed_users_boxplot.png")
    # plt.savefig("figs/fmnist_subset_users_boxplot.png")


def read_files():

    filename = 'results_fig'+ 'CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5'
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0.06_b0.6_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0.07_b0.5_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fTrue_a0.12_b0.6_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_mnist_I100_sTrue_fFalse_a0.12_b0.6_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    #
    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]


    print ("--------------- FIXED MNIST RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1)+np.median(gen1))/2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis,:],glob2[np.newaxis,:],glob3[np.newaxis,:],glob4[np.newaxis,:],gen1[np.newaxis,:],gen2[np.newaxis,:],gen3[np.newaxis,:],gen4[np.newaxis,:]), axis = 0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters


def read_files1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0.25_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0.25_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fTrue_a0.25_b0.1_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_mnist_I100_sTrue_fFalse_a0.06_b0.9_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_mnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- SUBSET MNIST RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                           gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files2():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'MOON_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'FedDyn_fmnist_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]


    print("--------------- FIXED Fashion-MNIST RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],glob6[np.newaxis, :],
                           glob7[np.newaxis, :],glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
                           gen4[np.newaxis, :],gen6[np.newaxis, :],gen7[np.newaxis, :],gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal','Scaffold\nGlobal',
    #               'MOON\nGlobal','FedDyn\nGlobal','(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen',
    #               'Scaffold\nC-Gen','MOON\nC-Gen','FedDyn\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files3():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),'r')
    df7 = h5py.File(os.path.join(directory,'MOON_fmnist_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSTrue_accFalse_gmJSD_lmJSD_eaaiTrue.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'FedDyn_fmnist_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- SUBSET Fashion-MNIST RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],glob6[np.newaxis, :],
                           glob7[np.newaxis, :],glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
                           gen4[np.newaxis, :],gen6[np.newaxis, :],gen7[np.newaxis, :],gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files4():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),'r')
    df7 = h5py.File(os.path.join(directory,'MOON_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedDyn_Cifar10_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- FIXED CIFAR-10 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],glob6[np.newaxis, :],
                           glob7[np.newaxis, :],glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
                           gen4[np.newaxis, :],gen6[np.newaxis, :],gen7[np.newaxis, :],gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters


def read_files5():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.75_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar10_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSTrue_accFalse_gmJSD_lmJSD_eaaiTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- SUBSET CIFAR-10 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    data = np.concatenate(
        (glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :], glob6[np.newaxis, :],
         glob7[np.newaxis, :], glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
         gen4[np.newaxis, :], gen6[np.newaxis, :], gen7[np.newaxis, :], gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files6():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.5_b0.2_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar100_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- FIXED CIFAR-100 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)

    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    data = np.concatenate(
        (glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :], glob6[np.newaxis, :],
         glob7[np.newaxis, :], glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
         gen4[np.newaxis, :], gen6[np.newaxis, :], gen7[np.newaxis, :], gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters


def read_files7():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar100_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]
    print("--------------- SUBSET CIFAR-100 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)

    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    data = np.concatenate(
        (glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :], glob6[np.newaxis, :],
         glob7[np.newaxis, :], glob8[np.newaxis, :], gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :],
         gen4[np.newaxis, :], gen6[np.newaxis, :], gen7[np.newaxis, :], gen8[np.newaxis, :]), axis=0)
    # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    iters_cols = ['Rep\nGlobal', 'RepFull\nGlobal', 'Full\nGlobal', 'FedAvg\nGlobal',
                  'Scaffold\nGlobal',
                  'MOON\nGlobal', 'FedDyn\nGlobal', 'Rep\nC-Gen', 'RepFull\nC-Gen', 'Full\nC-Gen',
                  'FedAvg\nC-Gen',
                  'Scaffold\nC-Gen', 'MOON\nC-Gen', 'FedDyn\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files2_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.03_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'MOON_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'FedDyn_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]


    print("--------------- FIXED Fashion-MNIST F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)

    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def read_files3_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),'r')
    df7 = h5py.File(os.path.join(directory,'MOON_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'FedDyn_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]

    print("--------------- SUBSET Fashion-MNIST F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def read_files4_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'Scaffold_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),'r')
    df7 = h5py.File(os.path.join(directory,'MOON_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedDyn_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]

    print("--------------- FIXED CIFAR-10 F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters


def read_files5_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.75_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar10_I100_sTrue_fFalse_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]

    print("--------------- SUBSET CIFAR-10 F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def read_files6_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]

    print("--------------- FIXED CIFAR-100 F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)


    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters


def read_files7_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'MOON_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'FedDyn_Cifar100_I100_sTrue_fTrue_a0.08_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]
    print("--------------- SUBSET CIFAR-100 F1-SCORE RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("fedavg glob:", np.median(glob4))
    print("fedavg gen:", np.median(gen4))
    print("fedavg spec:", np.median(spec4))
    print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("scaffold glob:", np.median(glob6))
    print("scaffold gen:", np.median(gen6))
    print("scaffold spec:", np.median(spec6))
    print("scaffold personalized:", (np.median(spec6) + np.median(gen6)) / 2)

    print("MOON glob:", np.median(glob7))
    print("MOON gen:", np.median(gen7))
    print("MOON spec:", np.median(spec7))
    print("MOON personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("FedDyn glob:", np.median(glob8))
    print("FedDyn gen:", np.median(gen8))
    print("FedDyn spec:", np.median(spec8))
    print("FedDyn personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT no transfer glob:", np.median(glob5))
    print("CDKT no transfer gen:", np.median(gen5))
    print("CDKT no transfer spec:", np.median(spec5))
    print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)


    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def scalability_f1():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Rep_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_RepFull_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Full_fmnist_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'CDKT_Rep_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_RepFull_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar10_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df9 = h5py.File(os.path.join(directory,
                                'CDKT_Full_Cifar100_srate0.5_numclient50_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                   'r')
    df10 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')

    df11 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df12 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_fmnist_srate0.2_numclient100_I200_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df13 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df14 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df15 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar10_srate0.2_numclient100_I200_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df16 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df17 = h5py.File(os.path.join(directory,
                                'CDKT_RepFull_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                   'r')
    df18 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar100_srate0.2_numclient100_I200_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')


    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['global_f1'][Table_index1:Table_index2]
    gen1 = df['gen_f1'][Table_index1:Table_index2]
    spec1 = df['spec_f1'][Table_index1:Table_index2]
    glob2 = df2['global_f1'][Table_index1:Table_index2]
    gen2 = df2['gen_f1'][Table_index1:Table_index2]
    spec2 = df2['spec_f1'][Table_index1:Table_index2]
    glob3 = df3['global_f1'][Table_index1:Table_index2]
    gen3 = df3['gen_f1'][Table_index1:Table_index2]
    spec3 = df3['spec_f1'][Table_index1:Table_index2]
    glob4 = df4['global_f1'][Table_index1:Table_index2]
    gen4 = df4['gen_f1'][Table_index1:Table_index2]
    spec4 = df4['spec_f1'][Table_index1:Table_index2]
    glob5 = df5['global_f1'][Table_index1:Table_index2]
    gen5 = df5['gen_f1'][Table_index1:Table_index2]
    spec5 = df5['spec_f1'][Table_index1:Table_index2]
    glob6 = df6['global_f1'][Table_index1:Table_index2]
    gen6 = df6['gen_f1'][Table_index1:Table_index2]
    spec6 = df6['spec_f1'][Table_index1:Table_index2]
    glob7 = df7['global_f1'][Table_index1:Table_index2]
    gen7 = df7['gen_f1'][Table_index1:Table_index2]
    spec7 = df7['spec_f1'][Table_index1:Table_index2]
    glob8 = df8['global_f1'][Table_index1:Table_index2]
    gen8 = df8['gen_f1'][Table_index1:Table_index2]
    spec8 = df8['spec_f1'][Table_index1:Table_index2]
    glob9 = df9['global_f1'][Table_index1:Table_index2]
    gen9 = df9['gen_f1'][Table_index1:Table_index2]
    spec9 = df9['spec_f1'][Table_index1:Table_index2]
    glob10 = df10['global_f1'][Table_index1:Table_index2]
    gen10 = df10['gen_f1'][Table_index1:Table_index2]
    spec10 = df10['spec_f1'][Table_index1:Table_index2]
    glob11 = df11['global_f1'][Table_index1:Table_index2]
    gen11 = df11['gen_f1'][Table_index1:Table_index2]
    spec11 = df11['spec_f1'][Table_index1:Table_index2]
    glob12 = df12['global_f1'][Table_index1:Table_index2]
    gen12 = df12['gen_f1'][Table_index1:Table_index2]
    spec12 = df12['spec_f1'][Table_index1:Table_index2]
    glob13 = df13['global_f1'][Table_index1:Table_index2]
    gen13 = df13['gen_f1'][Table_index1:Table_index2]
    spec13 = df13['spec_f1'][Table_index1:Table_index2]
    glob14 = df14['global_f1'][Table_index1:Table_index2]
    gen14 = df14['gen_f1'][Table_index1:Table_index2]
    spec14 = df14['spec_f1'][Table_index1:Table_index2]
    glob15 = df15['global_f1'][Table_index1:Table_index2]
    gen15 = df15['gen_f1'][Table_index1:Table_index2]
    spec15 = df15['spec_f1'][Table_index1:Table_index2]
    glob16 = df16['global_f1'][Table_index1:Table_index2]
    gen16 = df16['gen_f1'][Table_index1:Table_index2]
    spec16 = df16['spec_f1'][Table_index1:Table_index2]
    glob17 = df17['global_f1'][Table_index1:Table_index2]
    gen17 = df17['gen_f1'][Table_index1:Table_index2]
    spec17 = df17['spec_f1'][Table_index1:Table_index2]
    glob18 = df18['global_f1'][Table_index1:Table_index2]
    gen18 = df18['gen_f1'][Table_index1:Table_index2]
    spec18 = df18['spec_f1'][Table_index1:Table_index2]
    print("--------------- SUBSET FashionMNIST F1-SCORE 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("--------------- SUBSET CIFAR-10 F1-SCORE 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob4))
    print("CDKT Rep KL-N gen:", np.median(gen4))
    print("CDKT Rep KL-N spec:", np.median(spec4))
    print("CDKT Rep KL-N personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob5))
    print("CDKT Rep Full KL-N gen:", np.median(gen5))
    print("CDKT Rep Full KL-N spec:", np.median(spec5))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec5) + np.median(gen5)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob6))
    print("CDKT Full KL-N gen:", np.median(gen6))
    print("CDKT Full KL-N spec:", np.median(spec6))
    print("CDKT Full KL-N personalized:", (np.median(spec6) + np.median(gen6)) / 2)
    print("--------------- SUBSET CIFAR-100 F1-SCORE 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob7))
    print("CDKT Rep KL-N gen:", np.median(gen7))
    print("CDKT Rep KL-N spec:", np.median(spec7))
    print("CDKT Rep KL-N personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob8))
    print("CDKT Rep Full KL-N gen:", np.median(gen8))
    print("CDKT Rep Full KL-N spec:", np.median(spec8))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob9))
    print("CDKT Full KL-N gen:", np.median(gen9))
    print("CDKT Full KL-N spec:", np.median(spec9))
    print("CDKT Full KL-N personalized:", (np.median(spec9) + np.median(gen9)) / 2)

    print("--------------- SUBSET FashionMNIST F1-SCORE 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob10))
    print("CDKT Rep KL-N gen:", np.median(gen10))
    print("CDKT Rep KL-N spec:", np.median(spec10))
    print("CDKT Rep KL-N personalized:", (np.median(spec10) + np.median(gen10)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob11))
    print("CDKT Rep Full KL-N gen:", np.median(gen11))
    print("CDKT Rep Full KL-N spec:", np.median(spec11))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec11) + np.median(gen11)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob12))
    print("CDKT Full KL-N gen:", np.median(gen12))
    print("CDKT Full KL-N spec:", np.median(spec12))
    print("CDKT Full KL-N personalized:", (np.median(spec12) + np.median(gen12)) / 2)

    print("--------------- SUBSET CIFAR-10 F1-SCORE 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob13))
    print("CDKT Rep KL-N gen:", np.median(gen13))
    print("CDKT Rep KL-N spec:", np.median(spec13))
    print("CDKT Rep KL-N personalized:", (np.median(spec13) + np.median(gen13)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob14))
    print("CDKT Rep Full KL-N gen:", np.median(gen14))
    print("CDKT Rep Full KL-N spec:", np.median(spec14))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec14) + np.median(gen14)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob15))
    print("CDKT Full KL-N gen:", np.median(gen15))
    print("CDKT Full KL-N spec:", np.median(spec15))
    print("CDKT Full KL-N personalized:", (np.median(spec15) + np.median(gen15)) / 2)
    print("--------------- SUBSET CIFAR-100 F1-SCORE 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob16))
    print("CDKT Rep KL-N gen:", np.median(gen16))
    print("CDKT Rep KL-N spec:", np.median(spec16))
    print("CDKT Rep KL-N personalized:", (np.median(spec16) + np.median(gen16)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob17))
    print("CDKT Rep Full KL-N gen:", np.median(gen17))
    print("CDKT Rep Full KL-N spec:", np.median(spec17))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec17) + np.median(gen17)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob18))
    print("CDKT Full KL-N gen:", np.median(gen18))
    print("CDKT Full KL-N spec:", np.median(spec18))
    print("CDKT Full KL-N personalized:", (np.median(spec18) + np.median(gen18)) / 2)




    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)


    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def scalability():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Rep_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_RepFull_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Full_fmnist_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'CDKT_Rep_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_RepFull_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar10_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df7 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df8 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df9 = h5py.File(os.path.join(directory,
                                'CDKT_Full_Cifar100_srate0.5_numclient50_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                   'r')
    df10 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')

    df11 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df12 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_fmnist_srate0.2_numclient100_I200_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df13 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df14 = h5py.File(os.path.join(directory,
                                 'CDKT_RepFull_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df15 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar10_srate0.2_numclient100_I200_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df16 = h5py.File(os.path.join(directory,
                                 'CDKT_Rep_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')
    df17 = h5py.File(os.path.join(directory,
                                'CDKT_RepFull_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                   'r')
    df18 = h5py.File(os.path.join(directory,
                                 'CDKT_Full_Cifar100_srate0.2_numclient100_I200_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5'),
                    'r')


    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['root_test'][Table_index1:Table_index2]
    gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
    spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
    spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
    spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    gen6 = df6['cg_avg_data_test'][Table_index1:Table_index2]
    spec6 = df6['cs_avg_data_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    gen7 = df7['cg_avg_data_test'][Table_index1:Table_index2]
    spec7 = df7['cs_avg_data_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    gen8 = df8['cg_avg_data_test'][Table_index1:Table_index2]
    spec8 = df8['cs_avg_data_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    gen9 = df9['cg_avg_data_test'][Table_index1:Table_index2]
    spec9 = df9['cs_avg_data_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index3:Table_index4]
    gen10 = df10['cg_avg_data_test'][Table_index3:Table_index4]
    spec10 = df10['cs_avg_data_test'][Table_index3:Table_index4]
    glob11 = df11['root_test'][Table_index3:Table_index4]
    gen11 = df11['cg_avg_data_test'][Table_index3:Table_index4]
    spec11 = df11['cs_avg_data_test'][Table_index3:Table_index4]
    glob12 = df12['root_test'][Table_index3:Table_index4]
    gen12 = df12['cg_avg_data_test'][Table_index3:Table_index4]
    spec12 = df12['cs_avg_data_test'][Table_index3:Table_index4]
    glob13 = df13['root_test'][Table_index3:Table_index4]
    gen13 = df13['cg_avg_data_test'][Table_index3:Table_index4]
    spec13 = df13['cs_avg_data_test'][Table_index3:Table_index4]
    glob14 = df14['root_test'][Table_index3:Table_index4]
    gen14 = df14['cg_avg_data_test'][Table_index3:Table_index4]
    spec14 = df14['cs_avg_data_test'][Table_index3:Table_index4]
    glob15 = df15['root_test'][Table_index3:Table_index4]
    gen15 = df15['cg_avg_data_test'][Table_index3:Table_index4]
    spec15 = df15['cs_avg_data_test'][Table_index3:Table_index4]
    glob16 = df16['root_test'][Table_index3:Table_index4]
    gen16 = df16['cg_avg_data_test'][Table_index3:Table_index4]
    spec16 = df16['cs_avg_data_test'][Table_index3:Table_index4]
    glob17 = df17['root_test'][Table_index3:Table_index4]
    gen17 = df17['cg_avg_data_test'][Table_index3:Table_index4]
    spec17 = df17['cs_avg_data_test'][Table_index3:Table_index4]
    glob18 = df18['root_test'][Table_index3:Table_index4]
    gen18 = df18['cg_avg_data_test'][Table_index3:Table_index4]
    spec18 = df18['cs_avg_data_test'][Table_index3:Table_index4]
    print("--------------- SUBSET FashionMNIST 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob1))
    print("CDKT Rep KL-N gen:", np.median(gen1))
    print("CDKT Rep KL-N spec:", np.median(spec1))
    print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob2))
    print("CDKT Rep Full KL-N gen:", np.median(gen2))
    print("CDKT Rep Full KL-N spec:", np.median(spec2))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob3))
    print("CDKT Full KL-N gen:", np.median(gen3))
    print("CDKT Full KL-N spec:", np.median(spec3))
    print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)

    print("--------------- SUBSET CIFAR-10 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob4))
    print("CDKT Rep KL-N gen:", np.median(gen4))
    print("CDKT Rep KL-N spec:", np.median(spec4))
    print("CDKT Rep KL-N personalized:", (np.median(spec4) + np.median(gen4)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob5))
    print("CDKT Rep Full KL-N gen:", np.median(gen5))
    print("CDKT Rep Full KL-N spec:", np.median(spec5))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec5) + np.median(gen5)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob6))
    print("CDKT Full KL-N gen:", np.median(gen6))
    print("CDKT Full KL-N spec:", np.median(spec6))
    print("CDKT Full KL-N personalized:", (np.median(spec6) + np.median(gen6)) / 2)
    print("--------------- SUBSET CIFAR-100 50/0.5 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob7))
    print("CDKT Rep KL-N gen:", np.median(gen7))
    print("CDKT Rep KL-N spec:", np.median(spec7))
    print("CDKT Rep KL-N personalized:", (np.median(spec7) + np.median(gen7)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob8))
    print("CDKT Rep Full KL-N gen:", np.median(gen8))
    print("CDKT Rep Full KL-N spec:", np.median(spec8))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec8) + np.median(gen8)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob9))
    print("CDKT Full KL-N gen:", np.median(gen9))
    print("CDKT Full KL-N spec:", np.median(spec9))
    print("CDKT Full KL-N personalized:", (np.median(spec9) + np.median(gen9)) / 2)

    print("--------------- SUBSET FashionMNIST 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob10))
    print("CDKT Rep KL-N gen:", np.median(gen10))
    print("CDKT Rep KL-N spec:", np.median(spec10))
    print("CDKT Rep KL-N personalized:", (np.median(spec10) + np.median(gen10)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob11))
    print("CDKT Rep Full KL-N gen:", np.median(gen11))
    print("CDKT Rep Full KL-N spec:", np.median(spec11))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec11) + np.median(gen11)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob12))
    print("CDKT Full KL-N gen:", np.median(gen12))
    print("CDKT Full KL-N spec:", np.median(spec12))
    print("CDKT Full KL-N personalized:", (np.median(spec12) + np.median(gen12)) / 2)

    print("--------------- SUBSET CIFAR-10 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob13))
    print("CDKT Rep KL-N gen:", np.median(gen13))
    print("CDKT Rep KL-N spec:", np.median(spec13))
    print("CDKT Rep KL-N personalized:", (np.median(spec13) + np.median(gen13)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob14))
    print("CDKT Rep Full KL-N gen:", np.median(gen14))
    print("CDKT Rep Full KL-N spec:", np.median(spec14))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec14) + np.median(gen14)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob15))
    print("CDKT Full KL-N gen:", np.median(gen15))
    print("CDKT Full KL-N spec:", np.median(spec15))
    print("CDKT Full KL-N personalized:", (np.median(spec15) + np.median(gen15)) / 2)
    print("--------------- SUBSET CIFAR-100 100/0.2 RESULTS --------------")

    print("CDKT Rep KL-N glob:", np.median(glob16))
    print("CDKT Rep KL-N gen:", np.median(gen16))
    print("CDKT Rep KL-N spec:", np.median(spec16))
    print("CDKT Rep KL-N personalized:", (np.median(spec16) + np.median(gen16)) / 2)

    print("CDKT Rep Full KL-N glob:", np.median(glob17))
    print("CDKT Rep Full KL-N gen:", np.median(gen17))
    print("CDKT Rep Full KL-N spec:", np.median(spec17))
    print("CDKT Rep Full KL-N personalized:", (np.median(spec17) + np.median(gen17)) / 2)

    print("CDKT Full KL-N glob:", np.median(glob18))
    print("CDKT Full KL-N gen:", np.median(gen18))
    print("CDKT Full KL-N spec:", np.median(spec18))
    print("CDKT Full KL-N personalized:", (np.median(spec18) + np.median(gen18)) / 2)




    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)


    # data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
    #                        gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters
def read_files8():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'FedProx_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'fedkc_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedkc_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmJSD.h5'), 'r')

    df5 = h5py.File(os.path.join(directory, 'fedkc_mnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'fedkc_mnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmJSD.h5'), 'r')

    df7 = h5py.File(os.path.join(directory, 'fedkc_Cifar10_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'fedkc_Cifar10_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmJSD.h5'), 'r')


    # df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # stop1 = df['stop1'][:]
    #
    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    glob1 = df['root_test'][Table_index1:Table_index2]

    glob2 = df2['root_test'][Table_index1:Table_index2]

    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]

    # glob4 = df4['root_test'][Table_index1:Table_index2]
    # gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
    # spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
    # glob5 = df5['root_test'][Table_index1:Table_index2]
    # gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
    # spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]

    print("--------------- FEDLKD RESULTS --------------")

    print("FedLKD glob FMNIST KL:", np.median(glob3))
    print("FedLKD glob FMNIST JSD:", np.median(glob4))
    print("FedLKD glob MNIST KL:", np.median(glob5))
    print("FedLKD glob MNIST JSD:", np.median(glob6))
    print("FedLKD glob CIFAR10 KL:", np.median(glob7))
    print("FedLKD glob CIFAR10 JSD:", np.median(glob8))
    # print("CDKT Rep KL-N gen:", np.median(gen1))
    # print("CDKT Rep KL-N spec:", np.median(spec1))
    # print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)
    #
    # print("CDKT Rep Full KL-N glob:", np.median(glob2))
    # print("CDKT Rep Full KL-N gen:", np.median(gen2))
    # print("CDKT Rep Full KL-N spec:", np.median(spec2))
    # print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)
    #
    # print("CDKT Full KL-N glob:", np.median(glob3))
    # print("CDKT Full KL-N gen:", np.median(gen3))
    # print("CDKT Full KL-N spec:", np.median(spec3))
    # print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)
    #
    # print("fedavg glob:", np.median(glob4))
    # print("fedavg gen:", np.median(gen4))
    # print("fedavg spec:", np.median(spec4))
    # print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)
    #
    # print("CDKT no transfer glob:", np.median(glob5))
    # print("CDKT no transfer gen:", np.median(gen5))
    # print("CDKT no transfer spec:", np.median(spec5))
    # print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)


    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['FedAvg','FedProx','FedLKD']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
plot_final()