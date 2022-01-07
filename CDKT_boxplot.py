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

Table_index1 = 90
Table_index2 = 100
Start_index = 80
def plot_final():
    df_iters = read_files()  # 3 algorithms
    plot_box(df_iters,1)
    plt.savefig("figs/mnist_fixed_users_boxplot.pdf")
    df_iters = read_files1()
    plot_box(df_iters,2)
    plt.savefig("figs/mnist_subset_users_boxplot.pdf")
    df_iters = read_files2()
    plot_box(df_iters,3)
    plt.savefig("figs/fmnist_fixed_users_boxplot.pdf")
    df_iters = read_files3()
    plot_box(df_iters,4)
    plt.savefig("figs/fmnist_subset_users_boxplot.pdf")
    df_iters = read_files4()
    plot_box(df_iters, 5)
    plt.savefig("figs/cifar10_fixed_users_boxplot.pdf")
    df_iters = read_files5()
    plot_box(df_iters, 6)
    plt.savefig("figs/cifar10_subset_users_boxplot.pdf")
    df_iters = read_files6()
    plot_box(df_iters, 7)
    plt.savefig("figs/cifar100_fixed_users_boxplot.pdf")
    df_iters = read_files7()
    plot_box(df_iters, 8)
    plt.savefig("figs/cifar100_subset_users_boxplot.pdf")
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
def read_files3():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
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
def read_files4():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
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


def read_files5():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.75_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
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
def read_files6():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.5_b0.2_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
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


def read_files7():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
    df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')

    df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
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
plot_final()