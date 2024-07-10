import h5py as hf
import numpy as np
from Setting import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({'font.size': 16})  #font size 10 12 14 16 main 16
plt.rcParams['lines.linewidth'] = 2
YLim=0
#Global variable
markers_on = 10 #maker only at x = markers_on[]
OUT_TYPE = ".pdf" #.eps, .pdf, .jpeg #output figure type

color = {
    "gen": "royalblue",
    "cspe": "forestgreen",
    "cgen": "red",
    "c": "cyan",
    "gspe": "darkorange",  #magenta
    "gg": "yellow",
    "ggen": "darkviolet",
    "w": "white",
    "globloss":"orange"
}
marker = {
    "gen": "8",
    "gspe": "s",
    "ggen": "P",
    "cspe": "p",
    "cgen": "*"
}

def read_data(file_name = "../results/untitled.h5"):
    dic_data = {}
    with hf.File(file_name, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        for key in f.keys():
            try:
                dic_data[key] = f[key][:]
            except:
                dic_data[key] = f[key]
    return  dic_data









def plot_fixed_users():
    if  DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1= 1.0
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfKLNnew'])
    print("CDKTrep Global :", f_data['root_test'][12])
    print("CDKTrep Global:", f_data['root_test'][XLim-1])
    print("CDKTrep C-GEN:",f_data['cg_avg_data_test'][XLim-1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim,  YLim1)
    ax1.set_title("CDKT (Rep+KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLNnew'])
    print("CDKTRepFull Global", f_data['root_test'][12])
    print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim-1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, YLim1)
    ax2.grid()
    ax2.set_title("CDKT (RepFull+KL-N)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTfullfKLNnew'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, YLim1)
    ax3.grid()
    ax3.set_title("CDKT (Full+KL-N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavgfixednew'])
    print("FedAvg Global:", fed_data['root_test'][XLim-1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, YLim1)
    ax4.grid()
    ax4.set_title("FedAvg")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5

    fed_data = read_data(RS_PATH + name['CDKTnotransferF'])
    print("CDKTnotransfer Global:", fed_data['root_test'][XLim-1])
    ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax5.set_xlim(0, XLim)
    ax5.set_ylim(YLim, YLim1)
    ax5.grid()
    ax5.set_title("No transfer")
    ax5.set_xlabel("#Global Rounds")
    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "fixed_users" + OUT_TYPE)


    return 0

def plot_fixed_users_EAAI():
    if  DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1= 1.0
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, axes= plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(15.0, 8.8))
    f_data = read_data(RS_PATH + name['CDKTrepfKLNnew'])
    # print("CDKTrep Global :", f_data['root_test'][12])
    # print("CDKTrep Global:", f_data['root_test'][XLim-1])
    # print("CDKTrep C-GEN:",f_data['cg_avg_data_test'][XLim-1])

    axes[0,0].plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[0,0].plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    axes[0,0].plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    axes[0,0].set_xlim(0, XLim)
    axes[0,0].set_ylim(YLim,  YLim1)
    axes[0,0].set_title("CDKT (Rep+KL-N)")
    axes[0,0].set_xlabel("#Global Rounds")
    axes[0,0].set_ylabel("Testing Accuracy")
    axes[0,0].grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLNnew'])
    # print("CDKTRepFull Global", f_data['root_test'][12])
    # print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim-1])
    axes[0,1].plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[0,1].plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[0,1].plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    axes[0,1].set_xlim(0, XLim)
    axes[0,1].set_ylim(YLim, YLim1)
    axes[0,1].grid()
    axes[0,1].set_title("CDKT (RepFull+KL-N)")
    axes[0,1].set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTfullfKLNnew'])
    axes[0,2].plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    axes[0,2].plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    axes[0,2].plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    axes[0,2].set_xlim(0, XLim)
    axes[0,2].set_ylim(YLim, YLim1)
    axes[0,2].grid()
    axes[0,2].set_title("CDKT (Full+KL-N)")
    axes[0,2].set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavgfixednew'])
    # print("FedAvg Global:", fed_data['root_test'][XLim-1])
    axes[1,0].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    axes[1,0].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    axes[1,0].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1,0].set_xlim(0, XLim)
    axes[1,0].set_ylim(YLim, YLim1)
    axes[1,0].grid()
    axes[1,0].set_title("FedAvg")
    axes[1,0].set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5

    fed_data = read_data(RS_PATH + name['CDKTnotransferF'])
    # print("CDKTnotransfer Global:", fed_data['root_test'][XLim-1])
    axes[0,3].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    axes[0,3].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    axes[0,3].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[0,3].set_xlim(0, XLim)
    axes[0,3].set_ylim(YLim, YLim1)
    axes[0,3].grid()
    axes[0,3].set_title("No transfer")
    axes[0,3].set_xlabel("#Global Rounds")
    axes[1,0].set_ylabel("Testing Accuracy")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['Scaffoldfixed'])
    # print("Scaffoldfixed Global:", fed_data['root_test'][XLim - 1])
    axes[1,1].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[1,1].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[1,1].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1,1].set_xlim(0, XLim)
    axes[1,1].set_ylim(YLim, YLim1)
    axes[1,1].grid()
    axes[1,1].set_title("Scaffold")
    axes[1,1].set_xlabel("#Global Rounds")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['MOONfixed'])
    # print("MOONfixed Global:", fed_data['root_test'][XLim - 1])
    axes[1,2].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[1,2].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[1,2].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1,2].set_xlim(0, XLim)
    axes[1,2].set_ylim(YLim, YLim1)
    axes[1,2].grid()
    axes[1,2].set_title("MOON")
    axes[1,2].set_xlabel("#Global Rounds")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['FedDynfixed'])
    # print("FedDynfixed Global:", fed_data['root_test'][XLim - 1])
    axes[1,3].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[1,3].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[1,3].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1,3].set_xlim(0, XLim)
    axes[1,3].set_ylim(YLim, YLim1)
    axes[1,3].grid()
    axes[1,3].set_title("FedDyn")
    axes[1,3].set_xlabel("#Global Rounds")
    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles1, labels1 = axes[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,bbox_to_anchor=(0.5, 0.1),
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    fig.legend(handles1, labels1, loc="lower center", borderaxespad=0.1, ncol=5,bbox_to_anchor=(0.5, 0.1),
               prop={'size': 16}),  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "fixed_users_EAAI" + OUT_TYPE)


    return 0

def plot_subset_users():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepsKLNnew'])
    print("CDKTrep Global :", f_data['root_test'][12])
    print("CDKTrep C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, YLim1)
    ax1.set_title("CDKT (Rep+KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullsKLNnew'])
    print("CDKTRepFull Global", f_data['root_test'][12])
    print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, YLim1)
    ax2.grid()
    ax2.set_title("CDKT (RepFull+KL-N)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTfullsKLNnew'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, YLim1)
    ax3.grid()
    ax3.set_title("CDKT (Full+KL-N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavgsubnew'])
    print("FedAvg Global:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, YLim1)
    ax4.grid()
    ax4.set_title("FedAvg")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5

    fed_data = read_data(RS_PATH + name['CDKTnotransferS'])
    print("CDKT no transfer Global:", fed_data['root_test'][XLim-1])
    ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax5.set_xlim(0, XLim)
    ax5.set_ylim(YLim, YLim1)
    ax5.grid()
    ax5.set_title("No transfer")
    ax5.set_xlabel("#Global Rounds")
    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "subset_users" + OUT_TYPE)


    return 0

def scalability1():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTRep_50_05'])
    # print("CDKTrep Global :", f_data['root_test'][12])
    # print("CDKTrep C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, YLim1)
    ax1.set_title("CDKT (Rep+KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTRepFull_50_05'])
    print("CDKTRepFull Global", f_data['root_test'][12])
    print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, YLim1)
    ax2.grid()
    ax2.set_title("CDKT (RepFull+KL-N)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTFull_50_05'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, YLim1)
    ax3.grid()
    ax3.set_title("CDKT (Full+KL-N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4


    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "scalability1" + OUT_TYPE)


    return 0

def scalability2():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTRep_100_02'])
    # print("CDKTrep Global :", f_data['root_test'][12])
    # print("CDKTrep C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, YLim1)
    ax1.set_title("CDKT (Rep+KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTRepFull_100_02'])
    print("CDKTRepFull Global", f_data['root_test'][12])
    print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, YLim1)
    ax2.grid()
    ax2.set_title("CDKT (RepFull+KL-N)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTFull_100_02'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, YLim1)
    ax3.grid()
    ax3.set_title("CDKT (Full+KL-N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4


    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "scalability2" + OUT_TYPE)


    return 0


def plot_subset_users_EAAI():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(15.0, 8.8))
    f_data = read_data(RS_PATH + name['CDKTrepsKLNnew'])
    # print("CDKTrep Global :", f_data['root_test'][12])
    # print("CDKTrep C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    axes[0,0].plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[0,0].plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    axes[0,0].plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    axes[0,0].set_xlim(0, XLim)
    axes[0,0].set_ylim(YLim, YLim1)
    axes[0,0].set_title("CDKT (Rep+KL-N)")
    axes[0,0].set_xlabel("#Global Rounds")
    axes[0,0].set_ylabel("Testing Accuracy")
    axes[0,0].grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullsKLNnew'])
    # print("CDKTRepFull Global", f_data['root_test'][12])
    # print("CDKTRepFull C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    axes[0,1].plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[0,1].plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[0,1].plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    axes[0,1].set_xlim(0, XLim)
    axes[0,1].set_ylim(YLim, YLim1)
    axes[0,1].grid()
    axes[0,1].set_title("CDKT (RepFull+KL-N)")
    axes[0,1].set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTfullsKLNnew'])
    axes[0,2].plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[0,2].plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[0,2].plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    axes[0,2].set_xlim(0, XLim)
    axes[0,2].set_ylim(YLim, YLim1)
    axes[0,2].grid()
    axes[0,2].set_title("CDKT (Full+KL-N)")
    axes[0,2].set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavgsubnew'])
    # print("FedAvg Global:", fed_data['root_test'][XLim - 1])
    axes[1,0].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    axes[1,0].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    axes[1,0].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1,0].set_xlim(0, XLim)
    axes[1,0].set_ylim(YLim, YLim1)
    axes[1,0].grid()
    axes[1,0].set_title("FedAvg")
    axes[1,0].set_ylabel("Testing Accuracy")
    axes[1,0].set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5

    fed_data = read_data(RS_PATH + name['CDKTnotransferS'])
    # print("CDKT no transfer Global:", fed_data['root_test'][XLim-1])
    axes[0,3].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    axes[0,3].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    axes[0,3].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[0,3].set_xlim(0, XLim)
    axes[0,3].set_ylim(YLim, YLim1)
    axes[0,3].grid()
    axes[0,3].set_title("No transfer")
    axes[0,3].set_xlabel("#Global Rounds")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['Scaffoldsubset'])
    # print("Scaffoldsubset Global:", fed_data['root_test'][XLim - 1])
    axes[1, 1].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
                    markevery=markers_on)
    axes[1, 1].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
                    markevery=markers_on)
    axes[1, 1].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
                    markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1, 1].set_xlim(0, XLim)
    axes[1, 1].set_ylim(YLim, YLim1)
    axes[1, 1].grid()
    axes[1, 1].set_title("Scaffold")
    axes[1, 1].set_xlabel("#Global Rounds")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['MOONsubset'])
    # print("MOONsubset Global:", fed_data['root_test'][XLim - 1])
    axes[1, 2].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
                    markevery=markers_on)
    axes[1, 2].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
                    markevery=markers_on)
    axes[1, 2].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
                    markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1, 2].set_xlim(0, XLim)
    axes[1, 2].set_ylim(YLim, YLim1)
    axes[1, 2].grid()
    axes[1, 2].set_title("MOON")
    axes[1, 2].set_xlabel("#Global Rounds")
    plt.tight_layout()

    fed_data = read_data(RS_PATH + name['FedDynsubset'])
    # print("FedDynsubset Global:", fed_data['root_test'][XLim - 1])
    axes[1, 3].plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
                    markevery=markers_on)
    axes[1, 3].plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
                    markevery=markers_on)
    axes[1, 3].plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
                    markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    axes[1, 3].set_xlim(0, XLim)
    axes[1, 3].set_ylim(YLim, YLim1)
    axes[1, 3].grid()
    axes[1, 3].set_title("FedDyn")
    axes[1, 3].set_xlabel("#Global Rounds")
    plt.tight_layout()
    plt.grid(linewidth=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5, bbox_to_anchor=(0.5, 0.1),
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    fig.legend(handles1, labels1, loc="lower center", borderaxespad=0.1, ncol=5, bbox_to_anchor=(0.5, 0.1),
               prop={'size': 16}),  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "subset_users_EAAI" + OUT_TYPE)


    return 0
def plot_metric_fixed():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfKLNnew'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT (KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfKLnew'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT (KL)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepfNnew'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT (N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepfJnew'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT (JS)")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5

    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "metric_fixed" + OUT_TYPE)


    return 0


def plot_metric_subset():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    if DATASET=="Cifar-10":
        f_data = read_data(RS_PATH + name['CDKTrepfullsKLN'])
    else:
        f_data = read_data(RS_PATH + name['CDKTrepsKLNnew'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT (KL-N)")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepsKLnew'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT (KL)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepsNnew'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT (N)")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepsJnew'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT (JS)")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "metric_subset" + OUT_TYPE)


    return 0

def plot_same_vs_hetero_fixed():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7.5, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfKLNnew'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, YLim1)
    ax1.set_title("CDKT")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepHfKLNnew'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, YLim1)
    ax2.grid()
    ax2.set_title("CDKT (Heterogeneous)")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3
    # if DATASET=="Cifar-10" or DATASET == "Cifar-100":
    #     fed_data2 = read_data(RS_PATH + name['CDKTrepfullsKLNnew'])
    # else:
    #     fed_data2 = read_data(RS_PATH + name['CDKTrepsKLNnew'])
    # ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax3.set_xlim(0, XLim)
    # ax3.set_ylim(YLim, YLim1)
    # ax3.grid()
    # ax3.set_title("CDKT")
    # ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    # fed_data = read_data(RS_PATH + name['CDKTrepHsKLNnew'])
    # print("JSD:", fed_data['root_test'][XLim - 1])
    # ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax4.set_xlim(0, XLim)
    # ax4.set_ylim(YLim, YLim1)
    # ax4.grid()
    # ax4.set_title("CDKT (Heterogeneous)")
    # ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "same_vs_hetero_fixed" + OUT_TYPE)


    return 0

def plot_same_vs_hetero_subset():
    if DATASET == "Cifar-100":
        YLim1 = 0.5
    else:
        YLim1 = 1.0
    plt.rcParams.update({'font.size': 14})
    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7.5, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # f_data = read_data(RS_PATH + name['CDKTrepfKLNnew'])
    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])
    #
    # ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
    #          marker=marker["cspe"], markevery=markers_on,
    #          label="C-SPE")
    # ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
    #          marker=marker["cgen"], markevery=markers_on,
    #          label="C-GEN")
    # ax1.set_xlim(0, XLim)
    # ax1.set_ylim(YLim, YLim1)
    # ax1.set_title("CDKT")
    # ax1.set_xlabel("#Global Rounds")
    # ax1.set_ylabel("Testing Accuracy")
    # ax1.grid()
    # # subfig1-end---begin---subfig 2
    #
    # fed_data2 = read_data(RS_PATH + name['CDKTrepHfKLNnew'])
    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    # ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax2.set_xlim(0, XLim)
    # ax2.set_ylim(YLim, YLim1)
    # ax2.grid()
    # ax2.set_title("CDKT (Heterogeneous)")
    # ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3
    if DATASET=="Cifar-10" or DATASET == "Cifar-100":
        fed_data2 = read_data(RS_PATH + name['CDKTrepfullsKLNnew'])
    else:
        fed_data2 = read_data(RS_PATH + name['CDKTrepsKLNnew'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, YLim1)
    ax3.grid()
    ax3.set_title("CDKT")
    ax3.set_ylabel("Testing Accuracy")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepHsKLNnew'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, YLim1)
    ax4.grid()
    ax4.set_title("CDKT (Heterogeneous)")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)

    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "same_vs_hetero_subset" + OUT_TYPE)


    return 0

def plot_alpha_effect_fixed():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfKLN_alpha_0.01'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-Rep: $\\alpha=0.01$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfKLN_alpha_0.02'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-Rep: $\\alpha=0.02$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepfKLN_alpha_0.05'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-Rep: $\\alpha=0.05$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepfKLN_alpha_0.1'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-Rep: $\\alpha=0.1$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(PLOT_PATH+ DATASET + "alpha_effect_fixed" + OUT_TYPE)


    return 0

def plot_beta_effect_fixed():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfKLN_beta_0.01'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-Rep: $\\beta=0.01$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfKLN_beta_0.05'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-Rep: $\\beta=0.05$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepfKLN_beta_0.1'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-Rep: $\\beta=0.1$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepfKLN_beta_0.15'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-Rep: $\\beta=0.15$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "beta_effect_fixed" + OUT_TYPE)


    return 0
def plot_alpha_effect_subset():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepsKLN_alpha_0.05'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-Rep: $\\alpha=0.05$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepsKLN_alpha_0.1'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-Rep: $\\alpha=0.1$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepsKLN_alpha_0.15'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-Rep: $\\alpha=0.15$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepsKLN_alpha_0.2'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-Rep: $\\alpha=0.2$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(PLOT_PATH+ DATASET + "alpha_effect_subset" + OUT_TYPE)


    return 0

def plot_beta_effect_subset():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepsKLN_beta_0.005'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-Rep: $\\beta=0.005$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepsKLN_beta_0.05'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-Rep: $\\beta=0.05$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepsKLN_beta_0.1'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-Rep: $\\beta=0.1$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepsKLN_beta_0.2'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-Rep: $\\beta=0.2$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "beta_effect_subset" + OUT_TYPE)
def plot_alpha_effect2():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfullfKLN_alpha_0.01'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-FullRep: $\\alpha=0.01$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLN_alpha_0.009'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-FullRep: $\\alpha=0.009$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLN_alpha_0.007'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-FullRep: $\\alpha=0.007$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepfullfKLN_alpha_0.005'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-FullRep: $\\alpha=0.005$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "alpha_effect" + OUT_TYPE)


    return 0

def plot_beta_effect2():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['CDKTrepfullfKLN_beta_0.03'])
    print("CDKTKLN Global :", f_data['root_test'][12])
    print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("CDKT-FullRep: $\\beta=0.03$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLN_beta_0.02'])
    print("CDKTKL Global", f_data['root_test'][12])
    print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("CDKT-FullRep: $\\beta=0.02$")
    ax2.set_xlabel("#Global Rounds")

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['CDKTrepfullfKLN_beta_0.01'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("CDKT-FullRep: $\\beta=0.01$")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['CDKTrepfullfKLN_beta_0.008'])
    print("JSD:", fed_data['root_test'][XLim - 1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("CDKT-FullRep: $\\beta=0.008$")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
    #            prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    # plt.subplots_adjust(bottom=0.25)
    # plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)2
    # END-subfig4-begin-subfig5
    #
    # fed_data = read_data(RS_PATH + name['pFedMe'])
    # print("pFedMe Global:", fed_data['root_test'][XLim-1])
    # ax5.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax5.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax5.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # plt.legend(loc="best", prop={'size': 8})
    # ax5.set_xlim(0, XLim)
    # ax5.set_ylim(YLim, 1)
    # ax5.grid()
    # ax5.set_title("pFedMe")
    # ax5.set_xlabel("#Global Rounds")
    # plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "beta_effect" + OUT_TYPE)


    return 0

def plot_KSC():
    plt.rcParams.update({'font.size': 80})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(45.0, 25))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    fedavg_data = read_data(RS_PATH + name['fedavg_KSC'])
    fedBD_data = read_data(RS_PATH + name['FedBD_KSC'])
    fedBD_data_adam = read_data(RS_PATH + name['FedBD_KSC_Adam'])

    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(fedavg_data['cs_avg_data_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(fedBD_data['cs_avg_data_test'])), fedBD_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="FedBD")
    ax1.plot(np.arange(len(fedBD_data_adam['cs_avg_data_test'])), fedBD_data_adam['cs_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="FedABD")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("Specialized Performance")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(fedavg_data['cg_avg_data_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(fedBD_data['cg_avg_data_test'], label="FedBD", color=color["cspe"], marker=marker["cspe"],
             markevery=markers_on)
    ax2.plot(fedBD_data_adam['cg_avg_data_test'], label="AFedBD", color=color["cgen"], marker=marker["cgen"],
             markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title("Generalized Performance")
    ax2.set_xlabel("#Global Rounds")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=3,
               prop={'size': 80})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "KSC" + OUT_TYPE)
def plot_ICOIN2022():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    FedAvg_ICOIN = read_data(RS_PATH + name['FedAvg_ICOIN'])
    FedProx_ICOIN = read_data(RS_PATH + name['FedProx_ICOIN'])
    FedLKD_ICOIN = read_data(RS_PATH + name['FedLKD_ICOIN'])

    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(FedAvg_ICOIN['root_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(FedProx_ICOIN['root_test'])), FedProx_ICOIN['root_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="FedProx")
    ax1.plot(np.arange(len(FedLKD_ICOIN['root_test'])), FedLKD_ICOIN['root_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="FedLKD")
    ax1.set_xlim(0, XLim)
    if DATASET =="mnist":
        ax1.set_ylim(YLim, 1)
    elif DATASET =="fmnist":
        ax1.set_ylim(YLim, 0.9)
    elif DATASET=="Cifar-10":
        ax1.set_ylim(YLim, 0.6)
    # ax1.set_title()
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    # ax2.plot(fedavg_data['cg_avg_data_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data['cg_avg_data_test'], label="FedBD", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data_adam['cg_avg_data_test'], label="AFedBD", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax2.set_xlim(0, XLim)
    # ax2.set_ylim(YLim, 1)
    # ax2.grid()
    # ax2.set_title("Generalized Performance")
    # ax2.set_xlabel("#Global Rounds")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(loc="lower right",
               prop={'size': 16})
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=1,
    #            prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "ICOIN" + OUT_TYPE)

def plot_global_loss():
    plt.rcParams.update({'font.size': 16})
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    CDKT_KL_N = read_data(RS_PATH + name['CDKTKLN_EAAI'])
    CDKT_KL_KL = read_data(RS_PATH + name['CDKTKL_EAAI'])
    CDKT_N_N = read_data(RS_PATH + name['CDKTN_EAAI'])
    CDKT_JSD_JSD = read_data(RS_PATH + name['CDKTJSD_EAAI'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(CDKT_KL_N['global_loss'], label="KL-N", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(CDKT_KL_KL['global_loss'])), CDKT_KL_KL['global_loss'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="KL")
    ax1.plot(np.arange(len(CDKT_N_N['global_loss'])), CDKT_N_N['global_loss'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="N")
    ax1.plot(np.arange(len(CDKT_JSD_JSD['global_loss'])), CDKT_JSD_JSD['global_loss'], color=color["globloss"],
             marker=marker["cgen"], markevery=markers_on,
             label="JSD")
    ax1.set_xlim(0, XLim)
    if DATASET =="mnist":
        ax1.set_ylim(YLim, 0.1)
    elif DATASET =="fmnist":
        ax1.set_ylim(YLim, 4)
    elif DATASET=="Cifar-10" or DATASET=="Cifar-100":
        ax1.set_ylim(YLim, 6)
    # ax1.set_title()
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Loss")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    # ax2.plot(fedavg_data['cg_avg_data_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data['cg_avg_data_test'], label="FedBD", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data_adam['cg_avg_data_test'], label="AFedBD", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax2.set_xlim(0, XLim)
    # ax2.set_ylim(YLim, 1)
    # ax2.grid()
    # ax2.set_title("Generalized Performance")
    # ax2.set_xlabel("#Global Rounds")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(loc="upper right",
               prop={'size': 16})
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=1,
    #            prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(PLOT_PATH+ DATASET + "global_loss" + OUT_TYPE)

def plot_local_loss():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 4.4))
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    CDKT_KL_N = read_data(RS_PATH + name['CDKTKLN_EAAI'])
    CDKT_KL_KL = read_data(RS_PATH + name['CDKTKL_EAAI'])
    CDKT_N_N = read_data(RS_PATH + name['CDKTN_EAAI'])
    CDKT_JSD_JSD = read_data(RS_PATH + name['CDKTJSD_EAAI'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(CDKT_KL_N['local_loss'], label="KL-N", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(CDKT_KL_KL['local_loss'])), CDKT_KL_KL['local_loss'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="KL")
    ax1.plot(np.arange(len(CDKT_N_N['local_loss'])), CDKT_N_N['local_loss'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="N")
    ax1.plot(np.arange(len(CDKT_JSD_JSD['local_loss'])), CDKT_JSD_JSD['local_loss'], color=color["globloss"],
             marker=marker["cgen"], markevery=markers_on,
             label="JSD")
    ax1.set_xlim(0, XLim)
    if DATASET =="mnist":
        ax1.set_ylim(YLim, 0.1)
    elif DATASET =="fmnist":
        ax1.set_ylim(YLim, 6)
    elif DATASET=="Cifar-10" :
        ax1.set_ylim(YLim, 4)
    elif DATASET=="Cifar-100":
        ax1.set_ylim(YLim, 6)
    # ax1.set_title()
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Average loss")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    # ax2.plot(fedavg_data['cg_avg_data_test'], label="FedAvg", linestyle="--", color=color["gen"], marker=marker["gen"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data['cg_avg_data_test'], label="FedBD", color=color["cspe"], marker=marker["cspe"],
    #          markevery=markers_on)
    # ax2.plot(fedBD_data_adam['cg_avg_data_test'], label="AFedBD", color=color["cgen"], marker=marker["cgen"],
    #          markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax2.set_xlim(0, XLim)
    # ax2.set_ylim(YLim, 1)
    # ax2.grid()
    # ax2.set_title("Generalized Performance")
    # ax2.set_xlabel("#Global Rounds")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(loc="upper right",
               prop={'size': 14})
    # fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=1,
    #            prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(PLOT_PATH+ DATASET + "local_loss" + OUT_TYPE)




def plot_all_figs():
    # global CLUSTER_METHOD
    # plot_dem_vs_fed()  # plot comparision FED vs DEM
    # plot_demlearn_vs_demlearn_p()  # DEM, PROX vs K level
    # plot_demlearn_p_mu_vari()  # DEM Prox vs mu vary
    plot_fixed_users()
    plot_subset_users()
    plot_metric_fixed()
    plot_metric_subset()
    # plot_same_vs_hetero()
    plot_fixed_users_EAAI()
    # ### SUPPLEMENTAL FIGS ####
    # plot_demlearn_gamma_vari() # DEM AVG vs Gamma vary
    # plot_demlearn_gamma_vari_clients()
    # plot_demlearn_w_vs_g()
    # # ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # CLUSTER_METHOD = "gradient"
    # # plot_dendo_data_dem(file_name="prox3g")  # change file_name in order to get correct file to plot   #|
    # plot_dendo_data_dem(file_name="avg3g")  # change file_name in order to get correct file to plot   #|
    # plot_dendo_data_dem_shashi(file_name="avg3g", type="Gradient")
    # CLUSTER_METHOD = "weight"
    # # plot_dendo_data_dem(file_name="prox3w")
    # plot_dendo_data_dem(file_name="avg3w")
    # plot_dendo_data_dem_shashi(file_name="avg3w", type="Weight")
    ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # plot_from_file()
    # plt.show()


if __name__=='__main__':
    PLOT_PATH = PLOT_PATH_FIG
    RS_PATH = FIG_PATH

    #### PLOT MNIST #####
    print("----- PLOT MNIST ------")
    DATASET = "mnist"
    NUM_GLOBAL_ITERS = 100
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 4

    name = {
        # "avg1w": "mnist_demlearn_I60_K1_T2_b1-0_dTrue_m0-002_w.h5",
        # "avg1wf": "mnist_demlearn_I60_K1_T60_b1-0_dTrue_m0-002_w.h5",
        # # "avg2w": "demlearn_iter_60_k_2_w.h5",
        # "avg3g": "mnist_demlearn_I60_K3_T2_b1-0_dTrue_m0-002_g.h5",
        # "avg3w": "mnist_demlearn_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        # "avg3wf": "mnist_demlearn_I60_K3_T60_b1-0_dTrue_m0-002_w.h5",
        # "prox1w": "mnist_demlearn-p_I60_K1_T2_b1-0_dTrue_m0-002_w.h5",
        # "prox1wf": "mnist_demlearn-p_I60_K1_T60_b1-0_dTrue_m0-002_w.h5",
        # # "prox2w": "demlearn-p_iter_60_k_2_w.h5",
        # "prox3w": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        # "prox3g": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_g.h5",
        # "prox3wf": "mnist_demlearn-p_I60_K3_T60_b1-0_dTrue_m0-002_w.h5",
        # # "fedavg": "mnist_fedavg_I60.h5",
        # "fedprox": "mnist_fedprox_I60.h5",
        # "pFedMe": "mnist_pFedMe_I60.h5",
        # # "avg3b08": "demlearn_iter_60_k_3_w_beta_0_8.h5",
        # # "avg3wdecay": "demlearn_iter_60_k_3_w_decay.h5",
        # # "avg3wg08": "demlearn_iter_60_k_3_w_gamma_0_8.h5",
        # # "avg3g1": "demlearn_iter_60_k_3_w_gamma_1.h5",
        # # "prox3wg08": "demlearn-p_iter_60_k_3_w_gamma_0_8.h5",
        # # "prox3wg1": "demlearn-p_iter_60_k_3_w_gamma_1.h5",
        # "prox3wmu001": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-001_w.h5",
        # "prox3wmu002": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        # "prox3wmu0005": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-0005_w.h5",
        # "prox3wmu005": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-005_w.h5",
        "fedavgfixed":"fedavg_mnist_I100_sTrue_fTrue_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN":"CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN":"CDKT_mnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLN":"CDKT_mnist_I100_sTrue_fTrue_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "fedavgsub":"fedavg_mnist_I100_sTrue_fTrue_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepsKLN":"CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLN":"CDKT_mnist_I100_sTrue_fFalse_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTfullsKLN":"CDKT_mnist_I100_sTrue_fTrue_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullfKL":"CDKT_mnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmKL_lmKL.h5",
        "CDKTrepfullfN":"CDKT_mnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepfullfJ":"CDKT_mnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmJSD_lmJSD.h5",
        "CDKTrepsKL":"CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmKL_lmKL.h5",
        "CDKTrepsN":"CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmNorm2_lmNorm2.h5",
        "CDKTrepsJ":"CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmJSD_lmJSD.h5",
        "CDKTnotransferF":"CDKT_mnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTnotransferS":"CDKT_mnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullHfKLN":"CDKT_mnist_I100_sFalse_fFalse_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLN":"CDKT_mnist_I100_sFalse_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "fedavgfixednew": "fedavg_mnist_I100_sTrue_fFalse_a0.12_b0.6_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.06_b0.6_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.07_b0.5_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLNnew": "CDKT_mnist_I100_sTrue_fTrue_a0.12_b0.6_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "fedavgsubnew": "fedavg_mnist_I100_sTrue_fFalse_a0.06_b0.9_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.25_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.25_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullsKLNnew": "CDKT_mnist_I100_sTrue_fTrue_a0.25_b0.1_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLnew":"CDKT_mnist_I100_sTrue_fFalse_a2_b0.2_RFFalse_SSTrue_accFalse_gmKL_lmKL.h5",
        "CDKTrepsNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.25_b0.01_RFFalse_SSTrue_accFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepsJnew": "CDKT_mnist_I100_sTrue_fFalse_a2.5_b0.2_RFFalse_SSTrue_accFalse_gmJSD_lmJSD.h5",
        "CDKTrepfKLnew": "CDKT_mnist_I100_sTrue_fFalse_a0.3_b0.15_RFFalse_SSFalse_accFalse_gmKL_lmKL.h5",
        "CDKTrepfNnew": "CDKT_mnist_I100_sTrue_fFalse_a0.2_b0.05_RFFalse_SSFalse_accFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepfJnew": "CDKT_mnist_I100_sTrue_fFalse_a2_b0.5_RFFalse_SSFalse_accFalse_gmJSD_lmJSD.h5",
        "CDKTrepHfKLNnew": "CDKT_mnist_I100_sFalse_fFalse_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLNnew": "CDKT_mnist_I100_sFalse_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "FedAvg_ICOIN":"fedavg_mnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedProx_ICOIN":"FedProx_mnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedLKD_ICOIN":"fedkc_mnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5"

        }

    # plot_all_figs()
    # plot_ICOIN2022()

    #### PLOT FASHION-MNIST #####
    print("----- PLOT FASHION-MNIST ------")
    DATASET = "fmnist"
    NUM_GLOBAL_ITERS = 200
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7

    name = {

        "fedavgfixed": "fedavg_fmnist_I100_sTrue_fTrue_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN": "CDKT_fmnist_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN": "CDKT_fmnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLN": "CDKT_fmnist_I100_sTrue_fTrue_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "fedavgsub": "fedavg_fmnist_I100_sTrue_fTrue_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepsKLN": "CDKT_fmnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLN": "CDKT_fmnist_I100_sTrue_fFalse_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTfullsKLN": "CDKT_fmnist_I100_sTrue_fTrue_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullfKL": "CDKT_fmnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmKL_lmKL.h5",
        "CDKTrepfullfN": "CDKT_fmnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepfullfJ": "CDKT_fmnist_I100_sTrue_fFalse_RFTrue_SSFalse_gmJSD_lmJSD.h5",
        "CDKTrepsKL": "CDKT_fmnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmKL_lmKL.h5",
        "CDKTrepsN": "CDKT_fmnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmNorm2_lmNorm2.h5",
        "CDKTrepsJ": "CDKT_fmnist_I100_sTrue_fFalse_RFFalse_SSTrue_gmJSD_lmJSD.h5",
        "CDKTnotransferF": "CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTnotransferS":"CDKT_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullHfKLN": "CDKT_fmnist_I100_sFalse_fFalse_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLN": "CDKT_fmnist_I100_sFalse_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.05":"CDKT_fmnist_I100_sFalse_fFalse_a0.05_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.1": "CDKT_fmnist_I100_sFalse_fFalse_a0.1_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.5": "CDKT_fmnist_I100_sFalse_fFalse_a0.5_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.1":"CDKT_fmnist_I100_sFalse_fFalse_a0.01_b0.1_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.5": "CDKT_fmnist_I100_sFalse_fFalse_a0.01_b0.5_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_1": "CDKT_fmnist_I100_sFalse_fFalse_a0.01_b1_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.01": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.009": "CDKT_fmnist_I100_sTrue_fFalse_a0.005_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.007": "CDKT_fmnist_I100_sTrue_fFalse_a0.005_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_alpha_0.005": "CDKT_fmnist_I100_sTrue_fFalse_a0.005_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.03": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.03_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.02": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.02_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.01": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.01_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN_beta_0.008": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.008_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "fedavg_KSC":"fedavg_fmnist_I100_sTrue_fTrue_a0.01_b0.05_RFFalse_SSFalse_gmKL_lmKL.h5",
        "FedBD_KSC":"CDKT_fmnist_I100_sTrue_fTrue_a0.08_b0.08_RFFalse_SSFalse_gmKL_lmKL.h5",
        "FedBD_KSC_Adam":"CDKT_fmnist_I100_sTrue_fTrue_a0.02_b0.08_RFFalse_SSFalse_gmKL_lmKL(Adam).h5",
        "CDKTrepsKLN_alpha_0.05": "CDKT_fmnist_I100_sTrue_fFalse_a0.05_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_alpha_0.1": "CDKT_fmnist_I100_sTrue_fFalse_a0.1_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_alpha_0.15": "CDKT_fmnist_I100_sTrue_fFalse_a0.15_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_alpha_0.2": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.005": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.005_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.01": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.01_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.05": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.05_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.1": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.1_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.15": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLN_beta_0.2": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.2_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_alpha_0.01": "CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_alpha_0.02": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_alpha_0.05": "CDKT_fmnist_I100_sTrue_fFalse_a0.05_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_alpha_0.1": "CDKT_fmnist_I100_sTrue_fFalse_a0.1_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_beta_0.01": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.01_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_beta_0.05": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.05_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_beta_0.1": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN_beta_0.15": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullsKLNnew": "CDKT_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "fedavgsubnew": "fedavg_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "fedavgfixednew": "fedavg_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLNnew": "CDKT_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.04_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmKL.h5",
        "CDKTrepfNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.08_b0.1_RFFalse_SSFalse_accFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepfJnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.15_b0.06_RFFalse_SSFalse_accFalse_gmJSD_lmJSD.h5",
        "CDKTrepsKLnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.5_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmKL.h5",
        "CDKTrepsNnew": "CDKT_fmnist_I100_sTrue_fFalse_a0.25_b0.02_RFFalse_SSTrue_accFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepsJnew": "CDKT_fmnist_I100_sTrue_fFalse_a2_b0.1_RFFalse_SSTrue_accFalse_gmJSD_lmJSD.h5",
        "CDKTrepHfKLNnew": "CDKT_fmnist_I100_sFalse_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLNnew": "CDKT_fmnist_I100_sFalse_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedAvg_ICOIN": "fedavg_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedProx_ICOIN": "FedProx_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedLKD_ICOIN": "fedkc_fmnist_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "Scaffoldfixed":"Scaffold_fmnist_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "Scaffoldsubset":"Scaffold_fmnist_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONfixed":"MOON_fmnist_I100_sTrue_fTrue_a0.02_b0.1_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONsubset":"MOON_fmnist_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSTrue_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "FedDynfixed":"FedDyn_fmnist_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "FedDynsubset":"FedDyn_fmnist_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "CDKTKLN_EAAI":"CDKT_fmnist_I100_sTrue_fFalse_a0.01_b0.03_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTKL_EAAI":"CDKT_fmnist_I100_sTrue_fFalse_a0.04_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmKL_eaaiTrue.h5",
        "CDKTN_EAAI":"CDKT_fmnist_I100_sTrue_fFalse_a0.09_b0.05_RFFalse_SSFalse_accFalse_gmNorm2_lmNorm2_eaaiTrue.h5",
        "CDKTJSD_EAAI":"CDKT_fmnist_I100_sTrue_fFalse_a0.15_b0.1_RFFalse_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "CDKTRep_50_05":"CDKT_Rep_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_50_05":"CDKT_RepFull_fmnist_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_50_05": "CDKT_Full_fmnist_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRep_100_02": "CDKT_Rep_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_100_02": "CDKT_RepFull_fmnist_srate0.2_numclient100_I200_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_100_02": "CDKT_Full_fmnist_srate0.2_numclient100_I200_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",



        }
    # plot_metric_fixed()
    # plot_ICOIN2022()
    # plot_global_loss()
    # plot_local_loss()
    # plot_fixed_users_EAAI()
    # plot_subset_users_EAAI()
    # plot_all_figs()
    # plot_same_vs_hetero_fixed()
    # plot_same_vs_hetero_subset()
    # plot_alpha_effect_fixed()
    # plot_beta_effect_fixed()
    # # plot_fixed_users()
    # # plot_subset_users()
    # plot_alpha_effect_subset()
    # plot_beta_effect_subset()
    # plot_KSC()
    # scalability1()
    scalability2()
    # #### PLOT CIFAR10 #####
    print("----- PLOT CIFAR10 ------")
    DATASET = "Cifar-10"
    NUM_GLOBAL_ITERS = 200
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7
    name = {
        "fedavgfixed": "fedavg_Cifar10_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.015_b0.08_RFFalse_SSFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.015_b0.08_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLN": "CDKT_Cifar10_I100_sTrue_fTrue_a0.015_b0.08_RFTrue_SSFalse_gmKL_lmNorm2.h5",
        "fedavgsub": "fedavg_Cifar10_I100_sTrue_fFalse_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepsKLN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.4_b0.05_RFFalse_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.4_b0.05_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTfullsKLN": "CDKT_Cifar10_I100_sTrue_fTrue_a0.4_b0.05_RFTrue_SSTrue_gmKL_lmNorm2.h5",
        "CDKTrepfullfKL": "CDKT_Cifar10_I100_sTrue_fFalse_a0.03_b0.08_RFTrue_SSFalse_gmKL_lmKL.h5",
        "CDKTrepfullfN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.015_b0.04_RFTrue_SSFalse_gmNorm2_lmNorm2.h5",
        "CDKTrepfullfJ": "CDKT_Cifar10_I100_sTrue_fTrue_a0.1_b0.08_RFTrue_SSFalse_gmJSD_lmJSD.h5",
        "CDKTrepsKL": "CDKT_Cifar10_I100_sTrue_fFalse_a1_b0.05_RFTrue_SSTrue_gmKL_lmKL.h5",
        "CDKTrepsN": "CDKT_Cifar10_I100_sTrue_fFalse_a0.4_b0.01_RFTrue_SSTrue_gmNorm2_lmNorm2.h5",
        "CDKTrepsJ": "CDKT_Cifar10_I100_sTrue_fTrue_a2_b0.5_RFTrue_SSTrue_gmJSD_lmJSD.h5",
        "CDKTnotransferF": "CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTnotransferS":"CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepHfKLNnew": "CDKT_Cifar10_I100_sFalse_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLNnew": "CDKT_Cifar10_I100_sFalse_fFalse_a0.4_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "fedavg_KSC": "fedavg_Cifar10_I100_sTrue_fTrue_a0.02_b0.08_RFFalse_SSFalse_gmKL_lmKL.h5",
        "FedBD_KSC": "CDKT_Cifar10_I100_sTrue_fTrue_a0.045_b0.05_RFFalse_SSFalse_gmKL_lmKL.h5",
        # "FedBD_KSC_Adam": "CDKT_Cifar10_I100_sTrue_fTrue_a0.01_b0.05_RFFalse_SSFalse_gmKL_lmKL(Adam).h5"
        "FedBD_KSC_Adam": "CDKT_Cifar10_I100_sTrue_fTrue_a0.008_b0.03_RFFalse_SSFalse_gmKL_lmKL.h5",
        "fedavgfixednew": "fedavg_Cifar10_I100_sTrue_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLNnew": "CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLNnew": "CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLNnew": "CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "fedavgsubnew": "fedavg_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLNnew": "CDKT_Cifar10_I100_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLNnew": "CDKT_Cifar10_I100_sTrue_fFalse_a0.75_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullsKLNnew": "CDKT_Cifar10_I100_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedAvg_ICOIN": "fedavg_Cifar10_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedProx_ICOIN": "FedProx_Cifar10_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "FedLKD_ICOIN": "fedkc_Cifar10_I100_sTrue_fFalse_a0.02_b0.15_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "Scaffoldfixed": "Scaffold_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "Scaffoldsubset": "Scaffold_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONfixed": "MOON_Cifar10_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONsubset": "MOON_Cifar10_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSTrue_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "FedDynfixed": "FedDyn_Cifar10_I100_sTrue_fFalse_a0.8_b0.5_RFTrue_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "FedDynsubset": "FedDyn_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "CDKTKLN_EAAI":"CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTKL_EAAI": "CDKT_Cifar10_I100_sTrue_fFalse_a0.3_b0.03_RFTrue_SSFalse_accFalse_gmKL_lmKL_eaaiTrue.h5",
        "CDKTN_EAAI": "CDKT_Cifar10_I100_sTrue_fFalse_a0.1_b0.06_RFTrue_SSFalse_accFalse_gmNorm2_lmNorm2_eaaiTrue.h5",
        "CDKTJSD_EAAI": "CDKT_fmnist_I100_sTrue_fFalse_a0.15_b0.1_RFFalse_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue.h5",
        "CDKTRep_50_05": "CDKT_Rep_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_50_05":"CDKT_RepFull_Cifar10_srate0.5_numclient50_I100_sTrue_fFalse_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_50_05": "CDKT_Full_Cifar10_srate0.5_numclient50_I100_sTrue_fTrue_a0.2_b0.06_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRep_100_02": "CDKT_Rep_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_100_02": "CDKT_RepFull_Cifar10_srate0.2_numclient100_I200_sTrue_fFalse_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_100_02": "CDKT_Full_Cifar10_srate0.2_numclient100_I200_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",

        }
    # plot_fixed_users_EAAI()
    # plot_global_loss()
    # plot_local_loss()
    # plot_subset_users_EAAI()
    # plot_all_figs()
    # plot_fixed_users()
    # plot_subset_users()
    # plot_same_vs_hetero_fixed()
    # plot_same_vs_hetero_subset()
    # plot_alpha_effect2()
    # plot_beta_effect2()
    # plot_KSC()
    # plot_ICOIN2022()
    # scalability1()
    scalability2()
    # #### PLOT CIFAR100 #####
    print("----- PLOT CIFAR100 ------")
    DATASET = "Cifar-100"
    NUM_GLOBAL_ITERS = 200
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7
    name = {
        "CDKTnotransferF": "CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTnotransferS": "CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepHfKLNnew": "CDKT_Cifar100_I100_sFalse_fFalse_a0.05_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepHsKLNnew": "CDKT_Cifar100_I100_sFalse_fFalse_a0.4_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "fedavg_KSC": "fedavg_Cifar10_I100_sTrue_fTrue_a0.02_b0.08_RFFalse_SSFalse_gmKL_lmKL.h5",
        "FedBD_KSC": "CDKT_Cifar10_I100_sTrue_fTrue_a0.045_b0.05_RFFalse_SSFalse_gmKL_lmKL.h5",
        # "FedBD_KSC_Adam": "CDKT_Cifar10_I100_sTrue_fTrue_a0.01_b0.05_RFFalse_SSFalse_gmKL_lmKL(Adam).h5"
        "FedBD_KSC_Adam": "CDKT_Cifar10_I100_sTrue_fTrue_a0.008_b0.03_RFFalse_SSFalse_gmKL_lmKL.h5",
        "fedavgfixednew": "fedavg_Cifar100_I100_sTrue_fFalse_a0.5_b0.2_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfKLNnew": "CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullfKLNnew": "CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullfKLNnew": "CDKT_Cifar100_I100_sTrue_fTrue_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5",
        "fedavgsubnew": "fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepsKLNnew": "CDKT_Cifar100_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTrepfullsKLNnew": "CDKT_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "CDKTfullsKLNnew": "CDKT_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5",
        "Scaffoldfixed": "Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "Scaffoldsubset": "Scaffold_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONfixed": "MOON_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "MOONsubset": "MOON_Cifar100_I100_sTrue_fTrue_a0.2_b0.06_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "FedDynfixed": "FedDyn_Cifar100_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "FedDynsubset": "FedDyn_Cifar100_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue.h5",
        "CDKTKLN_EAAI": "CDKT_Cifar100_I100_sTrue_fFalse_a0.01_b0.02_RFTrue_SSFalse_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTKL_EAAI": "CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmKL_eaaiTrue_f1scoreTrue.h5",
        "CDKTN_EAAI": "CDKT_Cifar10_I100_sTrue_fFalse_a0.1_b0.06_RFTrue_SSFalse_accFalse_gmNorm2_lmNorm2_eaaiTrue.h5",
        "CDKTJSD_EAAI": "CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmJSD_lmJSD_eaaiTrue_f1scoreTrue.h5",
        "CDKTRep_50_05": "CDKT_Rep_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_50_05":"CDKT_RepFull_Cifar100_srate0.5_numclient50_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_50_05": "CDKT_Full_Cifar100_srate0.5_numclient50_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRep_100_02": "CDKT_Rep_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTRepFull_100_02": "CDKT_RepFull_Cifar100_srate0.2_numclient100_I200_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",
        "CDKTFull_100_02": "CDKT_Full_Cifar100_srate0.2_numclient100_I200_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2_eaaiTrue_f1scoreTrue.h5",

    }
    # plot_global_loss()
    # plot_local_loss()
    # plot_fixed_users_EAAI()
    # plot_subset_users_EAAI()
    # plot_all_figs()
    # plot_fixed_users()
    # plot_subset_users()
    # plot_same_vs_hetero_fixed()
    # plot_same_vs_hetero_subset()
    # plot_alpha_effect2()
    # plot_beta_effect2()
    # plot_KSC()
    # scalability1()
    scalability2()
    plt.show()

