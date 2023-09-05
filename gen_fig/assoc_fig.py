import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d, f_oneway, ttest_ind
import matplotlib as mpl
import os

pltfig = 3

# LMS vs EH

if int(pltfig) == 1:
    #genvar_path = 'D:/Ganesh_PhD/s4o/assoc/'
    genvar_path = './Data/assoc/'

    [lmserr, lmstrackg] = saveload('load', genvar_path + 'vars_res_phia_Falsesl_200c_24b', 1) # res lms
    #[lmserr2, _] = saveload('load', genvar_path + 'vars_res_phia_Falsesl_200c_24b_1.pickle', 1) # res lms
    #lmserr = np.concatenate([lmserr, lmserr2],axis=1)

    [eherr, ehtrackg] = saveload('load', genvar_path + 'vars_res_phia_Truesl_200c_24b', 1)  # res eh
    #[eherr2, _] = saveload('load', genvar_path + 'vars_res_phia_Truesl_200c_24b.pickle', 1)  # res eh
    #eherr = np.concatenate([eherr, eherr2],axis=1)

    [ffeherr, _] = saveload('load', genvar_path + 'vars_ff_relu_Truesl_200c_24b', 1)  # ff eh
    #[ffeherr2, _] = saveload('load', genvar_path + 'vars_ff_relu_Truesl_200c_24b_1.pickle', 1)  # ff eh
    #ffeherr = np.concatenate([ffeherr, ffeherr2],axis=1)

    [fflmserr, _] = saveload('load', genvar_path + 'vars_ff_relu_Falsesl_200c_24b', 1)  # ff eh
    #[ffeherr2, _] = saveload('load', genvar_path + 'vars_ff_relu_Truesl_200c_24b_1.pickle', 1)  # ff eh
    #ffeherr = np.concatenate([ffeherr, ffeherr2],axis=1)

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    f,p=f_oneway(eherr[3, :, 1], ffeherr[3, :, 1],lmserr[3, :, 1], axis=0)
    print('p val: ',np.mean(p))

    idx = 0
    lb = 0
    up = 107
    t, p = ttest_ind(np.trapz(fflmserr[idx, :, 1, lb:up]), np.trapz(lmserr[idx, :, 1, lb:up]))
    print(t, p)

    idx = 3
    lb = 0
    up = 200
    t, p = ttest_ind(np.trapz(lmserr[idx, :, 1, lb:up]), np.trapz(eherr[idx, :, 1, lb:up]))
    print(t, p)

    #allN = 2 ** np.arange(7, 12)
    #z = 1.96/np.sqrt(48)
    ncues = 200
    f = plt.figure(figsize=(4,2))
    ax = plt.subplot(111)

    #ax.plot(np.arange(1, ncues + 1), np.mean(eherr[3, :, 1], axis=0))  # res, eh, 1024
    # ax.plot(np.arange(1, ncues + 1), np.mean(eherr[0, :, 1], axis=0))  # res, eh, 128
    # ax.plot(np.arange(1, ncues + 1), np.mean(eherr[1, :, 1], axis=0))  # res, eh, 256
    # ax.plot(np.arange(1, ncues + 1), np.mean(eherr[3, :, 1], axis=0))  # res, eh, 512

    ax.plot(np.arange(1, ncues + 1), np.mean(ffeherr[0, :, 1], axis=0), color='tab:blue', linewidth=1,label='FF_128_EH')  # ff, EH, 128
    ax.plot(np.arange(1, ncues + 1), np.mean(fflmserr[0, :, 1], axis=0), color='tab:pink', linewidth=1,label='FF_128_LMS')  # res, lms, 128

    ax.plot(np.arange(1, ncues + 1), np.mean(eherr[0, :, 1], axis=0), color='tab:orange',linewidth=1, label='Res_128_EH') # res, eh, 128
    ax.plot(np.arange(1, ncues + 1), np.mean(lmserr[0, :, 1], axis=0), color='tab:green', linewidth=1,label='Res_128_LMS')  # res, lms, 128

    ax.plot(np.arange(1, ncues + 1), np.mean(ffeherr[3, :, 1], axis=0), color='tab:red', linewidth=1,label='FF_1024_EH')  # ff, EH, 1024
    ax.plot(np.arange(1, ncues + 1), np.mean(fflmserr[3, :, 1], axis=0), color='tab:gray', linewidth=1,label='FF_1024_LMS')  # ff, EH, 1024

    ax.plot(np.arange(1, ncues + 1), np.mean(eherr[3, :, 1], axis=0), color='tab:purple', linewidth=1, label='Res_1024_EH')
    ax.plot(np.arange(1, ncues + 1), np.mean(lmserr[3, :, 1], axis=0), color='tab:brown', linewidth=1,label='Res_1024_LMS')  # res, lms, 1024

    #plt.set_ylim(0, 300)
    #plt.xlim(0, 90)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])
    ax.legend(fontsize=6,frameon=False,loc='center left',bbox_to_anchor=(1, 0.5))

    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(ffeherr[0, :, 1],0.25, axis=0),
                     y2=np.quantile(ffeherr[0, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:blue')
    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(fflmserr[0, :, 1],0.25, axis=0),
                     y2=np.quantile(fflmserr[0, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:pink')

    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(eherr[0, :, 1],0.25, axis=0),
                     y2=np.quantile(eherr[0, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:orange')
    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(lmserr[0, :, 1],0.25, axis=0),
                     y2=np.quantile(lmserr[0, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:green')

    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(ffeherr[3, :, 1],0.25, axis=0),
                     y2=np.quantile(ffeherr[3, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:red')
    plt.fill_between(x=np.arange(1, ncues + 1), y1=np.quantile(fflmserr[3, :, 1], 0.25, axis=0),
                     y2=np.quantile(fflmserr[3, :, 1], 0.75, axis=0), alpha=0.1, facecolor='tab:gray')

    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(eherr[3, :, 1],0.25, axis=0),
                     y2=np.quantile(eherr[3, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:purple')
    plt.fill_between(x=np.arange(1, ncues+1),  y1=np.quantile(lmserr[3, :, 1],0.25, axis=0),
                     y2=np.quantile(lmserr[3, :, 1], 0.75,axis=0), alpha=0.1,facecolor='tab:brown')

    #plt.xticks(np.arange(0,60,10),np.arange(0,60,10))
    plt.xlabel('Number of cues')
    plt.ylabel('Recall MSE')
    plt.title('One-shot association capacity')
    ax.set_xticks(np.linspace(1,200,5,dtype=int),np.linspace(1,200,5,dtype=int))
    plt.tight_layout()
    #savefig('./Fig/assoc/cues_vs_N_relu',f)

    # f.savefig('./Fig/assoc_memory/cues_vs_N.png')
    # f.savefig('./Fig/assoc_memory/cues_vs_N.svg')

   # from scipy.stats import pearsonr, f_oneway, ttest_ind, linregress

    # units
    #fstat = f_oneway(eherr[0,:,1],eherr[1,:,1],eherr[2,:,1],eherr[3,:,1],eherr[4,:,1])

    #LMS vs EH
    #eh_lms = pearsonr(np.mean(eherr[3, :, 1],axis=0), np.mean(lmserr[0,:,1],axis=0))
    #eh_lms_f = f_oneway(eherr[3, :, 0], lmserr[0, :, 0])

    # pas = ttest_ind(eherr[3, :, 1,9],eherr[3, :, 1,-1],equal_var=False)
    # mdl = linregress(eherr[3, :, 1])
    #linpa = np.polyfit(,deg=1)

elif int(pltfig) == 2:
    [allerr, trackg] = saveload('load','../assoc/vars_EH_Ach_ach_4cues',1)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    f = plt.figure(figsize=(8,3.25))
    f.text(s='Cue 1', x=0.24, y=0.96, fontsize=10)
    f.text(s='Cue 2', x=0.73, y=0.96, fontsize=10)
    f.text(s='Cue 3', x=0.24, y=0.48, fontsize=10)
    f.text(s='Cue 4', x=0.73, y=0.48, fontsize=10)
    col = ['tab:blue','tab:orange','tab:green']
    trials = [1,5,2,6,3,7,4,8]
    j=1
    for c in range(4):

        for p in range(2):
            plt.subplot(2,4,j)

            plt.xticks([100, 350], [2, 7])
            plt.yticks([0, 1])
            plt.ylim([-0.8, 1.5])
            if p == 0:
                plt.title('Learning (Trial {})'.format(trials[j-1]), fontsize=8)
                plt.axvline(100, ymax=1, color='red', linestyle='--')
                plt.axvline(350, ymax=1, color='k', linestyle='--')
            else:
                plt.title('Recall & Deletion (Trial {})'.format(trials[j-1]), fontsize=8)
                #plt.axis('off')
                plt.gca().spines[['left']].set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                if c == 0 or c == 1 or c == 2:
                    plt.axvline(100, ymax=1, color='magenta', linestyle='--')
                    plt.axvline(350, ymax=1, color='k', linestyle='--')
                    if c == 0:
                        plt.text(150, 1.2, 'Ach=0.1')
                    elif c == 1:
                        plt.text(130, 1.2, 'Ach=0.01')
                    elif c == 2:
                        plt.text(115, 1.2, 'Ach=0.001')

            for i in range(3):
                plt.plot(trackg[c,p, :,i],alpha=1,color=col[i])
            j+=1

            #plt.hlines(y=1.3, xmin=30, xmax=180, color='k')
            #plt.text(90, 1.4, 'Learning')
    plt.tight_layout()
    plt.show()

    savefig('./Fig/assoc/learn_unlearn_ach',f)

    # f = plt.figure(figsize=(8,3))
    # col = ['tab:blue','tab:orange','tab:green']
    # for c in range(4):
    #     plt.subplot(2,2,c+1)
    #     for i in range(3):
    #         #plt.plot(trackg[c,:,i+3],alpha=0.2,color=col[i])
    #         plt.plot(trackg[c, :,i],alpha=1,color=col[i])
    #         #plt.xticks([])
    #         plt.xticks([0,250,550,800,1000], [0,5,0,4,10])
    #         plt.yticks([0,1])
    #
    #     plt.ylim([-0.5,1.5])
    #     plt.hlines(y=1.3,xmin=30,xmax=180,color='k')
    #     plt.text(90,1.4,'3s')
    #     plt.title('Cue {}'.format(c+1))
    #     plt.axvline(10, ymax=0.9, color='red', linestyle='--')
    #     plt.axvline(238,ymax=0.9,color='k',linestyle='--')
    #     if c == 0 or c == 1 or c==2:
    #         plt.axvline(650,ymax=0.9, color='magenta', linestyle='--')
    #         plt.axvline(900,ymax=0.9, color='k', linestyle='--')
    #         if c == 0:
    #             plt.text(650, 1.4, 'Ach=0.1')
    #         elif c == 1:
    #             plt.text(650, 1.4, 'Ach=0.01')
    #         elif c == 2:
    #             plt.text(650, 1.4, 'Ach=0.001')
    # f.tight_layout()
    #
    # savefig('./Fig/assoc/learn_unlearn_ach',f)

elif int(pltfig) == 3:
    from matplotlib.gridspec import GridSpec
    mpl.rcParams.update({'font.size': 8})
    sz = 20
    query = np.eye(sz) * 3
    coord = 2*np.random.uniform(size=[sz,2])-1
    keyvalue = np.concatenate([query[:,:18],coord,np.ones([sz,1])],axis=1)
    delmem = np.array([3,7,8,10,12,15,17, 18,19])
    keyvalue[delmem] = 0

    f = plt.figure(figsize=(3.5,2))
    gs = GridSpec(1, 3, figure=f, wspace=0.01)
    ax1 = f.add_subplot(gs[:2])
    ax2 = f.add_subplot(gs[2:])

    im = ax1.imshow(keyvalue[:,:18])
    plt.colorbar(im,ax=ax1,ticks=np.linspace(0,3,2),fraction=0.046, pad=0.04)
    ax1.set_title('Key matrix', fontsize=8)
    ax1.set_yticks(np.arange(0,20,5),np.arange(1,21,5))
    ax1.set_ylabel('Memory index', fontsize=8)
    ax1.set_xlabel('Sensory cue activity', fontsize=8)
    ax1.set_xticks([])

    im2 = ax2.imshow(keyvalue[:, 18:],vmin=-1,vmax=1)
    plt.colorbar(im2,ax=ax2,ticks=np.linspace(-1,1,3))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Value matrix', fontsize=8)
    ax2.set_xlabel("X | Y | $\\theta(R)$",x=0.75)
    #ax2.set_xticks(np.arange(3))
    #ax2.set_xticklabels(['X','Y',r'$\theta(R)$'], rotation=90, fontsize=8)
    #ax2.set_xticklabels(['', '', ''], rotation=0, fontsize=1)
    #f.suptitle('Symbolic Memory', fontsize=8,x=0.6, y=0.9)
    f.tight_layout()

    savefig('./Fig/assoc/sym_mem_{}'.format(sz),f)

elif int(pltfig) == 4:
    [_, ehag,ehrg] = saveload('load','../assoc_memory/Data/vars_eh_pc.pickle',1)
    [_, lmsag,lmsrg] = saveload('load', '../assoc_memory/Data/vars_lms_pc.pickle', 1)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})


    col = ['tab:blue','tab:orange','tab:green']
    nantg = np.zeros([50,3])
    nantg[nantg==0] = np.nan
    combehtg = np.concatenate([ehag,nantg,ehrg],axis=0)
    comblmstg = np.concatenate([lmsag, nantg, lmsrg], axis=0)

    f = plt.figure(figsize=(4,3))
    plt.plot(comblmstg[:, -1],color='blue')
    plt.plot(combehtg[:, -1], color='tab:green')
    plt.axvline(50,color='r',linestyle='--')
    plt.axvline(277, color='k', linestyle='--')
    plt.xticks([])
    plt.hlines(y=0, xmin=100, xmax=250, color='k')
    plt.text(175, 0.1, '3s')
    #plt.axhline(0.6,color='tab:purple',linestyle='--')
    plt.legend(['LMS','EH'],loc=4)
    plt.title('Recall value with different place cell input')
    plt.ylabel('Recall firing rate')
    plt.xlabel('Timestep')
    plt.tight_layout()

    f.savefig('./Fig/assoc_memory/eh_vs_lms_gbar.png')
    f.savefig('./Fig/assoc_memory/eh_vs_lms_gbar.svg')

if int(pltfig) == 5:
    #genvar_path = 'D:/Ganesh_PhD/Schema4PA/assoc_memory/'
    genvar_path = './Data/assoc/'

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    f,ax = plt.subplots(3,2,figsize=(8,6))
    ax = ax.flatten()
    allN = 2 ** np.arange(7, 12)
    ncues = 200
    z = 1.96 / np.sqrt(24)
    labels = ['Feedforward Relu LMS', 'Feedforward Relu EH', 'Reservoir Relu LMS', 'Reservoir Relu EH', 'Reservoir Phia LMS', 'Reservoir Phia EH']
    types = ['ff_relu_Falsesl','ff_relu_Truesl','res_relu_Falsesl','res_relu_Truesl','res_phia_Falsesl','res_phia_Truesl']
    for n in range(6):

        [err, trackg] = saveload('load', genvar_path + 'vars_{}_200c_24b.pickle'.format(types[n], 1),1)

        for i in range(5):
            if n == 3 and i == 4:
                pass
            else:
                ax[n].plot(np.arange(1, ncues + 1), np.mean(err[i, :, 1], axis=0), linewidth=1, label=allN[i])  # ff, EH, 1024
                ax[n].fill_between(x=np.arange(1, ncues + 1), y1=np.quantile(err[i, :, 1], 0.25, axis=0),
                                 y2=np.quantile(err[i, :, 1], 0.75, axis=0), alpha=0.1)

        ax[n].legend(frameon=False, fontsize=5)

        #box = ax[n].get_position()
        #ax[n].set_position([box.x0, box.y0, box.width * 1, box.height])
        #ax[n].legend(['N=128', 'N=256', 'N=512', 'N=1024', 'N=2048'], fontsize=6, frameon=False, loc='center left',                     bbox_to_anchor=(1, 0.5))

        #plt.xticks(np.arange(0,60,10),np.arange(0,60,10))
        ax[n].set_xlabel('Number of cues')
        ax[n].set_ylabel('Recall MSE')
        ax[n].set_title(labels[n])
        ax[n].set_xticks(np.linspace(1,200,5,dtype=int),np.linspace(1,200,5,dtype=int))
    #ax[-1].set_axis_off()
    #ax[-1].legend(['FF Relu LMS', 'FF Relu EH', 'Res Relu LMS', 'Res Relu EH', 'Res Phia LMS', 'Res Phia EH'],
    #             fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    #handles, _ = ax[0].get_legend_handles_labels()
    #f.legend(handles, labels, loc=0)
    f.tight_layout()
    savefig('./Fig/assoc/cues_vs_N_network_rule',f)

if int(pltfig) == 6:
    #genvar_path = 'D:/Ganesh_PhD/Schema4PA/assoc_memory/'
    genvar_path = './Data/assoc/'

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    f,ax = plt.subplots(3,2,figsize=(8,6))
    ax = ax.flatten()
    allN = 2 ** np.arange(7, 12)
    labels = ['FF Relu LMS', 'FF Relu EH', 'Res Relu LMS', 'Res Relu EH', 'Res Phia LMS', 'Res Phia EH']
    for i,mtype in enumerate(['ff_relu_Falsesl','ff_relu_Truesl','res_relu_Falsesl','res_relu_Truesl','res_phia_Falsesl','res_phia_Truesl']):

        [err, trackg] = saveload('load', genvar_path + 'vars_{}_200c_24b.pickle'.format(mtype), 1)

        z = 1.96/np.sqrt(24)
        ncues = 200

        for n in range(5):
            ax[n].plot(np.arange(1, ncues + 1), np.mean(err[n, :, 1], axis=0), linewidth=1, label=labels[i])  # ff, EH, 1024
            ax[n].fill_between(x=np.arange(1, ncues + 1), y1=np.quantile(err[n, :, 1], 0.25, axis=0),
                             y2=np.quantile(err[n, :, 1], 0.75, axis=0), alpha=0.1)

    for n in range(5):
        #box = ax[n].get_position()
        #ax[n].set_position([box.x0, box.y0, box.width * 1, box.height])
        #ax[n].legend(['N=128','N=256','N=512','N=1024','N=2048'], fontsize=6,frameon=False,      loc='center left',bbox_to_anchor=(1, 0.5))

        #plt.xticks(np.arange(0,60,10),np.arange(0,60,10))
        ax[n].set_xlabel('Number of cues')
        ax[n].set_ylabel('Recall MSE')
        ax[n].set_title('N = {}'.format(allN[n]))
        ax[n].set_xticks(np.linspace(1,200,5,dtype=int),np.linspace(1,200,5,dtype=int))
    ax[-1].set_axis_off()
    #ax[-1].legend(['FF Relu LMS', 'FF Relu EH', 'Res Relu LMS', 'Res Relu EH', 'Res Phia LMS', 'Res Phia EH'],
    #             fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    handles, _ = ax[0].get_legend_handles_labels()
    f.legend(handles, labels, loc=0)
    f.tight_layout()
    #savefig('./Fig/assoc/cues_vs_N_relu',f)
