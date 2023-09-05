import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings,get_savings_ind
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d, f_oneway
import matplotlib as mpl
import os
from scipy.stats import f_oneway, ttest_ind_from_stats, ttest_ind
from sklearn.decomposition import PCA

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

showall = False
if showall:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl','sym_0.5cb','res_0.5*Truesl']  # LR, EH, Sym
    #modellegend = ['ActorCritic','Sym. NAVIGATE', 'Neural NAVIGATE', 'AC+Sym. NAVGATE', 'AC+Neural NAVIGATE']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural','AC+Sym','AC+Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange', 'magenta','limegreen']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']

pltfig = 2

if pltfig == 1:
    def imshow_values(mat,val,ax=None,fz=6):
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                v = val[y, x]
                if mat[y,x]<-0.05:
                    k = 'w'
                else:
                    k = 'k'
                if ax is not None:
                    ax.text(x, y, v,
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=fz,color=k)
                else:
                    plt.text(x, y, v,
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=fz,color=k)

    path = 'C:/Users/ganesh_local/PycharmProjects/schema4one/coord/pc_shuffle_t300'
    [allxytrue, allxyest, allxyw, alltd,allpcidx]= saveload('load', path, 1)

    titles = ['Place cell selectivty left to right, top to bottom','Sorted place cell selectivity after shuffling']
    maze = ['PS', 'NM']

    weights = []

    for n in range(2):
        for w in range(3):
            if n ==1:
                prexshuf = allxyw[n][w]
                idx = allpcidx[n]
                weight = np.empty_like(prexshuf)
                weight[idx] = prexshuf
            else:
                weight = allxyw[n][w]

            if n == 0 and w == 2:
                weight = allxyw[n][w]
                weights.append(weight)
            elif n == 1 and w == 0:
                weight = allxyw[0][2]
                #prexshuf = allxyw[n][w]
                idx = allpcidx[n]
                #weight = np.empty_like(prexshuf)
                weight = weight[idx]

                weights.append(weight)
            elif n == 1 and w == 2:
                prexshuf = allxyw[n][w]
                idx = allpcidx[n]
                weight = np.empty_like(prexshuf)
                weight[idx] = prexshuf

                weights.append(weight)

    ylabels = ['OPA Trial 20','NM Trial 0', 'NM Trial 20']
    titles = ['After training in\noriginal environment','Shuffle place cell selectivity\nin new environment','After relearning\nin new environment']
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4))
    f.suptitle('Relearning metric representations after place cells remap',fontsize=10)
    for w,weight in enumerate(weights):
        if w == 0:
            idx = allpcidx[0].reshape(7,7)
        else:
            idx = allpcidx[1].reshape(7,7)

        mat = np.reshape(weight[:, 0], (7, 7))
        im = ax[0, w].imshow(mat)
        cbar = plt.colorbar(im, ax=ax[0, w], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        imshow_values(mat,idx, ax=ax[0, w])

        mat2 = np.reshape(weight[:, 1], (7, 7))
        im2 = ax[1, w].imshow(mat2)
        cbar2 = plt.colorbar(im2, ax=ax[1, w], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=6)
        imshow_values(mat2,idx, ax=ax[1, w])

        if w == 0:
            col = 'black'
        else:
            col = 'magenta'

        for axis in ['top', 'bottom', 'left', 'right']:
            ax[0, w].spines[axis].set_linewidth(3)  # change width
            ax[0, w].spines[axis].set_color(col)

            ax[1, w].spines[axis].set_linewidth(3)  # change width
            ax[1, w].spines[axis].set_color(col)

        ax[1, w].set_xlabel('Arena X axis', fontsize=8)

        ax[0, w].set_xticks([])
        ax[0, w].set_yticks([])
        ax[1, w].set_xticks([0,3,6],[-0.8,0,0.8], fontsize=6)
        ax[1, w].set_yticks([])
        ax[0, w].set_title(f'{titles[w]}', fontsize=8)

        if w == 0:
            ax[0, w].set_ylabel('$W^{coord} to X neuron$ \n Arena Y axis', fontsize=8)
            ax[1, w].set_ylabel('$W^{coord} to Y neuron$ \n Arena Y axis', fontsize=8)

            ax[0, w].set_yticks([0,3,6],[0.8,0,-0.8], fontsize=6)
            ax[1, w].set_yticks([0,3,6],[0.8,0,-0.8], fontsize=6)

    f.tight_layout()
    savefig('./Fig/coord/pc_shuffle_t300'.format(),f)

    # plt.figure()
    # for n in range(2):
    #     plt.plot(np.arange(1,21)+20*n, np.mean(alltd[n],axis=1))

elif pltfig ==2:
    alltau = [20,100,200,1000]
    [alltd] = saveload('load',f'../coord/alltds_tau1000.pickle',1)
    alltd = np.array(alltd)  # taus x btstp x n
    btstp = 20

    taus = [str(tau)+' ms' for tau in alltau]
    colors = ['black','tab:blue', 'tab:red','tab:green']
    f = plt.figure(figsize=(4,3))
    xs = [np.arange(1, 20 + 1), np.arange(20 + 1, 20 * 2 + 1)]
    for t, tau in enumerate(alltau):
        for n in range(2):
            if n == 0:
                legtau = taus[t]
            else:
                legtau = None
            td = np.mean(alltd[t,:,n],axis=0)
            var = np.std(alltd[t,:,n],axis=0)/np.sqrt(btstp)

            plt.plot(xs[n], td, color=colors[t],marker='o', label=legtau)
            plt.gca().fill_between(xs[n], td-var, td+var, alpha=0.2, facecolor=colors[t])

    plt.axvline(x=20.5,color='magenta',linestyle='--')
    plt.text(x=21,y=np.max(alltd)*0.9,s='Place cells remap in new environment', color='magenta')
    plt.title('Path integration TD error')
    plt.ylabel('Average TD error')
    plt.xlabel('Trial')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(loc=3, frameon=False)

    savefig('alltau_shuffle',f)


elif int(pltfig) == 5:
    #path = 'C:/Users/Razer/PycharmProjects/gradual4one/gen_fig/Data/dmp/coord_xy_state_dmp2.pickle'
    #[estxy] = saveload('load', path, 1)
    path = 'C:/Users/Razer/PycharmProjects/gradual4one/learn_coord/Data/vars3_learncoord_noneobs_1000tau_b1.pickle'
    [allstate, allxy, allxyw, allxyerror, stdxyerror]= saveload('load', path, 1)

    # mpl.rcParams['axes.spines.right'] = False
    # mpl.rcParams['axes.spines.top'] = False

    # f,ax = plt.subplots(2,3, figsize=(6,3))
    # for i,p in enumerate([1,3,9]):
    #     #coord = np.array(estxy[str(p)])
    #     #xy = coord[:, :2]
    #     #state = coord[:, 2:]
    #     ax[0,i].plot(state[:,0],state[:,1],color='k',zorder=1)
    #     ax[0, i].scatter(state[0,0],state[0,1], color='k', marker='s',zorder=2)
    #     ax[0, i].scatter(state[-1, 0], state[-1, 1], color='k', marker='X',zorder=2)
    #     #ax[0, i].set_xticks([])
    #     #ax[0, i].set_yticks([])
    #
    #     ax[1,i].plot(xy[:,0],xy[:,1],color='r',zorder=1)
    #     ax[1, i].scatter(xy[0, 0], xy[0, 1], color='r',marker='s',zorder=2)
    #     ax[1, i].scatter(xy[-1, 0], xy[-1, 1], color='r', marker='X',zorder=2)
    #     #ax[1, i].set_xticks([])
    #     #ax[1, i].set_yticks([])
    # ax[0,0].set_ylabel('True')
    # ax[1, 0].set_ylabel('Estimated')
    # f.tight_layout()
    # f.savefig('./Fig/dmp/dmp_est_coord2.png'.format())
    # f.savefig('./Fig/dmp/dmp_est_coord2.svg'.format())

    mpl.rcParams.update({'font.size': 8})
    clen = 10*1000//20
    ms = 20
    f,ax = plt.subplots(4,3, figsize=(4,4))
    trials = [2,9,20]
    for i in range(3):
        #coord = np.array(estxy[str(p)])
        #xy = coord[:, :2]
        #state = coord[:, 2:]

        im = ax[0,i].imshow(allxyw[0][i][:,0].reshape(7,7))
        cbar = plt.colorbar(im,ax=ax[0,i],fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        im2 = ax[1, i].imshow(allxyw[0][i][:, 1].reshape(7,7))
        cbar = plt.colorbar(im2,ax=ax[1,i],fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

        state = allstate[0][i,:clen]
        xy = allxy[0][i,:clen]
        ax[2,i].plot(state[:,0],state[:,1],color='k',zorder=1, linewidth=1)
        ax[2, i].scatter(state[0,0],state[0,1], color='k', marker='s',zorder=2,s=ms)
        ax[2, i].scatter(state[-1, 0], state[-1, 1], color='k', marker='X',zorder=2,s=ms)
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])

        ax[2, i].spines.right.set_visible(False)
        ax[2, i].spines.top.set_visible(False)
        #ax[2, i].set_aspect('equal', adjustable='box')
        ax[2,i].axis('square')

        ax[3,i].plot(xy[:,0],xy[:,1],color='r',zorder=1, linewidth=1)
        ax[3, i].scatter(xy[0, 0], xy[0, 1], color='r',marker='s',zorder=2,s=ms)
        ax[3, i].scatter(xy[-1, 0], xy[-1, 1], color='r', marker='X',zorder=2,s=ms)
        ax[3, i].set_xticks([])
        ax[3, i].set_yticks([])
        ax[0,i].set_title('Trial {}'.format(trials[i]))

        ax[3,i].spines.right.set_visible(False)
        ax[3,i].spines.top.set_visible(False)
        ax[3, i].set_aspect('equal', adjustable='box')
        ax[3, i].axis('square')

    ax[0,0].set_ylabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[2,0].set_ylabel('True')
    ax[3, 0].set_ylabel('Estimated')
    f.tight_layout()

    f.savefig('./Fig/dmp/xy_w_3.png'.format())
    f.savefig('./Fig/dmp/xy_w_3.svg'.format())


elif int(pltfig) == 6:
    path = 'C:/Users/Razer/PycharmProjects/gradual4one/learn_coord/Data/vars_random_forage_tderr_300t_3b.pickle'
    [allstate, allxy, allxyw, allxyerror, stdxyerror, alltd]= saveload('load', path, 1)

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    alltd = np.vstack(alltd)

    plt.figure(figsize=(4,3))
    plt.errorbar(np.arange(20),np.mean(alltd,axis=0),np.std(alltd,axis=0)/np.sqrt(3),color='k',marker='o')
    plt.xticks(np.linspace(0, 19,5,dtype=int),np.linspace(1, 20,5,dtype=int))
    plt.xlabel('Trial')
    plt.ylabel('TD MSE')
    plt.title('Path integration TD error')
    plt.tight_layout()

    plt.savefig('./Fig/dmp/random_forage_td_300t.png')
    plt.savefig('./Fig/dmp/random_forage_td_300t.svg')

    allxy = np.vstack(allxy)

    pca = PCA(n_components=0.95)
    xyt = pca.fit_transform(allxy[0].T)


