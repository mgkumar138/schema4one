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
    modellegend = ['Actor-Critic', 'Symbolic', 'Neural','AC+Sym','AC+Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange', 'magenta','limegreen']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['Actor-Critic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']


pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    #genvar_path = '../dmp/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/dmp/'
    totiter = 720

    for model in allmodels:
        model_data = glob.glob(genvar_path+'genvars*_dmp_*{}*'.format(model))
        print(len(model_data))

        totlat, totdgr = [], []
        for idx, data in enumerate(model_data):
            [lat, dgr, _] = saveload('load',data,1)

            totlat.append(lat)
            totdgr.append(dgr)

        totlat = np.concatenate(totlat,axis=0)
        totdgr = np.concatenate(totdgr,axis=0)

        if len(totdgr) > totiter:
            lidx = np.random.choice(np.arange(len(totdgr)), totiter, replace=False)
        else:
            lidx = np.arange(len(totdgr))

        saveload('save','./Data/dmp/comb_genvar_dmp_{}_{}b'.format(model[:7], len(lidx)), [totlat[lidx], totdgr[lidx]])

elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    scalet = 1000/1000

    genvar_path = './Data/dmp/'

    tl, td = [], []
    sl, sd = [], []
    isl = []
    dfm, dfs = [], []
    firstlat = []
    secondlat = []
    z = 1.96
    for idx, model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'comb_genvar_dmp_{}*'.format(model[:7]))

        [totlat, totdgr] = saveload('load',model_data[0],1)

        latdif = totlat[:, :, 0] - totlat[:, :, 1]
        latt, latp = ttest_1samp(latdif, popmean=0)

        firstlat.append(totlat[:,:,0])
        secondlat.append(totlat[:, :, 1])

        tl.append(np.mean(totlat*scalet, axis=0).reshape(-1))
        #tl.append(np.median(totlat * scalet, axis=0).reshape(-1))
        #td.append(np.array([np.nanquantile(totlat, 0.25, axis=0).reshape(-1),np.nanquantile(totlat, 0.75, axis=0).reshape(-1)]))
        td.append(z*np.array(np.std(totlat*scalet, axis=0)/np.sqrt(len(totlat))).reshape(-1))

        totsavingsm, totsavingss = get_savings(totlat*scalet)
        sl.append(totsavingsm)
        sd.append(z*totsavingss)
        isl.append(get_savings_ind(totlat*scalet))

        dfm.append(np.mean(totdgr, axis=0))
        dfs.append(z*np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))

    f1,ax1 = plt.subplots(1,1,figsize=(4,2))
    f2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    f3, ax3 = plt.subplots(1, 1, figsize=(4, 2))
    for idx in range(len(modellegend)):
        #ax1.plot(np.arange(1, 1 + totlat.shape[1]*totlat.shape[2]), tl[idx], color=modelcolor[idx])
        ax1.errorbar(x=np.arange(1, 1 + totlat.shape[1]*totlat.shape[2]), y=tl[idx],yerr=td[idx],color=modelcolor[idx])
        #ax1.fill_between(np.arange(1, 1 + totlat.shape[1]*totlat.shape[2]), td[idx][0], td[idx][1], alpha=0.1, facecolor=modelcolor[idx])
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title('Time to reach single target')
    ax1.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax1.set_xticks(np.arange(1, 45, step=5))

    for idx in range(len(modellegend)):
        #ax2.plot(np.arange(1, 1 + totsavingsm.shape[0]), sl[idx], color=modelcolor[idx])
        ax2.errorbar(x=np.arange(1, 1 + totsavingsm.shape[0]), y=sl[idx],yerr=sd[idx], color=modelcolor[idx])
        #ax2.fill_between(np.arange(1, 1 + totsavingsm.shape[0]), sd[idx][0], sd[idx][1], alpha=0.1, facecolor=modelcolor[idx])
    ax2.set_xlabel('Session')
    ax2.set_title('Latency savings T1-T2')
    ax2.set_ylabel('Savings (s)')
    #ax2.legend(modellegend, loc='upper left', fontsize=6, frameon=False)

    f_oneway(isl[2],isl[1])
    ttest_ind(isl[2][:,0],isl[1][:,0])

    tp = np.zeros([3,9,2])
    for m in range(3):
        for d in range(9):
            if m == 2:
                tp[2, d] = ttest_ind_from_stats(sl[1][d], sd[1][d], 480, sl[2][d], sd[2][d], 480)
            else:
                tp[m, d] = ttest_ind_from_stats(sl[m + 1][d], sd[m + 1][d], 480, sl[0][d], sd[0][d], 480)

    ttest_ind(np.mean(firstlat[1],axis=1),np.mean(firstlat[2],axis=1))
    ttest_ind(np.mean(secondlat[1], axis=1), np.mean(secondlat[2], axis=1))

    for idx in range(len(modellegend)):
        ax3.errorbar(x=np.arange(1, 1 + totdgr.shape[1]), y=dfm[idx],yerr=dfs[idx], color=modelcolor[idx]) #, marker='o', ms=5
        #ax3.plot(np.arange(1, 1 + totdgr.shape[1]), dfm[idx], color=modelcolor[idx])
        #ax3.fill_between(np.arange(1, 1 + totdgr.shape[1]), dfs[idx][0], dfs[idx][1], alpha=0.1, facecolor=modelcolor[idx])
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Time at target (%)')
    ax3.set_title('Time spent at target during probe trial')
    ax3.legend(modellegend, loc='upper left', fontsize=6, frameon=False)

    f1.tight_layout()
    f2.tight_layout()
    f3.tight_layout()

    savefig('./Fig/dmp/dmp_latency_leg',f1)
    savefig('./Fig/dmp/dmp_savings_noleg', f2)
    #savefig('./Fig/dmp/dmp_dgr', f3)

    # plt.figure()
    # for i in range(9):
    #     plt.subplot(3, 3, i + 1)
    #     plt.hist(totlat[:, i, 0])
    #     plt.hist(totlat[:, i, 1], alpha=0.5)
    #     plt.title(f'Session {i + 1}')

    # schema vs ac
    #f_oneway(sl)

    # f1.savefig('./Fig/dmp/dmp_latency_all{}.png'.format(showall))
    # f1.savefig('./Fig/dmp/dmp_latency_all{}.svg'.format(showall))
    # f2.savefig('./Fig/dmp/dmp_save_all{}.png'.format(showall))
    # f2.savefig('./Fig/dmp/dmp_save_all{}.svg'.format(showall))
    # f3.savefig('./Fig/dmp/dmp_dgr_all{}.png'.format(showall))
    # f3.savefig('./Fig/dmp/dmp_dgr_all{}.svg'.format(showall))

elif int(pltfig) == 3:
    #genvar_path = '../dmp/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/dmp/'

    N = 4
    allpath = np.zeros([len(allmodels), N, 9, 3001, 2])
    #allrandidx = np.array([93,68,:,:,:])
    for m,model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'genvars_*{}*'.format(model))

        a = saveload('load', model_data[-1],1)
        #allpath[m] = a[2]
        paths = a[2]
        dgr = a[1]
        latency = a[0]
        savings = latency[:,:,0]-latency[:,:,1]

        if m == 0:
            idx = np.arange(len(paths))[np.all(dgr[:,:1]<1,axis=1)*np.all(dgr[:,6:]<1,axis=1)]
            randidx = np.array(idx)[np.random.choice(np.arange(len(idx)), N, replace=False)]
        elif m ==1:
            idx = np.arange(len(paths))[np.all(dgr[:,:1]<1,axis=1)*np.all(dgr[:,6:]>5,axis=1)]
            randidx = np.array(idx)[np.random.choice(np.arange(len(idx)), N, replace=False)]
        else:
            idx = np.arange(len(paths))[np.all(dgr[:,:1]<1,axis=1)*np.all(dgr[:,6:]>5,axis=1)]
            randidx = np.array(idx)[np.random.choice(np.arange(len(idx)), N, replace=False)]

        print(model, randidx)
        allpath[m] = paths[randidx]


    # midx = np.linspace(0, 9 - 1, 3, dtype=int)
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.top'] = True

    for t in range(N):
        f, ax = plt.subplots(len(allmodels), 9, figsize=(7, 2.5))

        for m in range(len(allmodels)):
            for i in range(9):
                mvpath = allpath[m, t]
                ax[m, i].plot(mvpath[i, :-1, 0], mvpath[i, :-1, 1], linewidth=0.5, color=modelcolor[m])
                ax[m, i].scatter(mvpath[i, 0, 0], mvpath[i, 0, 1], color=modelcolor[m], zorder=2, marker='s', s=5)
                ax[m, i].scatter(mvpath[i, -2, 0], mvpath[i, -2, 1], color=modelcolor[m], zorder=2, marker='X', s=5)

                rloc = mvpath[i, -1]
                ax[m, i].axis([-0.8, 0.8, -0.8, 0.8])

                ax[m, i].set_aspect('equal', adjustable='box')
                circle = plt.Circle(rloc, 0.03, color='r', zorder=9)
                ax[m, i].add_artist(circle)
                circle2 = plt.Circle(rloc, 0.05, color='k', zorder=10, fill=False)
                ax[m, i].add_artist(circle2)

                ax[m, i].set_xticks([])
                ax[m, i].set_yticks([])
                if m == 0:
                    ax[m, i].set_title('PT{}'.format(i + 1), fontsize=8)
                if i == 0:
                    ax[m, i].set_ylabel('{}'.format(modellegend[m]), fontsize=8)
        f.tight_layout()
        savefig('./Fig/dmp/dmp_traj{}'.format(t),f)

elif int(pltfig) == 4:
    #genvar_path = '../dmp/Data/'
    genvar_path = 'D:/Ganesh_PhD/Schema4PA/dmp/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    hp = get_default_hp(task='dmp', platform='laptop')

    for m,model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'vars_dmp_*{}*'.format(model))
        N = len(model_data)
        print(N)

        if N != 0:
            policy = np.zeros([9, 2,15,15])
            value = np.zeros([9, 15,15])

            for d in range(2):

                [alldyn, pweights, mvpath, lat, dgr] = saveload('load',model_data[d],1)

                # qdyn = alldyn[1]
                # cdyn = alldyn[2]
                # nonrlen = np.array(qdyn[list(qdyn.keys())[0]]).shape[0]
                # bins = 15
                # cues = 1
                #
                # sess = list(cdyn.keys())
                # for p in range(9):
                #     qfr = np.zeros([cues, nonrlen, 40])
                #     cfr = np.zeros([cues, nonrlen, 2])
                #     coord = np.zeros([nonrlen * cues, 2])
                #     newx = np.zeros([225, 2])
                #     for i in range(15):
                #         st = i * 15
                #         ed = st + 15
                #         newx[st:ed, 0] = np.arange(15)
                #     for i in range(15):
                #         st = i * 15
                #         ed = st + 15
                #         newx[st:ed, 1] = i * np.ones(15)
                #
                #     qfr[0] = np.array(qdyn[sess[p]])[-nonrlen:]
                #     cfr[0] = np.array(cdyn[sess[p]])[-nonrlen:]
                #
                #     qfr = np.reshape(qfr, newshape=(cues * nonrlen, 40))
                #     cfr = np.reshape(cfr, newshape=(cues * nonrlen, 2))
                #
                #     coord = mvpath[p][:nonrlen]
                #
                #     policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr)
                #
                #     policy[p] += policymap
                #     value[p] += valuemap


                idx = [0,2,8]
                f,ax = plt.subplots(2,3,figsize=(6,3))
                for p in range(3):
                    coordw = pweights[idx[p]][0]

                    im = ax[0,p].imshow(np.array(coordw[:,0]).reshape(7,7), aspect='auto')
                    plt.colorbar(im,ax=ax[0,p], fraction=0.046, pad=0.04)
                    ax[0,p].set_xticks([])
                    ax[0,p].set_yticks([])
                    ax[0,p].set_aspect('equal', adjustable='box')
                    ax[0, p].set_title('PT{}'.format(idx[p]+1))

                    im = ax[1, p].imshow(np.array(coordw[:, 1]).reshape(7,7), aspect='auto')
                    plt.colorbar(im,ax=ax[1,p], fraction=0.046, pad=0.04)
                    ax[1,p].set_xticks([])
                    ax[1,p].set_yticks([])
                    ax[1,p].set_aspect('equal', adjustable='box')

                ax[0, 0].set_ylabel('X')
                ax[1,0].set_ylabel('Y')
                f.tight_layout()

                f.text(0.01,0.01,model[:7])
                f.savefig('./Fig/dmp/dmp_coord_{}_{}_b{}.png'.format(model[:7],p,N))
                f.savefig('./Fig/dmp/dmp_coord_{}_{}_b{}.svg'.format(model[:7],p, N))
                #plt.close()

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








