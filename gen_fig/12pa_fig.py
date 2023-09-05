import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d
import matplotlib as mpl
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

allsym = ['12pa_sym_1000N_0cb', '12pa_sym_1000N_1cb']
allffeh = ['12pa_ff_*128N*Truesl*', '12pa_ff_*256N*Truesl*', '12pa_ff_*512N*Truesl*', '12pa_ff_*1024N*Truesl*', '12pa_ff_*2048N*7.5e-06glr*Truesl*']
allfflms = ['12pa_ff_*128N*Falsesl*', '12pa_ff_*256N*Falsesl*', '12pa_ff_*512N*Falsesl*', '12pa_ff_*1024N*Falsesl*', '12pa_ff_*2048N*7.5e-06glr*Falsesl*']
allreseh = ['12pa_res_*128N*Truesl*', '12pa_res_*256N*Truesl*', '12pa_res_*512N*Truesl*', '12pa_res_*1024N*Truesl*',
            '12pa_res_*2048N*Truesl*7.5e-06glr*'] #,'12pa_res_*2048N*Truesl*_5e-06glr*'
allreslms = ['12pa_res_*128N*Falsesl', '12pa_res_*256N*Falsesl', '12pa_res_*512N*Falsesl', '12pa_res_*1024N*Falsesl',
             '12pa_res_*2048N*Falsesl']

completemodels = [allsym, allfflms, allffeh, allreslms, allreseh]

modelcolor = ['deeppink', 'dodgerblue', 'purple','forestgreen','black','orange']


pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    #genvar_path = '../12pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/12pa/'
    totiter = 288

    for allmodels in completemodels:
        for model in allmodels:
            print(model)
            model_data = glob.glob(genvar_path+'genvars_*{}*'.format(model))
            print(len(model_data))

            totpi, totdgr = [], []
            for idx, data in enumerate(model_data):
                [dgr, pi, _] = saveload('load',data,1)

                totpi.append(pi)
                totdgr.append(dgr)

            if len(model_data) !=0:
                totpi = np.concatenate(totpi,axis=0)
                totdgr = np.concatenate(totdgr,axis=0)

                if len(totdgr) > totiter:
                    lidx = np.random.choice(np.arange(len(totdgr)), totiter, replace=False)
                else:
                    lidx = np.arange(len(totdgr))

                newstr = model.replace("*", "_")
                saveload('save','./Data/12pa/comb_genvar_12pa_{}_{}b'.format(newstr, len(lidx)), [totpi[lidx], totdgr[lidx]])


elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    genvar_path = './Data/12pa/'

    dgr = []
    pi = []
    z = 1.96
    for allmodels in completemodels:

        alldgr = np.zeros([2,len(allmodels)])
        allpi = np.zeros([2, len(allmodels)])
        fulldgr = []

        for idx, model in enumerate(allmodels):
            newstr = model.replace("*", "_")
            model_data = glob.glob(genvar_path+'comb_genvar_12pa_{}*'.format(newstr))

            [totpi, totdgr] = saveload('load',model_data[0],1)

            alldgr[0,idx] = np.mean(totdgr, axis=0)[-1]
            alldgr[1,idx] = z*(np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))[-1]
            #alldgr[1, idx] = np.std(totdgr, axis=0)[-1]

            allpi[0,idx] = np.mean(totpi, axis=0)[-1]
            allpi[1,idx] = z*(np.std(totpi, axis=0)/np.sqrt(len(totpi)))[-1]
            #allpi[1, idx] = np.std(totpi, axis=0)[-1]

            fulldgr.append(totdgr[:,-1])

        dgr.append(alldgr)
        pi.append(allpi)

    # plot cb = 0,1
    x = 2 ** np.arange(7, 12)
    f = plt.figure(figsize=(4, 2))
    ax = plt.subplot(111)
    f2 = plt.figure(figsize=(4, 2))
    ax2 = plt.subplot(111)
    alldgr = dgr[0]
    allpi = pi[0]
    for i in range(2):
        ax.plot(x, np.tile(alldgr[0,i],(len(x))), color=modelcolor[i], linewidth=1)
        ax2.plot(x, np.tile(allpi[0, i], (len(x))), color=modelcolor[i], linewidth=1)

    for i in [0,2,1,3]:
        alldgr = dgr[i+1]
        allpi = pi[i+1]
        ax.plot(x, alldgr[0], color=modelcolor[i+2], marker='o', ms=3, linewidth=1)
        #ax.errorbar(x=x, y=alldgr[0],yerr=alldgr[1] ,color=modelcolor[i + 2], marker='o')
        ax2.plot(x, allpi[0], color=modelcolor[i+2], marker='o',ms=3, linewidth=1)

    ax.legend(['ActorCritic','Symbolic','FF_LMS','Res_LMS','FF_EH','Res_EH']
              , fontsize=8, frameon=False, loc='center left',bbox_to_anchor=(1, 0.5))
    ax2.legend(['ActorCritic','Symbolic','FF_LMS','Res_LMS','FF_EH','Res_EH']
              , fontsize=8, frameon=False, loc='center left',bbox_to_anchor=(1, 0.5))

    ax.axhline(100/12, color='r',linestyle='--')
    alldgr = dgr[0]
    allpi = pi[0]
    for i in range(2):
        ax.fill_between(x=x, y1=alldgr[0, i] - np.tile(alldgr[1, i], (len(x))),
                        y2=alldgr[0, i] + np.tile(alldgr[1, i], (len(x))), alpha=0.1, color=modelcolor[i])
        ax2.fill_between(x=x,y1=allpi[0,i] - np.tile(allpi[1,i],(len(x))), y2=allpi[0,i] + np.tile(allpi[1,i],(len(x))),alpha=0.1,color=modelcolor[i])

    # plot individual
    for i in range(4):
        alldgr = dgr[i+1]
        allpi = pi[i+1]
        ax.fill_between(x=x, y1=alldgr[0] - alldgr[1], y2=alldgr[0] + alldgr[1], alpha=0.2, color=modelcolor[i+2])
        ax2.fill_between(x=x, y1=allpi[0] - allpi[1], y2=allpi[0] + allpi[1], alpha=0.2, color=modelcolor[i+2])

    ax.set_xscale('log', base=2)
    ax2.set_yticks(np.linspace(0, 12, 5, dtype=int), np.linspace(0, 12, 5, dtype=int))
    ax.set_ylabel('Visit Ratio (%)')
    ax.set_xlabel('Number of nonlinear units')
    ax.set_title('One-shot learning of 12NPA')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])

    # ax2.set_xlim(left=110,right=7000)
    ax2.set_yticks(np.linspace(0,12,5,dtype=int),np.linspace(0,12,5,dtype=int))
    ax2.set_xscale('log', base=2)
    ax2.set_ylabel('# PAs learned')
    ax2.set_xlabel('Number of nonlinear units')
    ax2.set_title('One-shot learning of 12NPA')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 1, box.height])

    # from scipy.stats import ttest_1samp
    # for m in range(7):
    #     tp = ttest_1samp(fulldgr[m], 100/12,alternative='two-sided')
    #     print(tp)
    #     ts = ttest_ind_from_stats(mean1=alldgr[0,m],std1=alldgr[1,m],nobs1=96, mean2=alldgr[0,1],std2=alldgr[1,1],nobs2=96,alternative='two-sided')
    #     print(ts)
    #     if ts[0]>0 and ts[1]<0.001 and m>1:
    #         ax.annotate('***', (x[m-2]-30, alldgr[0,m]*1.04))

    f.tight_layout()
    f2.tight_layout()
    plt.show()
    # savefig('./Fig/12pa/12pa_dgr',f)
    # savefig('./Fig/12pa/12pa_pi', f2)


    # f2.savefig('./Fig/12pa/12pa_pi.png')
    # f2.savefig('./Fig/12pa/12pa_pi.svg')
    # f.savefig('./Fig/12pa/12pa_dgr.png')
    # f.savefig('./Fig/12pa/12pa_dgr.svg')

elif int(pltfig) == 3:
    genvar_path = '../12pa/Data/'
    fulltraj = False
    N = 4
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 12))
    trajalpha = 1
    fullpath = np.zeros([3, N, 12, 3001,2])

    for idx, model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'genvars_*{}*'.format(model))

        [totdgr, totpi, totpath] = saveload('load',model_data[0],1)

        if idx == 0:
            cidx = np.arange(len(totdgr))[totdgr[:,-1]<100/12]
            fullpath[idx] = totpath[np.random.choice(cidx,N,replace=False)]
        else:
            cidx = np.arange(len(totdgr))[totdgr[:,-1]>38]
            fullpath[idx] = totpath[np.random.choice(cidx,N,replace=False)]

    for n in range(N):
        f,ax = plt.subplots(3,12,figsize=(12,3))
        for d in range(3):
            for i in range(12):
                traj = fullpath[d,n,i]
                goal = traj[-1]
                coord = traj[:-1]

                if fulltraj:
                    idx = -1
                else:
                    idx = np.argmax(np.linalg.norm(goal - traj, axis=1) < 0.1) - 1

                ax[d,i].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=colors[i], alpha=trajalpha,zorder=2, ms=2)
                ax[d,i].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=colors[i], alpha=trajalpha,zorder=2,ms=2)

                ax[d,i].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=colors[i], alpha=trajalpha,zorder=1,linewidth=1)
                circle = plt.Circle(goal, 0.03, color=colors[i],zorder=3)
                ax[d,i].add_artist(circle)
                circle2 = plt.Circle(goal, 0.03, color='k',zorder=3,fill=False)
                ax[d,i].add_artist(circle2)

                ax[d, i].axis((-1.6 / 2, 1.6 / 2, -1.6 / 2, 1.6 / 2))
                ax[d, i].set_aspect('equal', adjustable='box')
                ax[d, i].set_xticks([])
                ax[d, i].set_yticks([])
            ax[d, 0].set_ylabel(modellegend[d],fontsize=8)
        #f.tight_layout()

        f.savefig('./Fig/12pa/12pa_traj{}.png'.format(n))
        f.savefig('./Fig/12pa/12pa_traj{}.svg'.format(n))


elif int(pltfig) == 4:
    genvar_path = '../6pa/Data/'
    #genvar_path = 'D:/Ganesh_PhD/Schema4PA/6pa/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    hp = get_default_hp(task='6pa', platform='laptop')
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA']

    bins = 15
    newx = np.zeros([bins**2, 2])
    for i in range(bins):
        st = i * bins
        ed = st + bins
        newx[st:ed, 0] = np.arange(bins)
    for i in range(bins):
        st = i * bins
        ed = st + bins
        newx[st:ed, 1] = i * np.ones(bins)

    for m,model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)

        if N == 0:
            pass
        else:

            policy = np.zeros([6,6, 2,bins,bins])
            value = np.zeros([6,6, bins,bins])

            for d in range(N):

                [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',model_data[d],1)

                sess = list(alldyn[1].keys())

                for p in range(6):
                    ss = sess[p * 6:p * 6 + 6]

                    for c in ss:
                        cue = int(c[-1])-1
                        if cue == 6:
                            cue = 0
                        elif cue == 7:
                            cue = 5
                        qfr = np.vstack(alldyn[1][c])
                        cfr = np.vstack(alldyn[2][c])
                        coord = mvpath[p,cue]

                        policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr)

                        policy[p,cue] += policymap
                        value[p,cue] += valuemap

            value /=N
            policy /= N

            f,ax = plt.subplots(6,6,figsize=(10,10))
            for p in range(6):
                for c in range(6):
                    im = ax[p,c].imshow(value[p,c].T, aspect='auto', origin='lower')
                    ax[p,c].quiver(newx[:, 1], newx[:, 0], policy[p,c,1].reshape(bins ** 2), policy[p,c,0].reshape(bins ** 2),
                               units='xy', color='w')
                    plt.colorbar(im,ax=ax[p,c], fraction=0.046, pad=0.04)
                    ax[p,c].set_xticks([], [])
                    ax[p,c].set_yticks([], [])
                    ax[p,c].set_aspect('equal', adjustable='box')
                    ax[p,c].set_title('{}_C{}'.format(mlegend[p],c+1))

            f.tight_layout()
            f.text(0.01,0.01,model[:11])
            f.savefig('./Fig/6pa/6pa_maps_{}_b{}.png'.format(model[:11],N))
            f.savefig('./Fig/6pa/6pa_maps_{}_b{}.svg'.format(model[:11],N))
            plt.close()
