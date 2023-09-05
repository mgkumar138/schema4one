import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d
import matplotlib as mpl
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

showall = True
if showall:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb','sym_0.4cb','res_0.4']  # LR, EH, Sym
    #modellegend = ['ActorCritic','Sym. NAVIGATE', 'Neural NAVIGATE', 'AC+Sym. NAVGATE', 'AC+Neural NAVIGATE']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural','AC+Sym','AC+Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange', 'darkorchid','limegreen']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']

pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    genvar_path = 'D:/Ganesh_PhD/s4o/obs/'
    #genvar_path = '../obs/Data/'
    totiter = 480

    for model in allmodels:
        print(model)
        model_data = glob.glob(genvar_path+'genvars_*{}*'.format(model))
        print(len(model_data))

        totlat, totdgr, totpi = [], [], []
        for idx, data in enumerate(model_data):
            [lat, dgr, pi] = saveload('load',data,1)

            totlat.append(lat)
            totdgr.append(dgr)
            totpi.append(pi)

        try:
            totlat = np.concatenate(totlat,axis=0)
            totdgr = np.concatenate(totdgr,axis=0)
            totpi = np.concatenate(totpi,axis=0)

            if len(totdgr) > totiter:
                lidx = np.random.choice(np.arange(len(totdgr)), totiter, replace=False)
            else:
                lidx = np.arange(len(totdgr))

            saveload('save','./Data/obs/comb_genvar_obs_{}_{}b'.format(model[:7], len(lidx)), [totlat[lidx], totdgr[lidx],totpi[lidx]])
        except ValueError: print('No data {}'.format(model))

elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    genvar_path = './Data/obs/'
    scalet = 20/1000

    tl, td = [], []
    dfm, dfs = [], []
    pfm, pfs = [], []
    fulldgr = []
    fulllat = []
    z = 1.96
    for idx, model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'comb_genvar_obs_{}*'.format(model[:7]))

        [totlat, totdgr, totpi] = saveload('load',model_data[0],1)

        tl.append(np.mean(totlat[:,:50]*scalet, axis=0))
        #td.append(np.array([np.nanquantile(totlat[:,:50]*scalet, 0.25, axis=0),np.nanquantile(totlat[:,:50]*scalet, 0.75, axis=0)]))

        td.append(np.array([np.mean(totlat[:,:50] * scalet, axis=0) - z*np.std(totlat[:,:50]*scalet, axis=0)/np.sqrt(len(totdgr)),
                            np.mean(totlat[:,:50] * scalet, axis=0) + z*np.std(totlat[:,:50]*scalet, axis=0)/np.sqrt(len(totdgr))]))

        dfm.append(np.mean(totdgr, axis=0))
        dfs.append(z*np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))

        pfm.append(np.mean(totpi, axis=0))
        pfs.append(z*np.std(totpi, axis=0)/np.sqrt(len(totpi)))

        fulldgr.append(totdgr)
        fulllat.append(totlat[:,:50]*scalet)

    from scipy.stats import ttest_1samp
    stt = np.zeros([5,6,2])
    for t in range(6):
        for m in range(5):
            stt[m,t] = ttest_1samp(fulldgr[m][:,t], 100/6,alternative='greater')

    # from scipy.stats import pearsonr
    # latt = []
    # nonanidx = ~np.isnan(fulllat[0])
    # for i in range(5):
    #     x = fulllat[i]
    #     latt.append(pearsonr(fulllat[i][nonanidx],fulllat[0][nonanidx]))
    # print(latt)

    f1,ax1 = plt.subplots(1,1,figsize=(4,2))
    for idx in range(len(modellegend)):
        ax1.plot(np.arange(1, 1 + 50), tl[idx], color=modelcolor[idx], linewidth=1, ms=2, marker='o')
    #ax1.legend(modellegend, loc='upper right', fontsize=6)
    for idx in range(len(modellegend)):
        ax1.fill_between(np.arange(1, 1 + 50), td[idx][0], td[idx][1], alpha=0.1, facecolor=modelcolor[idx])
        ax1.fill_between(np.linspace(0.75, 1.4,2), [td[idx][0][0],td[idx][0][0]], [td[idx][1][0],td[idx][1][0]], alpha=0.1, facecolor=modelcolor[idx])
    ax1.set_xlabel('Session')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title('Average time to reach correct target')
    ax1.set_xticks(np.linspace(1,50,6,dtype=int),np.linspace(1,50,6,dtype=int))
    sessidx = [2,22,40]
    for pt in range(3):
        ax1.annotate('PS{}'.format(pt + 1), (sessidx[pt]-0.7, 400), rotation=90, fontsize=8)

    f2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(dfm).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax2, yerr=ds, legend=True, color=modelcolor)
    ax2.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax2.axhline(100/6,color='r',linestyle='--')
    ax2.set_ylabel('Visit ratio (%)')
    ax2.set_title('Ratio of time at correct target')
    ax2.set_ylim(0, 65)

    ti = 0
    for p,t in zip(ax2.patches,stt[:,:3].reshape([15,2])):
        if t[0] > 0:
            print(p.get_x(), t[0])
            if t[1] < 0.0001:
                ax2.annotate('****', (p.get_x(), p.get_height() * 1.05),fontsize=4)
            elif t[1] < 0.001:
                ax2.annotate('***', (p.get_x(), p.get_height() * 1.05),fontsize=4)
            elif t[1] < 0.01:
                ax2.annotate('**', (p.get_x(), p.get_height() * 1.05),fontsize=4)
            elif t[1] < 0.05:
                ax2.annotate('*', (p.get_x(), p.get_height() * 1.05),fontsize=4)

    f3, ax3 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(dfm).T[3:], index=['OPA', '2NPA', '6NPA'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T[3:], index=['OPA', '2NPA', '6NPA'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax3, yerr=ds, legend=False, color=modelcolor)
    ax3.axhline(100 / 6, color='r', linestyle='--')
    #ax3.legend(modellegend, loc='upper left', fontsize=6)
    ax3.set_ylabel('Visit ratio (%)')
    ax3.set_title('Ratio of time at correct target after 1 session')
    ax3.set_ylim(0, 65)

    ti = 0
    for p,t in zip(ax3.patches,stt[:,3:].reshape([15,2])):
        if t[0] > 0:
            print(p.get_x(), t[0])
            if t[1] < 0.0001:
                ax3.annotate('****', (p.get_x(), p.get_height() * 1.05))
            elif t[1] < 0.001:
                ax3.annotate('***', (p.get_x(), p.get_height() * 1.05))
            elif t[1] < 0.01:
                ax3.annotate('**', (p.get_x(), p.get_height() * 1.05))
            elif t[1] < 0.05:
                ax3.annotate('*', (p.get_x(), p.get_height() * 1.05))


    f4, ax4 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(pfm).T[3:], index=['OPA', '2NPA', '6NPA'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(pfs).T[3:], index=['OPA', '2NPA', '6NPA'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax4, yerr=ds, legend=False, color=modelcolor)
    #ax3.axhline(100 / 6, color='r', linestyle='--')
    #ax3.legend(modellegend, loc='upper left', fontsize=6)
    ax4.set_ylabel('Performance index (%)')
    ax4.set_title('Number of paired associations learned')
    for p in ax4.patches:
        ax4.annotate(str(np.round(p.get_height(),1)), (p.get_x() * 1.005, p.get_height() + 0.2))

    f1.tight_layout()
    f2.tight_layout()
    f3.tight_layout()
    f4.tight_layout()

    # savefig('./Fig/obs/obs_latency',f1)
    # savefig('./Fig/obs/obs_ps_dgr', f2)
    # savefig('./Fig/obs/obs_nm', f3)

    # f1.savefig('./Fig/obs/obs_latency_{}all.png'.format(showall))
    # f1.savefig('./Fig/obs/obs_latency_{}all.svg'.format(showall))
    # f2.savefig('./Fig/obs/obs_dgr_opa_{}all.png'.format(showall))
    # f2.savefig('./Fig/obs/obs_dgr_opa_{}all.svg'.format(showall))
    # f3.savefig('./Fig/obs/obs_dgr_npa_{}all.png'.format(showall))
    # f3.savefig('./Fig/obs/obs_dgr_npa_{}all.svg'.format(showall))
    # f4.savefig('./Fig/obs/obs_pi_npa_{}all.png'.format(showall))
    # f4.savefig('./Fig/obs/obs_pi_npa_{}all.svg'.format(showall))

elif int(pltfig) == 3:
    genvar_path = 'D:/Ganesh_PhD/s4o/obs/'
    #genvar_path = '../obs/Data/'
    from backend.maze import Maze
    from backend.utils import get_default_hp

    hp = get_default_hp(task='6pa_obs', platform='laptop')
    hp['obs'] = True

    mtype = ['train','train','train','opa','2npa','6npa']
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA']
    fulltraj = False
    fullcue = 2
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol1,trajcol1,trajcol1,trajcol2,trajcol3]
    trajalpha = 0.8
    tnum = 1

    for m,model in enumerate(allmodels):

        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)
        #ridx = np.random.choice(np.arange(N),5, replace=False)
        ridx = []
        for n in range(N)[::-1]:
            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load', model_data[n], 1)
            print(dgr[3:6])

            if m==0 and dgr[3]>50:
                ridx.append(n)
            elif m==1 and dgr[3] < 17:
                ridx.append(n)
            elif m == 2 and dgr[3]<30 and dgr[3]>17:
                ridx.append(n)
            elif m == 3 and dgr[3]>50 and dgr[4]>50 and dgr[5]>50:
                ridx.append(n)
            elif m==4 and dgr[3]>50 and dgr[4]>45 and dgr[5]>45:
                ridx.append(n)

        print(len(ridx))
        f,ax = plt.subplots(nrows=5, ncols=6, figsize=(10, 8))
        ridx = np.array(ridx)
        np.random.shuffle(ridx)
        for d, didx in enumerate(ridx[:5]):
            df = model_data[didx]
            print(df)
            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',df,1)

            for i in range(6):
                env = Maze(hp)
                env.make(mtype[i])
                trajcol = col[i]

                for c in [0,5]:
                    traj = mvpath[i, c]
                    rloc = env.rlocs[c]

                    if fulltraj:
                        idx = -1
                    else:
                        idx = np.argmax(np.linalg.norm(rloc - traj, axis=1) < 0.1) - 1

                    ax[d,i].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=trajcol[c], alpha=trajalpha,zorder=2)
                    ax[d,i].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=trajcol[c], alpha=trajalpha,zorder=2)

                    ax[d,i].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[c], alpha=trajalpha,zorder=1,linewidth=1)
                    circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c],zorder=3)
                    ax[d,i].add_artist(circle)
                    circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k',zorder=3,fill=False)
                    ax[d,i].add_artist(circle2)

                for ld in env.obstacles:
                    circle = plt.Circle(ld, env.obssz, color='k', zorder=1)
                    ax[d,i].add_artist(circle)

                ax[d,i].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
                ax[d,i].set_aspect('equal', adjustable='box')
                ax[d,i].set_xticks([])
                ax[d,i].set_yticks([])
                ax[d,i].set_title(mlegend[i])
        f.tight_layout()

        f.text(0.01,0.01,model)
        savefig('./Fig/obs/obs_traj{}_{}cue_{}.png'.format(tnum,fulltraj, model[:7]), f)
        plt.close('all')


elif int(pltfig) == 4:
    #genvar_path = '../obs/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/obs/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    from backend.maze import Maze

    mtype = ['train','train','train','opa','2npa','6npa']
    # trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    # trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    # trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    # col = [trajcol1,trajcol1,trajcol1,trajcol1,trajcol2,trajcol3]

    hp = get_default_hp(task='obs', platform='laptop')
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA']

    bins = 15
    varN = 3
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

        randidx = np.random.choice(np.arange(N), varN,replace=False)

        if N == 0:
            pass
        else:

            policy = np.zeros([6,6, 2,bins,bins])
            value = np.zeros([6,6, bins,bins])

            for d in randidx:

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

                        policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr, bins=bins)

                        policy[p,cue] += policymap
                        value[p,cue] += valuemap

            value /=N
            policy /= N
            saveload('save', './Data/obs/var_obs_maps_{}_b{}'.format(model[:7],varN), [value, policy])

            # f,ax = plt.subplots(6,6,figsize=(10,10))
            # for p in range(6):
            #     env = Maze(hp)
            #     env.make(mtype[p])
            #     for c in range(6):
            #         im = ax[p,c].imshow(value[p,c].T, aspect='auto', origin='lower')
            #         ax[p,c].quiver(newx[:, 1], newx[:, 0], policy[p,c,0].reshape(bins ** 2), policy[p,c,1].reshape(bins ** 2),
            #                    units='xy', color='w')
            #         plt.colorbar(im,ax=ax[p,c], fraction=0.046, pad=0.04)
            #         ax[p,c].set_xticks([], [])
            #         ax[p,c].set_yticks([], [])
            #         ax[p,c].set_aspect('equal', adjustable='box')
            #         ax[p,c].set_title('{}_C{}'.format(mlegend[p],c+1))
            #
            # f.tight_layout()
            #f.text(0.01,0.01,model[:7])
            #f.savefig('./Fig/obs/obs_maps_{}_b{}.png'.format(model[:7], len(randidx)))
            #f.savefig('./Fig/obs/obs_maps_{}_b{}.svg'.format(model[:7], len(randidx)))
            #plt.close()

elif int(pltfig) == 5:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    # genvar_path = 'D:/Ganesh_PhD/Schema4PA/obs/'  # cb=0.4
    # scalet = 20/1000
    #
    # for m in ['sym','res']:
    #     xrange = [0,0.25,0.3,0.4,0.5,1]
    #     y = np.zeros([len(xrange),4])
    #     for i, cb in enumerate(xrange):
    #         model_data = glob.glob(genvar_path+'genvar*_obs_{}_{}cb*stpos*'.format(m,cb))
    #
    #         if len(model_data) > 0:
    #             for j, data in enumerate(model_data):
    #                 [totlat, totdgr, totpi] = saveload('load',data,1)
    #
    #                 y[i,:3] += np.mean(totdgr[:,3:],axis=0)
    #                 y[i,3] += totdgr.shape[0]
    #
    #             y[i] /= (j+1)
    #
    #     print(y)

    genvar_path = 'D:/Ganesh_PhD/Schema4PA/1pa/'  # cb=0.3
    scalet = 20 / 1000

    for m in ['sym', 'res']:
        xrange = [0, 0.25, 0.3, 0.4, 0.5, 1]
        y = np.zeros([len(xrange), 3])
        for i, cb in enumerate(xrange):
            model_data = glob.glob(genvar_path + 'genvar*_center_{}_{}cb*stpos*'.format(m, cb))

            if len(model_data) > 0:
                for j, data in enumerate(model_data):
                    [totlat, totdgr, _] = saveload('load', data, 1)

                    y[i, 0] += np.mean(totdgr[:, -1], axis=0)
                    y[i, 1] += np.mean(np.mean(totlat, axis=0)[-16:-6])
                    y[i, 2] += totdgr.shape[0]

                y[i,:2] /= (j + 1)

        print(y)

elif int(pltfig) == 6:
    #genvar_path = '../obs/Data/'
    genvar_path = 'D:/Ganesh_PhD/Schema/obs/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    from backend.maze import Maze

    mtype = ['train','train','train','opa','2npa','6npa']
    # trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    # trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    # trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    # col = [trajcol1,trajcol1,trajcol1,trajcol1,trajcol2,trajcol3]

    hp = get_default_hp(task='6pa_obs', platform='laptop')
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

    for m,model in enumerate([allmodels[2]]):
        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)
        #randidx = np.random.choice(np.arange(N), 24,replace=False)
        randidx = np.arange(N)

        if N == 0:
            pass
        else:

            policy = np.zeros([6,6, 2,bins,bins])
            value = np.zeros([6,6, bins,bins])
            recall = np.zeros([6,6, bins,bins])

            for d in randidx:

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
                        # qfr = np.vstack(alldyn[1][c])
                        # cfr = np.vstack(alldyn[2][c])
                        recallfr = np.vstack(alldyn[3][c])
                        coord = mvpath[p,cue]

                        #policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr, bins=bins)
                        recallmap = binned_statistic_2d(coord[:, 0], coord[:, 1], recallfr[:, -1], bins=bins, statistic='mean')[0]
                        recallmap = np.nan_to_num(recallmap)

                        #policy[p,cue] += policymap
                        #value[p,cue] += valuemap
                        recall[p,cue] += recallmap

            # value /=N
            # policy /= N
            recall /= N

            f,ax = plt.subplots(6,6,figsize=(10,10))
            for p in range(6):
                env = Maze(hp)
                env.make(mtype[p])
                for c in range(6):
                    im = ax[p,c].imshow(recall[p,c].T, aspect='auto', origin='lower')
                    # ax[p,c].quiver(newx[:, 1], newx[:, 0], policy[p,c,0].reshape(bins ** 2), policy[p,c,1].reshape(bins ** 2),
                    #            units='xy', color='w')
                    plt.colorbar(im,ax=ax[p,c], fraction=0.046, pad=0.04)
                    ax[p,c].set_xticks([], [])
                    ax[p,c].set_yticks([], [])
                    ax[p,c].set_aspect('equal', adjustable='box')
                    ax[p,c].set_title('{}_C{}'.format(mlegend[p],c+1))

            f.tight_layout()
            f.text(0.01,0.01,model[:7])
            f.savefig('./Fig/obs/obs_recall_{}_{}bins_b{}.png'.format(model[:7],bins, len(randidx)))
            f.savefig('./Fig/obs/obs_recall_{}_{}bins_b{}.svg'.format(model[:7],bins, len(randidx)))
            plt.close()


elif int(pltfig) == 8:

    from backend.maze import Maze
    from backend.utils import get_default_hp

    hp = get_default_hp(task='obs', platform='laptop')
    hp['obs'] = True
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol2,trajcol3]
    mtype = ['opa','2npa','6npa']
    env = Maze(hp)

    mlegend = ['OPA \n Cue 2', '2NPA \n Cue 7', '6NPA \n Cue 16']
    modlegend = ['ActorCritic \n \u03B2 = 0','Symbolic \n \u03B2 = 1','Neural \n \u03B2 = 1','AC+Symbolic \n \u03B2 = 0.4','AC+Neural \n \u03B2 = 0.4']
    midx = np.array([[3,1],[4,0],[5,5]])
    varN = 94
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

    f = plt.figure(figsize=(8, 4))
    i = -1
    idx = [1,6,11,2,7,12,3,8,13,4,9,14,5,10,15]
    for m,model in enumerate(allmodels):
        [value, policy] = saveload('load', './Data/obs/var_obs_maps_{}_b{}.pickle'.format(model[:7],varN), 1)
        for p in range(3):
            i +=1
            val = value[midx[p,0], midx[p,1]]
            pol = policy[midx[p,0], midx[p,1]]

            ax2 = f.add_subplot(3,5,idx[i],label='map')
            im = ax2.imshow(val.T, aspect='auto', origin='lower')
            ax2.quiver(newx[:, 1], newx[:, 0], pol[0].reshape(bins ** 2), pol[1].reshape(bins ** 2), units='xy', color='w')
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04) #,orientation="horizontal",
            cbar.ax.tick_params(labelsize=6)

            ax3 = f.add_subplot(3,5,idx[i], label='loc', frame_on=False)
            env.make(mtype[p])
            trajcol = col[p]
            for c in range(6):
                rloc = env.rlocs[c]
                circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c], zorder=3)
                ax3.add_artist(circle)
            for j in range(len(env.obstacles)):
                circle = plt.Circle(env.obstacles[j], env.obssz, color='k')
                ax3.add_artist(circle)

            ax2.set_xticks([], [])
            ax2.set_yticks([], [])
            ax2.set_aspect('equal', adjustable='box')

            ax3.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
            ax3.set_aspect('equal', adjustable='box')
            ax3.set_xticks([], [])
            ax3.set_yticks([], [])

            if idx[i]==1 or idx[i]==6 or idx[i]==11:
                ax2.set_ylabel(mlegend[p], fontsize=8)
            if idx[i] <6:
                ax2.set_title(modlegend[m], fontsize=8)

    f.tight_layout()
    savefig('./Fig/obs/obs_cuemap_hori_b{}'.format(varN), f)

elif int(pltfig) == 9:
    genvar_path = 'D:/Ganesh_PhD/s4o/obs/'
    #genvar_path = '../obs/Data/'
    from backend.maze import Maze
    from backend.utils import get_default_hp

    hp = get_default_hp(task='obs', platform='laptop')
    hp['obs'] = True

    mtype = ['opa','2npa','6npa']
    mlegend = ['OPA \n 2 & 5', '2NPA \n 7 & 8', '6NPA \n 11 & 16']
    modlegend = ['ActorCritic','Symbolic','Neural','AC+Sym','AC+Neural']
    fulltraj = False
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol2,trajcol3]
    trajalpha = 0.8
    tnum = 3

    allridx = np.zeros([5, tnum], dtype=int)
    for m,model in enumerate(allmodels):

        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)
        #ridx = np.random.choice(np.arange(N),5, replace=False)
        ridx = []
        for n in range(N)[::-1]:

            if len(ridx) >= tnum:
                print('model {} loaded'.format(model))
                break
            else:
                [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load', model_data[n], 1)
                print(np.round(dgr[np.array([3, 4, 5])]))

                if m==0 and dgr[3]>55 and dgr[4]< 17 and dgr[5]<17:
                    ridx.append(n)
                elif m==1 and dgr[3] < 10 and dgr[4]<10 and dgr[5]<10:
                    ridx.append(n)
                elif m == 2 and dgr[3] >35 and dgr[4]<10 and dgr[5]<10:
                    ridx.append(n)
                elif m == 3 and dgr[3]>55 and dgr[4]>55 and dgr[5]>45:
                    ridx.append(n)
                elif m==4 and dgr[3]>45 and dgr[4]>40 and dgr[5]>35:
                    ridx.append(n)

        allridx[m] = np.random.choice(ridx,tnum, replace=False)

    for t in range(tnum):
        f, ax = plt.subplots(nrows=3, ncols=5, figsize=(7, 4))

        for m, model in enumerate(allmodels):
            model_data = glob.glob(genvar_path + 'vars_*{}*'.format(model))

            didx = int(allridx[m,t])
            df = model_data[didx]
            print(df)
            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',df,1)
            print(np.round(dgr[np.array([3, 4, 5])]))

            for i in range(3):
                env = Maze(hp)
                env.make(mtype[i])
                trajcol = col[i]
                if i == 0:
                    cidx = [1,4]
                else:
                    cidx = [0,5]

                for c in cidx:
                    traj = mvpath[i+3, c]
                    rloc = env.rlocs[c]

                    if fulltraj:
                        idx = -1
                    else:
                        idx = np.argmax(np.linalg.norm(rloc - traj, axis=1) < 0.1) - 1

                    ax[i,m].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=trajcol[c], alpha=trajalpha,zorder=2)
                    ax[i,m].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=trajcol[c], alpha=trajalpha,zorder=2)

                    ax[i,m].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[c], alpha=trajalpha,zorder=1,linewidth=1)
                    circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c],zorder=3)
                    ax[i,m].add_artist(circle)
                    circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k',zorder=3,fill=False)
                    ax[i,m].add_artist(circle2)

                for ld in env.obstacles:
                    circle = plt.Circle(ld, env.obssz, color='k', zorder=1)
                    ax[i,m].add_artist(circle)

                ax[i,m].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
                ax[i,m].set_aspect('equal', adjustable='box')
                ax[i,m].set_xticks([])
                ax[i,m].set_yticks([])

                ax[i,0].set_ylabel(mlegend[i], fontsize=12)
                #ax[0,m].set_title(modlegend[m], fontsize=4)
        f.tight_layout()

        #f.text(0.01,0.01,model)
        savefig('./Fig/obs/obs_traj_{}3'.format(tnum), f)
        plt.close('all')