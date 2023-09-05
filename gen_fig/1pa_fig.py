import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d, f_oneway,ttest_ind
import matplotlib as mpl
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

showall = True
if showall:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb','sym_0.3','res_0.3']  # LR, EH, Sym
    #modellegend = ['ActorCritic','Sym. NAVIGATE', 'Neural NAVIGATE', 'AC+Sym. NAVGATE', 'AC+Neural NAVIGATE']
    modellegend = ['Actor-Critic', 'Symbolic', 'Neural','AC+Symbolic','AC+Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange', 'darkorchid','limegreen']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['Actor-Critic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']

pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    #genvar_path = '../1pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/1pa/'
    totiter = 480

    for model in allmodels:
        model_data = glob.glob(genvar_path+'genvars_*{}*'.format(model))

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
        print(len(model_data), len(totdgr))

        saveload('save','./Data/1pa/comb_genvar_1pa_{}_{}b'.format(model[:7], len(lidx)), [totlat[lidx], totdgr[lidx]])

elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    genvar_path = './Data/1pa/'

    scalet = 1000/1000

    tl, td,tstd = [], [], []
    dfm, dfs = [], []
    fulldgr = []
    fulllat = []
    z = 1.96
    for idx, model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'comb_genvar_1pa_{}*'.format(model[:7]))

        [totlat, totdgr] = saveload('load',model_data[0],1)

        tl.append(np.mean(totlat*scalet, axis=0))
        tstd.append(z*np.std(totlat*scalet,axis=0)/np.sqrt(len(totdgr)))
        #td.append(np.array([np.nanquantile(totlat*scalet, 0.25, axis=0),np.nanquantile(totlat*scalet, 0.75, axis=0)]))


        td.append(np.array([np.mean(totlat * scalet, axis=0) - z*np.std(totlat*scalet, axis=0)/np.sqrt(len(totdgr)),
                            np.mean(totlat * scalet, axis=0) + z*np.std(totlat*scalet, axis=0)/np.sqrt(len(totdgr))]))

        dfm.append(np.mean(totdgr, axis=0))
        dfs.append(z*np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))

        fulldgr.append(totdgr)
        fulllat.append(totlat*scalet)

    stt = np.zeros([5,3,2])

    for t in range(3):
        for m in range(5):
            stt[m,t] = ttest_ind_from_stats(mean1=np.vstack(dfm)[m, t], std1=np.vstack(dfs)[m, t], nobs1=len(totdgr),
                                      mean2=np.vstack(dfm)[0, t], std2=np.vstack(dfs)[0, t], nobs2=len(totdgr), alternative='two-sided')

    f1,ax1 = plt.subplots(1,1,figsize=(4,2))
    f2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    for idx in range(len(modellegend)):
        ax1.plot(np.arange(1, 1 + totlat.shape[1]), tl[idx], color=modelcolor[idx])
        ax1.fill_between(np.arange(1, 1 + totlat.shape[1]), td[idx][0], td[idx][1], alpha=0.1, facecolor=modelcolor[idx])
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title('Time to reach single target')
    ax1.set_xlim(-2,60)
    #ax1.legend(modellegend, loc='lower left', fontsize=6, frameon=False)
    #ax1.legend(modellegend, loc='upper left', fontsize=6)
    sessidx = [9,33,57]
    for pt in range(3):
        ax1.annotate('PT{}'.format(pt + 1), (sessidx[pt], 100), rotation=90)

    from scipy.stats import pearsonr
    latt = []
    nonanidx = ~np.isnan(fulllat[0])
    for i in range(5):
        x = fulllat[i]
        latt.append(pearsonr(fulllat[i][nonanidx],fulllat[0][nonanidx]))
    print(latt)

    df = pd.DataFrame(np.vstack(dfm).T, index=['PT1', 'PT2', 'PT3'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T, index=['PT1', 'PT2', 'PT3'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax2, yerr=ds, legend=False, color=modelcolor)
    ax2.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax2.set_ylabel('Time at target (%)')
    ax2.set_title('Time spent at target during probe')
    ax2.set_ylim(0, 8.5)

    # for p,t in zip(ax2.patches,stt.reshape([15,2])):
    #     if t[0] > 0:
    #         print(p.get_x(), t[0], t[1])
    #         if t[1] < 0.0001:
    #             ax2.annotate('****', (p.get_x() * 1, p.get_height() * 1.05))
    #         elif t[1] < 0.001:
    #             ax2.annotate('***', (p.get_x()*1, p.get_height() * 1.05))
    #         elif t[1] < 0.01:
    #             ax2.annotate('**', (p.get_x()*1, p.get_height() * 1.05))
    #         elif t[1] < 0.05:
    #             ax2.annotate('*', (p.get_x(), p.get_height() * 1.05))

    f1.tight_layout()
    f2.tight_layout()

    f_oneway(tl[0][~np.isnan(tl[0])], tl[3][~np.isnan(tl[3])])
    midx = 0
    f_oneway(fulldgr[midx][:,0], fulldgr[midx][:,1],fulldgr[midx][:,2])

    for midx in range(5):
        t,p = ttest_ind(fulldgr[midx][:,2],fulldgr[midx][:,0], alternative='greater')
        print(np.round(t,3),np.round(p,4))


    #savefig('./Fig/1pa/1pa_latency',f1)
    #savefig('./Fig/1pa/1pa_dgr', f2)


elif int(pltfig) == 3:
    genvar_path = '../1pa/Data/'
    #genvar_path = 'D:/Ganesh_PhD/s4o/1pa/'
    from backend.maze import Static_Maze
    from backend.utils import get_default_hp
    grad_col = np.zeros([5,3,3])+0.5
    for m,mc in enumerate(modelcolor):
        grad_col[m,-1] = matplotlib.colors.to_rgb(mc)
        grad_col[m, 1] = 0.25
        grad_col[m, 0] = 0.5
        #grad_col[m,0] = (grad_col[m,1]+grad_col[m,-1])/2

    hp = get_default_hp(task='center', platform='laptop')
    hp['obstype'] = 'cave'
    env = Static_Maze(hp)
    env.make('square', rloc=24)
    fulltraj = False
    trajcol = ['tab:blue','tab:green','tab:red']
    trajalpha = [0.5,0.7,0.9]
    #trajalpha = [0.5,0.7,0.9]
    lw = 1
    tnum=0

    for m,model in enumerate(allmodels):
        #trajcol = grad_col[m]
        model_data = glob.glob(genvar_path+'genvars_*{}*b96*'.format(model))

        [_, _, totpath] = saveload('load',model_data[-1],1)

        # a = np.linalg.norm(totpath,axis=4)
        # criterias = 0
        # for i in range(96):
        #     for p in range(3):
        #         for t in range(6):
        #             dist = a[i,p,t,:1000]
        #             hit = np.argmax(dist<0.03)
        #             if m==0:
        #                 if p==2 and hit>0:
        #
        #
        #
        #             if m == 1 or m ==2:
        #                 if



        ridx = np.random.choice(np.arange(totpath.shape[0]), 20, replace=False)
        mvpaths = totpath[ridx]

        f = plt.figure(figsize=(10,8))
        for n in range(20):
            hitr = []
            stl = []
            time = []
            mvpath = mvpaths[n]
            for i in range(3):
                for p in range(6):
                    if m == 1 or m == 2:
                        hitr.append(np.argmax(np.linalg.norm(mvpath[i, p], axis=1) > 0.3) - 1)
                    # elif i == 1:
                    #     hitr.append(np.argmax(np.linalg.norm(mvpath[i, p], axis=1) < 0.1) - 1)
                    else:
                        hitr.append(np.argmax(np.linalg.norm(mvpath[i, p], axis=1) < 0.03) - 1)

                    stl.append((mvpath[i, p, 0] != np.array([0, -0.8])).any())
            hitr = np.reshape(hitr, (3, 6))
            stl = np.reshape(stl, (3, 6))
            pidx = []
            for i in range(3):
                pidx.append(np.argmax(stl[i] * hitr[i] > 0))

            ax = plt.subplot(4,5,n+1)
            for i in range(3):

                for j in range(len(env.obstacles)):
                    circle = plt.Circle(env.obstacles[j], env.obssz, color='k', zorder=1)
                    ax.add_artist(circle)

                traj = mvpaths[n, i,pidx[i]]

                if fulltraj:
                    idx = -1
                else:
                    idx = np.argmax(np.linalg.norm(np.zeros(2) - traj, axis=1) < 0.03) - 1

                ax.plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=trajcol[i], alpha=trajalpha[i], zorder=i+1, linewidth=lw)
                ax.plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=trajcol[i], alpha=trajalpha[i], zorder=i+1, linewidth=lw)

                ax.plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[i], alpha=trajalpha[i], zorder=i+1, linewidth=lw)
            circle = plt.Circle(env.rloc, env.rrad, color='red', zorder=10)
            ax.add_artist(circle)
            circle1 = plt.Circle(env.rloc, env.rrad, color='k', zorder=11, fill=False)
            ax.add_artist(circle1)
            ax.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
        f.tight_layout()

        f.text(0.01,0.01,model)
        savefig('./Fig/1pa/1pa_traj{}_{}_col.png'.format(tnum,model[:7]),f)


elif int(pltfig) == 4:
    #genvar_path = '../1pa/Data/sharm/'
    genvar_path = 'D:/Ganesh_PhD/s4o/1pa/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    hp = get_default_hp(task='center', platform='laptop')

    bins = 15  # 7, 11, 15, 21, 27
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

        policy = np.zeros([3, 2,bins,bins])
        value = np.zeros([3, bins,bins])

        for d in range(96):
            #print(model_data[d])

            [alldyn, _, mvpath, lat, dgr] = saveload('load',model_data[d],1)

            qdyn = alldyn[1]
            cdyn = alldyn[2]
            nonrlen = np.array(qdyn[list(qdyn.keys())[0]]).shape[0]
            trials = [2, 6, 10]
            cues = 6
            for p in range(3):
                trial = trials[p]
                qfr = np.zeros([cues, nonrlen, 40])
                cfr = np.zeros([cues, nonrlen, 2])
                coord = np.zeros([nonrlen * cues, 2])

                sess = [v for v in cdyn.keys() if v.startswith('square_s{}'.format(trial))]
                for c, s in enumerate(sess):
                    qfr[c] = np.array(qdyn[s])[-nonrlen:]
                    cfr[c] = np.array(cdyn[s])[-nonrlen:]

                qfr = np.reshape(qfr, newshape=(cues * nonrlen, 40))
                cfr = np.reshape(cfr, newshape=(cues * nonrlen, 2))

                for i, s in enumerate(sess):
                    st = i * nonrlen
                    ed = st + nonrlen
                    coord[st:ed] = mvpath[p, i][-nonrlen:]

                policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr, bins=bins)

                policy[p] += policymap
                value[p] += valuemap

        value /=N
        policy /= N
        saveload('save', './Data/1pa/maps_{}'.format(model), 1)

        f = plt.figure()
        for p in range(3):
            plt.subplot(2,2,p+1)
            im = plt.imshow(value[p].T, aspect='auto', origin='lower')

            #plt.quiver(newx[:, 1], newx[:, 0], policy[p,1].reshape(bins ** 2), policy[p,0].reshape(bins ** 2), units='xy', color='w')
            plt.quiver(newx[:, 1], newx[:, 0], policy[p, 0].reshape(bins ** 2), policy[p, 1].reshape(bins ** 2), units='xy', color='w')

            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            plt.xticks([], [])
            plt.yticks([], [])
            plt.gca().set_aspect('equal', adjustable='box')

        f.tight_layout()
        f.text(0.01,0.01,'{}_{}b'.format(model[:7],N))
        f.savefig('./Fig/1pa/1pa_map_{}_{}_b{}_alt.png'.format(model[:7],bins, N))
        f.savefig('./Fig/1pa/1pa_map_{}_{}_b{}_alt.svg'.format(model[:7],bins,N))
        plt.close()









