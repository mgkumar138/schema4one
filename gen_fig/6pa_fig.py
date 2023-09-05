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

showall = False
if showall:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl','sym_0.5cb','res_0.5*Truesl']  # LR, EH, Sym
    #modellegend = ['ActorCritic','Sym. NAVIGATE', 'Neural NAVIGATE', 'AC+Sym. NAVGATE', 'AC+Neural NAVIGATE']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural','AC+Sym','AC+Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange', 'magenta','limegreen']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']

pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    #genvar_path = '../6pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/6pa/'
    totiter = 720

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

        totlat = np.concatenate(totlat,axis=0)
        totdgr = np.concatenate(totdgr,axis=0)
        totpi = np.concatenate(totpi,axis=0)

        if len(totdgr) > totiter:
            lidx = np.random.choice(np.arange(len(totdgr)), totiter, replace=False)
        else:
            lidx = np.arange(len(totdgr))

        saveload('save','./Data/6pa/comb_genvar_6pa_{}_{}b'.format(model[:7], len(lidx)), [totlat[lidx], totdgr[lidx],totpi[lidx]])

elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    genvar_path = './Data/6pa/'
    scalet = 20/1000

    tl, td = [], []
    dfm, dfs = [], []
    pfm, pfs = [], []
    fulllat = []
    fulldgr = []
    z = 1.96
    for idx, model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'comb_genvar_6pa_{}*'.format(model[:7]))

        [totlat, totdgr, totpi] = saveload('load',model_data[0],1)

        fulllat.append(totlat)
        fulldgr.append(totdgr)

        tl.append(np.mean(totlat*scalet, axis=0))
        #td.append(np.array([np.nanquantile(totlat*scalet, 0.25, axis=0),np.nanquantile(totlat*scalet, 0.75, axis=0)]))

        td.append(np.array([np.mean(totlat * scalet, axis=0) - z*np.std(totlat*scalet, axis=0)/np.sqrt(len(totdgr)),
                            np.mean(totlat * scalet, axis=0) + z*np.std(totlat*scalet, axis=0)/np.sqrt(len(totdgr))]))

        dfm.append(np.mean(totdgr, axis=0))
        dfs.append(z*np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))

        pfm.append(np.mean(totpi, axis=0))
        pfs.append(z*np.std(totpi, axis=0)/np.sqrt(len(totpi)))

    f1,ax1 = plt.subplots(1,1,figsize=(4,2))
    for idx in range(len(modellegend)):
        ax1.plot(np.arange(1, 1 + 20), tl[idx][:20], color=modelcolor[idx], marker='o', ms=5, linewidth=1)
    ax1.legend(modellegend, loc='upper right', fontsize=6, frameon=False)
    for idx in range(len(modellegend)):
        ax1.fill_between(np.arange(1, 1 + 20), td[idx][0][:20], td[idx][1][:20], alpha=0.1, facecolor=modelcolor[idx])
        ax1.fill_between(np.linspace(0.75, 1.4,2), [td[idx][0][0],td[idx][0][0]], [td[idx][1][0],td[idx][1][0]], alpha=0.1, facecolor=modelcolor[idx])
    ax1.set_xlabel('Session')
    ax1.set_ylabel('Average latency (s)')
    ax1.set_title('Average time to reach correct target')
    ax1.set_xticks(np.linspace(1,20,10,dtype=int),np.linspace(1,20,10,dtype=int))
    ax1.set_ylim(0, 300)
    sessidx = [2,9,16]
    for pt in range(3):
        ax1.annotate('PS {}'.format(pt + 1), (sessidx[pt], 80), rotation=90)

    tp = np.zeros([3,6,2])
    for s in range(6):
        for m in range(3):
            tp[m,s] = ttest_1samp(fulldgr[m][:,s],100/6,alternative='greater')

    f2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(dfm).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax2, yerr=ds, legend=True, color=modelcolor)
    ax2.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax2.axhline(100 / 6, color='r', linestyle='--')
    ax2.set_ylabel('Visit ratio (%)')
    ax2.set_title('Ratio of time at correct target')
    ax2.set_ylim(0, 105)

    f3, ax3 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(dfm).T[3:7], index=['OPA', '2NPA', '6NPA', 'NM'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T[3:7], index=['OPA', '2NPA', '6NPA', 'NM'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax3, yerr=ds, legend=False, color=modelcolor)
    ax3.legend(modellegend, fontsize=6, loc='upper right', frameon=False)
    ax3.axhline(100 / 6, color='r', linestyle='--')
    ax3.set_ylabel('Visit ratio (%)')
    ax3.set_title('Ratio of time at correct target after 1 session')
    ax3.set_ylim(0, 105)

    f4, ax4 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(pfm).T[3:7], index=['OPA', '2NPA', '6NPA', 'NM'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(pfs).T[3:7], index=['OPA', '2NPA', '6NPA', 'NM'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax4, yerr=ds, legend=False, color=modelcolor)
    #ax3.axhline(100 / 6, color='r', linestyle='--')
    #ax3.legend(modellegend, loc='upper left', fontsize=6)
    ax4.set_ylabel('Performance index (%)')
    ax4.set_title('Number of paired associations learned')
    for p in ax4.patches:
        ax4.annotate(str(np.round(p.get_height(),1)), (p.get_x() * 1.005, p.get_height() + 0.2))

    # NM
    f6,ax6 = plt.subplots(1,1,figsize=(4,2))
    for idx in range(len(modellegend)):
        ax6.plot(np.arange(21, 21 + 22), tl[idx][-22:], color=modelcolor[idx], marker='o', ms=5, linewidth=1)
    ax6.legend(modellegend, loc='upper right', fontsize=6, frameon=False)
    for idx in range(len(modellegend)):
        ax6.fill_between(np.arange(21, 21 + 22), td[idx][0][-22:], td[idx][1][-22:], alpha=0.1, facecolor=modelcolor[idx])
        ax6.fill_between(np.linspace(20.75, 21.4,2), [td[idx][0][-22],td[idx][0][-22]], [td[idx][1][-22],td[idx][1][-22]], alpha=0.1, facecolor=modelcolor[idx])
    ax6.set_xlabel('Session')
    ax6.set_ylabel('Average latency (s)')
    ax6.set_title('Average time to reach correct target in New Maze')
    ax6.set_xticks(np.linspace(21,41,10,dtype=int),np.linspace(21,41,10,dtype=int))
    ax6.set_ylim(0, 600)
    sessidx = [22,29,36]
    for pt in range(3):
        ax6.annotate('NM {}'.format(pt + 1), (sessidx[pt], 210), rotation=90)

    f5, ax5 = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.vstack(dfm).T[6:], index=['NM1', 'NM2', 'NM3', '6NPANM'], columns=modellegend)
    ds = pd.DataFrame(np.vstack(dfs).T[6:], index=['NM1', 'NM2', 'NM3', '6NPANM'], columns=modellegend)
    df.plot.bar(rot=0, ax=ax5, yerr=ds, legend=True, color=modelcolor)
    ax5.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax5.axhline(100 / 6, color='r', linestyle='--')
    ax5.set_ylabel('Visit ratio (%)')
    ax5.set_title('Ratio of time at correct target in New Maze')
    ax5.set_ylim(0, 105)

    f1.tight_layout()
    f2.tight_layout()
    f3.tight_layout()
    f4.tight_layout()
    f5.tight_layout()
    f6.tight_layout()

    # savefig('./Fig/6pa/6pa_latency',f1)
    # savefig('./Fig/6pa/6pa_ps_dgr', f2)
    # savefig('./Fig/6pa/6pa_npa', f3)
    # savefig('./Fig/6pa/6pa_pi', f4)
    # savefig('./Fig/6pa/6pa_nm_dgr', f5)
    # savefig('./Fig/6pa/6pa_nm_lat', f6)

elif int(pltfig) == 3:
    genvar_path = 'D:/Ganesh_PhD/s4o/6pa/'
    from backend.maze import Maze
    from backend.utils import get_default_hp

    mpl.rcParams['axes.linewidth'] = 2

    hp = get_default_hp(task='6pa', platform='laptop')

    mtype = ['train','train','train','opa','2npa','6npa','nm','nm','nm', '6nm']
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA','NM1','NM2','NM3', '6NPANM']
    fulltraj = False
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    trajcol4 = ['blue', 'black', 'green', 'red', 'yellow', 'purple']
    col = [trajcol1,trajcol1,trajcol1,trajcol1,trajcol2,trajcol3,trajcol3,trajcol3,trajcol3, trajcol4]
    trajalpha = 0.8
    tnum = 1
    pidx = 10

    allridx = np.zeros([3,tnum],dtype=int)
    for m,model in enumerate(allmodels):
        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)

        ridx = []
        for d in np.arange(N)[::-1]:
            if len(ridx)>=tnum:
                print('model {} loaded'.format(model))
                break
            else:
                [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load', model_data[d], 1)
                print(np.round(dgr[np.array([3,4,5,8,9])]))

                if m == 0 and dgr[3]>90 and dgr[4]<17 and dgr[8] > 40 and dgr[9]<17:
                    ridx.append(d)
                elif m == 1 and dgr[3]>90 and dgr[4]>90 and dgr[5]>90 and dgr[8]>90 and dgr[9]>80:
                    ridx.append(d)
                elif m == 2 and dgr[3] > 60 and dgr[4] > 60 and dgr[5] > 60 and dgr[8]>65 and dgr[9]>65: # 75 70 50 65 65
                    ridx.append(d)

        allridx[m] = np.random.choice(ridx,tnum, replace=False)

    for t in range(tnum):

        f, ax = plt.subplots(nrows=3, ncols=7, figsize=(8, 3.5))
        f2, ax2 = plt.subplots(nrows=3, ncols=4, figsize=(5, 3.5))

        for m, model in enumerate(allmodels):
            model_data = glob.glob(genvar_path + 'vars_*{}*'.format(model))

            didx = allridx[m,t]
            df = model_data[didx]
            print(df)
            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',df,1)

            j = 0
            for i in range(10):  # mazetype
                env = Maze(hp)
                env.make(mtype[i])
                trajcol = col[i]

                if i >=6:
                    ax2[m,j].spines['bottom'].set_color('m')
                    ax2[m,j].spines['top'].set_color('m')
                    ax2[m,j].spines['right'].set_color('m')
                    ax2[m,j].spines['left'].set_color('m')
                    for c in range(6):  # cue
                        traj = mvpath[i, c]
                        rloc = env.rlocs[c]

                        if fulltraj:
                            idx = -1
                        else:
                            idx = np.argmax(np.linalg.norm(rloc - traj, axis=1) < 0.1) - 1

                        ax2[m,j].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s', color=trajcol[c],
                                      alpha=trajalpha, zorder=2, ms=2)
                        ax2[m,j].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1], marker='X', color=trajcol[c],
                                      alpha=trajalpha, zorder=2, ms=2)

                        ax2[m,j].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[c],
                                      alpha=trajalpha, zorder=1, linewidth=0.75)
                        circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c], zorder=3)
                        ax2[m,j].add_artist(circle)
                        circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k', zorder=3, fill=False)
                        ax2[m,j].add_artist(circle2)

                    ax2[m,j].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
                    ax2[m,j].set_aspect('equal', adjustable='box')
                    ax2[m,j].set_xticks([])
                    ax2[m,j].set_yticks([])
                    ax2[m,0].set_ylabel(modellegend[m], fontsize=12)
                    ax2[0,j].set_title(mlegend[i], fontsize=12)
                    j+=1
                if i <=6:
                    if i == 6:
                        ax[m,i].spines['bottom'].set_color('m')
                        ax[m,i].spines['top'].set_color('m')
                        ax[m,i].spines['right'].set_color('m')
                        ax[m,i].spines['left'].set_color('m')

                    for c in range(6):  # cue
                        traj = mvpath[i, c]
                        rloc = env.rlocs[c]

                        if fulltraj:
                            idx = -1
                        else:
                            idx = np.argmax(np.linalg.norm(rloc - traj, axis=1) < 0.1) - 1

                        ax[m,i].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=trajcol[c], alpha=trajalpha,zorder=2,ms=2)
                        ax[m,i].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=trajcol[c], alpha=trajalpha,zorder=2, ms=2)

                        ax[m,i].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[c], alpha=trajalpha,zorder=1,linewidth=0.75)
                        circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c],zorder=3)
                        ax[m,i].add_artist(circle)
                        circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k',zorder=3,fill=False)
                        ax[m,i].add_artist(circle2)

                    ax[m,i].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
                    ax[m,i].set_aspect('equal', adjustable='box')
                    ax[m,i].set_xticks([])
                    ax[m,i].set_yticks([])
                    ax[m,0].set_ylabel(modellegend[m], fontsize=12)
                    #ax[m, 6].set_ylabel(modellegend[m], fontsize=8)
                    ax[0,i].set_title(mlegend[i], fontsize=12)

        f.tight_layout()
        f2.tight_layout()
        # f.text(0.01,0.01,tnum)
        #savefig('./Fig/6pa/6pa_traj{}'.format(tnum+pidx), f)
        savefig('./Fig/6pa/6pa_nm_traj{}'.format(tnum+pidx), f2)
        plt.close(f)
        plt.close(f2)


elif int(pltfig) == 4:
    #genvar_path = '../6pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/6pa/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    from backend.maze import Maze

    mtype = ['train','train','train','opa','2npa','6npa','nm','nm','nm']

    hp = get_default_hp(task='6pa', platform='laptop')
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA', 'NM1', 'NM2', 'NM3']

    bins = 15
    varN = 96
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

            policy = np.zeros([9,6, 2,bins,bins])
            value = np.zeros([9,6, bins,bins])

            ridx = np.random.choice(np.arange(N),varN, replace=False)

            for d in ridx:

                [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',model_data[d],1)

                sess = list(alldyn[1].keys())

                for p in range(9):
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
            saveload('save', './Data/6pa/var_6pa_maps_{}_b{}'.format(model[:7],varN), [value, policy])

            f,ax = plt.subplots(9,6,figsize=(10,10))
            for p in range(9):
                env = Maze(hp)
                env.make(mtype[p])
                for c in range(6):
                    im = ax[p,c].imshow(value[p,c].T, aspect='auto', origin='lower')
                    ax[p,c].quiver(newx[:, 1], newx[:, 0], policy[p,c,0].reshape(bins ** 2), policy[p,c,1].reshape(bins ** 2),
                               units='xy', color='w')
                    plt.colorbar(im,ax=ax[p,c], fraction=0.046, pad=0.04)
                    ax[p,c].set_xticks([], [])
                    ax[p,c].set_yticks([], [])
                    ax[p,c].set_aspect('equal', adjustable='box')

                    ax[0,c].set_title('C{}'.format(c+1), fontsize=10)
                    ax[p, 0].set_ylabel('{}'.format(mlegend[p]), fontsize=10)

            f.tight_layout()
            f.text(0.01,0.01,model[:7])
            savefig('./Fig/6pa/6pa_maps_full_{}_b{}'.format(model[:7],varN), f)
            plt.close()


elif int(pltfig) == 5:
    #genvar_path = '../6pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/Schema4PA/6pa/'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    from backend.utils import get_default_hp

    hp = get_default_hp(task='6pa', platform='laptop')

    for m,model in enumerate([allmodels[-1]]):
        model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
        N = len(model_data)
        print(N)

        diffw = np.zeros([N, 4, 3])  # compare opa, npa, nw against trw

        for d in range(N):

            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',model_data[d],1)

            [trw, opaw, npaw, nmw] = allw

            for j,nw in enumerate([0,1,2,5]):
                for i,newweights in enumerate([opaw,npaw,nmw]):
                    diffw[d, j,i] = np.linalg.norm(newweights[nw]-trw[nw],ord=2)

    wlegend = ['XY','Goal','Critic', 'Actor']
    collegend = ['OPA','2NPA','6NPA']

    dw = np.zeros_like(diffw)
    dw[:,0] = diffw[:,0]
    dw[:,1] = diffw[:, 3]
    dw[:,2] = diffw[:, 1]
    dw[:,3] = diffw[:, 2]

    f, ax = plt.subplots(1, 1, figsize=(4, 2))
    df = pd.DataFrame(np.mean(dw,axis=0), index=wlegend, columns=collegend)
    ds = pd.DataFrame(np.std(dw,axis=0)/N**0.5, index=wlegend, columns=collegend)
    df.plot.bar(rot=0, ax=ax, yerr=ds, legend=True)
    ax.legend(collegend, loc='upper right', fontsize=6)
    ax.set_ylabel('Weight change (L2)')
    ax.set_title('Weights after 1 session of new learning')
    f.tight_layout()

    f.savefig('./Fig/6pa/6pa_dw_b{}.png'.format(N))
    f.savefig('./Fig/6pa/6pa_dw_b{}.svg'.format(N))

elif int(pltfig) == 6:
    from backend.maze import Maze
    from backend.utils import get_default_hp

    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol2,trajcol3]

    mtype = ['opa','2npa','6npa']
    mlegend = ['OPA', '2NPA', '6NPA']

    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 4))

    for d in range(2):
        hp = get_default_hp(task='6pa', platform='laptop')
        if d == 1:
            hp['obs'] = True
        else:
            hp['obs'] = False

        for i in range(3):
            env = Maze(hp)
            env.make(mtype[i])
            trajcol = col[i]

            for c in range(6):
                rloc = env.rlocs[c]

                circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c],zorder=3)
                ax[d,i].add_artist(circle)
                #circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k',zorder=3,fill=False)
                #ax[d,i].add_artist(circle2)

                if d == 1:
                    for ld in env.obstacles:
                        circle3 = plt.Circle(ld, env.obssz, color='k', zorder=1)
                        ax[d,i].add_artist(circle3)

            ax[d,i].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
            ax[d,i].set_aspect('equal', adjustable='box')
            ax[d,i].set_xticks([])
            ax[d,i].set_yticks([])
            ax[d,i].set_title(mlegend[i])
            f.tight_layout()

        f.savefig('./Fig/6pa/6pa_maze_nooutline.png')
        f.savefig('./Fig/6pa/6pa_maze_nooutline.svg')


elif int(pltfig) == 7:

    mlegend = ['OPA \n Cue 2', '2NPA \n Cue 7', '6NPA \n Cue 16']
    midx = np.array([[3,1],[4,0],[5,5]])
    varN = 96
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

    f, ax = plt.subplots(3, 3, figsize=(3, 3.5))
    for m,model in enumerate(allmodels):
        [value, policy] = saveload('load', './Data/6pa/var_6pa_maps_{}_b{}.pickle'.format(model[:7],varN), 1)
        for p in range(3):
            val = value[midx[p,0], midx[p,1]]
            pol = policy[midx[p,0], midx[p,1]]
            im = ax[m,p].imshow(val.T, aspect='auto', origin='lower')
            ax[m,p].quiver(newx[:, 1], newx[:, 0], pol[0].reshape(bins ** 2), pol[1].reshape(bins ** 2), units='xy', color='w')
            cbar = plt.colorbar(im, ax=ax[m,p], fraction=0.046, pad=0.04,orientation="horizontal") #,orientation="horizontal",
            cbar.ax.tick_params(labelsize=8)
            ax[m,p].set_xticks([], [])
            ax[m,p].set_yticks([], [])
            ax[m,p].set_aspect('equal', adjustable='box')
            ax[m, 0].set_ylabel(modellegend[m], fontsize=8)
            ax[0, p].set_title(mlegend[p], fontsize=8)

    f.tight_layout()
    savefig('./Fig/6pa/6pa_cuemap_vert_b{}'.format(varN), f)


elif int(pltfig) == 8:

    from backend.maze import Maze
    from backend.utils import get_default_hp

    hp = get_default_hp(task='6pa', platform='laptop')
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol2,trajcol3]
    mtype = ['opa','2npa','6npa']
    env = Maze(hp)

    mlegend = ['OPA \n Cue 2', '2NPA \n Cue 7', '6NPA \n Cue 16']
    midx = np.array([[3,1],[4,0],[5,5]])
    varN = 96
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

    f = plt.figure(figsize=(3, 3.5))
    i = 0
    for m,model in enumerate(allmodels):
        [value, policy] = saveload('load', './Data/6pa/var_6pa_maps_{}_b{}.pickle'.format(model[:7],varN), 1)
        for p in range(3):
            i+=1
            val = value[midx[p,0], midx[p,1]]
            pol = policy[midx[p,0], midx[p,1]]

            ax2 = f.add_subplot(3,3,i,label='map')
            im = ax2.imshow(val.T, aspect='auto', origin='lower')
            ax2.quiver(newx[:, 1], newx[:, 0], pol[0].reshape(bins ** 2), pol[1].reshape(bins ** 2), units='xy', color='w')
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,orientation="horizontal") #,orientation="horizontal",
            cbar.ax.tick_params(labelsize=8)

            ax3 = f.add_subplot(3,3,i, label='loc', frame_on=False)
            env.make(mtype[p])
            trajcol = col[p]
            for c in range(6):
                rloc = env.rlocs[c]
                circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c], zorder=3)
                ax3.add_artist(circle)

            ax2.set_xticks([], [])
            ax2.set_yticks([], [])
            ax2.set_aspect('equal', adjustable='box')

            ax3.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
            ax3.set_aspect('equal', adjustable='box')
            ax3.set_xticks([], [])
            ax3.set_yticks([], [])

            if i==1 or i==4 or i ==7:
                ax2.set_ylabel(modellegend[m], fontsize=8)
            if i <4:
                ax2.set_title(mlegend[p], fontsize=8)

    f.tight_layout()
    savefig('./Fig/6pa/6pa_cuemap_vert_b{}'.format(varN), f)