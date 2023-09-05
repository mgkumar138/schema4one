import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from backend.utils import saveload, savefig, get_savings
import glob
import os
from scipy.stats import ttest_ind_from_stats,ttest_1samp,linregress, binned_statistic_2d, f_oneway, ttest_ind
import matplotlib as mpl
import os
#import pingouin as pg

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

showall = True
if showall:
    allmodels = ['0dist*_da*0mlr', '1dist*_da*0mlr', '1dist*_da*1mlr','2dist*_da*0mlr','2dist*_da*1mlr']  # LR, EH, Sym
    #allmodels = ['0dist*etda*0mlr', '1dist*etda*0mlr', '1dist*etda*1mlr', '2dist*etda*0mlr','2dist*etda*1mlr']  # LR, EH, Sym
    #modellegend = ['ActorCritic','Sym. NAVIGATE', 'Neural NAVIGATE', 'AC+Sym. NAVGATE', 'AC+Neural NAVIGATE']
    modellegend = ['0D, -DA', '1D, -DA', '1D, +DA','2D -DA','2D +DA']
    modelcolor = ['green', 'deepskyblue', 'gold', 'blue','red']
else:
    allmodels = ['sym_0cb', 'sym_1cb', 'res_1cb*Truesl']  # LR, EH, Sym
    #modellegend = [r'$\beta=0$', r'S. $\beta=1$', r'N. $\beta=1$']
    modellegend = ['ActorCritic', 'Symbolic', 'Neural']
    modelcolor = ['deeppink', 'dodgerblue', 'orange']

pltfig = 2 #input('1)comb data 2) latency, Visit Ratio 3) trajectory 4) Maps')

# plot latency
if int(pltfig) == 1:
    genvar_path = 'D:/Ganesh_PhD/s4o/wkm/'
    totiter = 624

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

        newstr = model.replace("*", "_")
        saveload('save','./Data/wkm/comb_genvar_workmem_{}_{}b'.format(newstr, len(lidx)), [totlat[lidx], totdgr[lidx],totpi[lidx]])

elif int(pltfig) == 2:
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    genvar_path = './Data/wkm/'
    scalet = 20/1000

    tl, td = [], []
    dfm, dfs = [], []
    pfm, pfs = [], []
    alldgr = []
    fulllat = []
    z = 1.96
    for idx, model in enumerate(allmodels):
        newstr = model.replace("*", "_")
        model_data = glob.glob(genvar_path+'comb_genvar_workmem_{}*'.format(newstr))

        [totlat, totdgr, totpi] = saveload('load',model_data[0],1)

        fulllat.append(totlat)
        tl.append(np.mean(totlat[:,:20]*scalet, axis=0))
        #td.append(np.array([np.nanquantile(totlat[:,:20]*scalet, 0.25, axis=0),np.nanquantile(totlat[:,:20]*scalet, 0.75, axis=0)]))

        td.append(np.array([np.mean(totlat[:,:20] * scalet, axis=0) - z*np.std(totlat[:,:20]*scalet, axis=0)/np.sqrt(len(totdgr)),
                            np.mean(totlat[:,:20] * scalet, axis=0) + z*np.std(totlat[:,:20]*scalet, axis=0)/np.sqrt(len(totdgr))]))

        dfm.append(np.mean(totdgr, axis=0))
        dfs.append(z*np.std(totdgr, axis=0)/np.sqrt(len(totdgr)))

        pfm.append(np.mean(totpi, axis=0))
        pfs.append(z*np.std(totpi, axis=0)/np.sqrt(len(totpi)))

        alldgr.append(totdgr)

    from scipy.stats import ttest_1samp
    stt = np.zeros([5,7,2])
    opt = np.zeros_like(stt)
    for t in range(7):
        for m in range(5):
            stt[m,t] = ttest_1samp(alldgr[m][:,t], 100/6,alternative='two-sided')
            opt[m, t] = ttest_1samp(alldgr[m][:, t], np.mean(alldgr[0][:, t]), alternative='two-sided')

    twonpa = ttest_ind_from_stats(np.mean(alldgr[1][:, 4], axis=0), np.std(alldgr[1][:, 4], axis=0), alldgr[1].shape[0],
                         np.mean(alldgr[2][:, 4], axis=0), np.std(alldgr[2][:, 4], axis=0), alldgr[2].shape[0],
                         alternative='two-sided')
    sixnpa_onedist = ttest_ind_from_stats(np.mean(alldgr[1][:, 5], axis=0), np.std(alldgr[1][:, 5], axis=0), alldgr[1].shape[0],
                         np.mean(alldgr[2][:, 5], axis=0), np.std(alldgr[2][:, 5], axis=0), alldgr[2].shape[0],
                         alternative='two-sided')

    sixnpa_twodist = ttest_ind_from_stats(np.mean(alldgr[3][:, 5], axis=0), np.std(alldgr[3][:, 5], axis=0), alldgr[3].shape[0],
                         np.mean(alldgr[4][:, 5], axis=0), np.std(alldgr[4][:, 5], axis=0), alldgr[4].shape[0],
                         alternative='two-sided')

    f1,ax1 = plt.subplots(1,1,figsize=(4,2))
    for idx in range(len(modellegend)):
        ax1.plot(np.arange(1, 1 + 20), tl[idx], color=modelcolor[idx], marker='o',ms=2, linewidth=1)
    #ax1.legend(modellegend, loc='upper right', fontsize=6)
    for idx in range(len(modellegend)):
        ax1.fill_between(np.arange(1, 1 + 20), td[idx][0], td[idx][1], alpha=0.1, facecolor=modelcolor[idx])
        ax1.fill_between(np.linspace(0.75, 1.4,2), [td[idx][0][0],td[idx][0][0]], [td[idx][1][0],td[idx][1][0]], alpha=0.1, facecolor=modelcolor[idx])
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title('Average time to reach multiple target')
    ax1.set_xticks(np.linspace(1,20,10,dtype=int),np.linspace(1,20,10,dtype=int))
    ax1.set_ylim(0, 320)
    sessidx = [2,9,16]
    for pt in range(3):
        ax1.annotate('PS {}'.format(pt + 1), (sessidx[pt], 80), rotation=90)

    f2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    dfps = pd.DataFrame(np.vstack(dfm).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    dsps = pd.DataFrame(np.vstack(dfs).T[:3], index=['PS1', 'PS2', 'PS3'], columns=modellegend)
    dfps.plot.bar(rot=0, ax=ax2, yerr=dsps, legend=False, color=modelcolor)
    ax2.legend(modellegend, loc='upper left', fontsize=6, frameon=False)
    ax2.axhline(100/6,color='r',linestyle='--')
    ax2.set_ylabel('Visit ratio (%)')
    ax2.set_title('Ratio of time at correct target')
    ax2.set_ylim(0, 80)

    midx = 2
    f_oneway(alldgr[midx][:,0],alldgr[midx][:,1],alldgr[midx][:,2],alldgr[midx][:,3])

    # ti = 0
    # for p,t in zip(ax2.patches,stt[:,:3].reshape([15,2])):
    #     if t[0] > 0:
    #         print(p.get_x(), t[0])
    #         if t[1] < 0.0001:
    #             ax2.annotate('****', (p.get_x(), p.get_height() * 1.05))
    #         if t[1] < 0.001:
    #             ax2.annotate('***', (p.get_x(), p.get_height() * 1.05))
    #         elif t[1] < 0.01:
    #             ax2.annotate('**', (p.get_x(), p.get_height() * 1.05))
    #         elif t[1] < 0.05:
    #             ax2.annotate('*', (p.get_x(), p.get_height() * 1.05))

    f3, ax3 = plt.subplots(1, 1, figsize=(4, 2))
    dfnp = pd.DataFrame(np.vstack(dfm).T[3:], index=['OPA', '2NPA', '6NPA', 'NM'], columns=modellegend)
    dsnp = pd.DataFrame(np.vstack(dfs).T[3:], index=['OPA', '2NPA', '6NPA','NM'], columns=modellegend)
    dfnp.plot.bar(rot=0, ax=ax3, yerr=dsnp, legend=False, color=modelcolor)
    ax3.legend(modellegend, loc='upper right', fontsize=6, frameon=False)
    ax3.axhline(100 / 6, color='r', linestyle='--')
    ax3.set_ylabel('Visit ratio (%)')
    ax3.set_title('Ratio of time at correct target after 1 session')
    ax3.set_ylim(0, 80)

    # ti = 0
    # for p,t in zip(ax3.patches,stt[:,3:].reshape([20,2])):
    #     if t[0] > 0:
    #         print(p.get_x(), t[0])
    #         if t[1] < 0.0001:
    #             ax3.annotate('****', (p.get_x(), p.get_height() * 1.05))
    #         if t[1] < 0.001:
    #             ax3.annotate('***', (p.get_x(), p.get_height() * 1.05))
    #         elif t[1] < 0.01:
    #             ax3.annotate('**', (p.get_x(), p.get_height() * 1.05))
    #         elif t[1] < 0.05:
    #             ax3.annotate('*', (p.get_x(), p.get_height() * 1.05))

    f4, ax4 = plt.subplots(1, 1, figsize=(4, 2))
    dfpi = pd.DataFrame(np.vstack(pfm).T[3:], index=['OPA', '2NPA', '6NPA','NM'], columns=modellegend)
    dspi = pd.DataFrame(np.vstack(pfs).T[3:], index=['OPA', '2NPA', '6NPA','NM'], columns=modellegend)
    dfpi.plot.bar(rot=0, ax=ax4, yerr=dspi, legend=False, color=modelcolor)
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

    # savefig('./Fig/wkm/wkm_latency',f1)
    # savefig('./Fig/wkm/wkm_dgr', f2)
    # savefig('./Fig/wkm/wkm_npa', f3)
    # savefig('./Fig/wkm/wkm_pi', f4)

elif int(pltfig) == 3:
    genvar_path = 'D:/Ganesh_PhD/Schema4PA/6pa/'
    from backend.maze import Maze
    from backend.utils import get_default_hp

    hp = get_default_hp(task='6pa', platform='laptop')

    mtype = ['train','train','train','opa','2npa','6npa']
    mlegend = ['PS1', 'PS2', 'PS3', 'OPA', '2NPA', '6NPA']
    fulltraj = False
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
        ridx = np.random.choice(np.arange(N),5, replace=False)

        f,ax = plt.subplots(nrows=5, ncols=6, figsize=(10, 8))

        for d, didx in enumerate(ridx):
            df = model_data[didx]
            print(df)
            [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',df,1)

            for i in range(6):
                env = Maze(hp)
                env.make(mtype[i])
                trajcol = col[i]

                for c in range(6):
                    traj = mvpath[i, c]
                    rloc = env.rlocs[c]

                    if fulltraj:
                        idx = -1
                    else:
                        idx = np.argmax(np.linalg.norm(rloc - traj, axis=1) < 0.1) - 1

                    ax[d,i].plot(np.array(traj)[0, 0], np.array(traj)[0, 1], marker='s',color=trajcol[c], alpha=trajalpha,zorder=2)
                    ax[d,i].plot(np.array(traj)[idx, 0], np.array(traj)[idx, 1],  marker='X',color=trajcol[c], alpha=trajalpha,zorder=2)

                    ax[d,i].plot(np.array(traj)[:idx, 0], np.array(traj)[:idx, 1], color=trajcol[c], alpha=trajalpha,zorder=1,linewidth=0.75)
                    circle = plt.Circle(env.rlocs[c], env.rrad, color=trajcol[c],zorder=3)
                    ax[d,i].add_artist(circle)
                    circle2 = plt.Circle(env.rlocs[c], env.rrad, color='k',zorder=3,fill=False)
                    ax[d,i].add_artist(circle2)

                ax[d,i].axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
                ax[d,i].set_aspect('equal', adjustable='box')
                ax[d,i].set_xticks([])
                ax[d,i].set_yticks([])
                ax[d,i].set_title(mlegend[i])
        f.tight_layout()

        f.text(0.01,0.01,model)
        f.savefig('./Fig/6pa/6pa_traj{}_{}full_{}.png'.format(tnum,fulltraj, model[:7]))
        f.savefig('./Fig/6pa/6pa_traj{}_{}full_{}.svg'.format(tnum, fulltraj, model[:7]))


elif int(pltfig) == 4:
    # plot maps activity
    #genvar_path = '../6pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/Schema/6pa/'
    from backend.utils import get_binned_stat
    from backend.utils import get_default_hp
    from backend.maze import Maze

    mtype = ['train','train','train','opa','2npa','6npa']
    trajcol1 = ['tab:green','tab:gray','gold','tab:orange','tab:blue','tab:red']
    trajcol2 = ['blueviolet', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'forestgreen']
    trajcol3 = ['seagreen', 'darkgray', 'yellowgreen', 'coral', 'royalblue', 'brown']
    col = [trajcol1,trajcol1,trajcol1,trajcol1,trajcol2,trajcol3]

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

    for m,model in enumerate([allmodels[-1]]):
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

            saveload('save', './Data/wkm/wkm_value_policy_{}'.format(model), [value, policy])
            f,ax = plt.subplots(6,6,figsize=(10,10))
            for p in range(6):
                env = Maze(hp)
                env.make(mtype[p])
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
            f.text(0.01,0.01,model[:7])
            f.savefig('./Fig/6pa/6pa_maps_{}_b{}.png'.format(model[:7],N))
            f.savefig('./Fig/6pa/6pa_maps_{}_b{}.svg'.format(model[:7],N))
            plt.close()


elif int(pltfig) == 5:
    #genvar_path = '../6pa/Data/'
    genvar_path = 'D:/Ganesh_PhD/s4o/wkm/2dist/'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 6})

    from backend.utils import get_default_hp

    hp = get_default_hp(task='6pa', platform='laptop')
    z=1.96
    #for m,model in enumerate([allmodels[-1]]):
    #model_data = glob.glob(genvar_path+'vars_*{}*'.format(model))
    model_data = glob.glob(genvar_path + 'vars_*2dist*_*da_0.0001mlr*')
    N = len(model_data)
    randidx = np.random.choice(np.arange(N),N, replace=False)
    print(N)
    widx = ['Coord', 'Critic','Actor','Input','rec','Goal','Gate', 'bumprec']
    diffw = np.zeros([3, N, 5, 4])  # compare opa, npa, nw against trwN
    for d in randidx:

        [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load',model_data[d],1)

        [trw, opaw, npa2w,npa6w, nmw] = allw

        for j,nw in enumerate([5, 0, 6, 2,1]):
            for i,newweights in enumerate([opaw,npa2w,npa6w,nmw]):
                diffw[0, d, j,i] = np.linalg.norm(newweights[nw]-trw[nw],ord=1)
                diffw[1, d, j,i] = np.linalg.norm(newweights[nw]-trw[nw],ord=2)
                diffw[2, d, j, i] = np.mean((newweights[nw] - trw[nw])**2)

    saveload('save', './Data/wkm/wkm_dW_6npa_etda_{}b'.format(N), [diffw])

    [diffw] = saveload('load', './Data/wkm/wkm_dW_6npa_etda_281b.pickle', 1)

    ttest_ind(diffw[2,:,4,2],diffw[2,:,4,3]) # critic
    f_oneway(diffw[2,:,1,0],diffw[2,:,1,1],diffw[2,:,1,2],diffw[2,:,1,3])

    import itertools
    combi = list(itertools.combinations(np.arange(4),2))
    ttest = np.zeros([5,6,2])
    for w in range(5):
        for c,comb in enumerate(combi):
            n1 = comb[0]
            n2 = comb[1]
            ttest[w,c] = ttest_ind(diffw[2,:,w,n1],diffw[2,:,w,n2])


    wlegend = ['Goal','Coord', 'Gate', 'Actor', 'Critic']
    collegend = ['OPA','2NPA','6NPA','NM']

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FormatStrFormatter
    f, ax = plt.subplots(1, 5, figsize=(8, 1.5))
    for i in range(5):
        df = pd.DataFrame([np.mean(diffw[2],axis=0)[i]],columns=collegend)
        ds = pd.DataFrame([(z*np.std(diffw[2],axis=0)/N**0.5)[i]],columns=collegend)
        ax[i].yaxis.major.formatter.set_powerlimits((0, 0))
        #ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if i == 1:
            axb = ax[i]
            divider = make_axes_locatable(axb)
            axt = divider.new_vertical(size="100%", pad=0.1)
            f.add_axes(axt)

            axb.spines['top'].set_visible(False)
            axt.tick_params(bottom=False, labelbottom=False)
            axt.spines['bottom'].set_visible(False)
            axb.set_ylim(0, 0.0005)
            axt.set_ylim(0.035, 0.055)
            axt.set_title(wlegend[i])
            axt.yaxis.major.formatter.set_powerlimits((0, 0))
            axb.yaxis.major.formatter.set_powerlimits((0, 0))

            df.plot.bar(rot=0, ax=axb, yerr=ds, legend=False)
            df.plot.bar(rot=0, ax=axt, yerr=ds, legend=False)
            axb.set_xticks([])
            #axt.legend(collegend, loc='best', fontsize=6, frameon=False)

        else:
            df.plot.bar(rot=0, ax=ax[i], yerr=ds, legend=False)
            ax[i].set_title(wlegend[i])
            ax[i].set_xticks([])
            ax[0].set_ylabel('Squared Weight Change')

            ylim = ax[i].get_ylim()[1]
            ax[i].set_ylim((0,ylim*1.25))
        # allp = ax[i].patches
        # for c,comb in enumerate(combi):
        #     n1 = comb[0]
        #     n2 = comb[1]
        #     pval = ttest[i,c,1]
        #
        #     x1 = (allp[n1].get_x() + allp[n1].get_width())/2
        #     x2 = (allp[n2].get_x() + allp[n2].get_width())/2
        #     mid = (x1-x2)/2
        #     lineh = ylim*(1+0.025*c)
        #     texth = lineh+ylim/100
        #     # draw line
        #     ax[i].hlines(y=lineh, xmin=x1,xmax=x2, color='k',linewidth=1)
        #     print(x1, x2, mid)
        #
        #     if pval < 0.0001:
        #         ax[i].text(mid,texth, '****', ha='center', fontsize=4)
        #     if pval < 0.001:
        #         ax[i].text(mid,texth, '***', ha='center', fontsize=4)
        #     elif pval < 0.01:
        #         ax[i].text(mid,texth, '**', ha='center', fontsize=4)
        #     elif pval < 0.05:
        #         ax[i].text(mid,texth, '*', ha='center', fontsize=4)
        #     else:
        #         ax[i].text(mid,texth, 'n.s.', ha='center', fontsize=4)

    box = ax[-1].get_position()
    ax[-1].set_position([box.x0, box.y0, box.width, box.height])
    ax[-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    f.suptitle('Average Squared Weight change after 1 session', fontsize=8)
    f.tight_layout()

    fone = f_oneway(np.reshape(diffw[2,:,:,0],-1),np.reshape(diffw[2,:,:,1],-1),np.reshape(diffw[2, :,:,2],-1),np.reshape(diffw[2, :,:,3],-1))
    print(fone)

    f.savefig('./Fig/wkm/wkm_dw_4_2dist_da_b{}.png'.format(N))
    f.savefig('./Fig/wkm/wkm_dw_4_2dist_da_b{}.svg'.format(N))

elif int(pltfig) == 6:
    genvar_path = 'D:/Ganesh_PhD/s4o/wkm/'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    wtype = 'da'

    model_data = glob.glob(genvar_path + 'vars_*2dist*_{}_0.0001mlr*'.format(wtype))
    N = len(model_data)
    print(N)
    da = np.zeros([7,6,3000])
    gate = np.zeros([7,6,3000])

    for d in range(N):
        [alldyn, allw, mvpath, lat, dgr, pi] = saveload('load', model_data[d], 1)

        sess = list(alldyn[2].keys())

        for i in range(7):
            ss = sess[i * 6:i * 6 + 6]
            for c in ss:
                cue = int(c[-1])-1
                if cue == 6:
                    cue = 0
                elif cue == 7:
                    cue = 5

                da[i,cue] += np.vstack(alldyn[2][c])[:, 1]
                gate[i,cue] += np.vstack(alldyn[3][c])[:, 0]

    da /= N
    gate /= N

    saveload('save','./Data/wkm/wkm_s4o_{}_gate_act'.format(wtype),[da,gate])

    # import matplotlib.gridspec as gridspec
    # phase = ['PS1','PS2','PS3','OPA','2NPA','6NPA']
    # tcue = np.arange(1,7)
    # ncue = [7,2,3,4,5,8]
    # mcue = np.arange(11,17)
    # cues = [tcue, tcue, tcue, tcue, ncue, mcue]
    #
    # fig = plt.figure(figsize=(4, 4))
    # outer = gridspec.GridSpec(6, 6, wspace=0.3, hspace=0.5)
    # for i in range(6):
    #     for c in range(6):
    #         inner = gridspec.GridSpecFromSubplotSpec(2, 1,
    #                                                  subplot_spec=outer[i,c], wspace=0.1, hspace=0.1)
    #
    #         axt = plt.Subplot(fig, inner[0])
    #         axt.plot(da[i,c, :1000], color='darkviolet', zorder=2)
    #         axt.set_xticks([])
    #         axt.set_yticks([])
    #         axt.set_title('C{}'.format(cues[i][c]),y=0.8)
    #         # if c == 0:
    #         #     outer[i,c].set_ylabel(phase[i])
    #         axt.axvline(3*1000/20,color='r',linestyle='--')
    #         fig.add_subplot(axt)
    #
    #         axb = plt.Subplot(fig, inner[1])
    #         axb.plot(gate[i,c,:1000], color='darkgreen', zorder=2)
    #         axb.set_xticks([])
    #         axb.set_yticks([])
    #         axb.axvline(3 * 1000 / 20, color='r', linestyle='--')
    #         fig.add_subplot(axb)
    #
    # savefig('./Fig/wkm/wkm_{}_gate_b{}.png'.format(wtype, N), fig)

elif int(pltfig) == 7:
    genvar_path = './Data/wkm/tackm_2dist.pickle'
    [trackm] = saveload('load', genvar_path, 1)
    mpl.rcParams.update({'font.size': 8})
    f = plt.figure(figsize=(4,2))
    im = plt.imshow(np.vstack(trackm).T, aspect='auto')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.title('Cue 4 and 2 distractors')
    plt.ylabel('Bump neuron')
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(0,1001,250),np.arange(0,1001,250)*20//1000)
    plt.tight_layout()
    #plt.show()
    plt.savefig('./Fig/wkm/mem_2dist_.png')
    plt.savefig('./Fig/wkm/mem_2dist_.svg')

elif int(pltfig) == 8:
    wtype = 'da'
    [da,gate] = saveload('load','./Data/wkm/wkm_{}_gate_act.pickle'.format(wtype),1)
    dalim = [np.max(da), np.min(da)]
    gatelim = [np.max(gate), np.min(gate)]
    ptype = 'mean'

    import matplotlib.gridspec as gridspec
    phase = ['PS1','PS2','PS3','OPA','2NPA','6NPA']
    tcue = np.arange(1,7)
    ncue = [7,2,3,4,5,8]
    mcue = np.arange(11,17)
    cues = [tcue, tcue, tcue, tcue, ncue, mcue]

    time = 30*1000//20
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = False
    mpl.rcParams.update({'font.size': 8})

    fig = plt.figure(figsize=(6, 4))
    outer = list(gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.5))
    for i in range(6):
        for c in range(1):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                     subplot_spec=outer[i], wspace=0.1, hspace=0.1)

            axt = plt.Subplot(fig, inner[0])
            mpl.rcParams['axes.spines.bottom'] = True

            if ptype == 'mean':
                axt.plot(np.mean(da[i, :, :time],axis=0), color='darkviolet', zorder=2)
                axt.set_title('{}'.format(phase[i]), y=0.8)
            else:
                axt.plot(da[i,c, :time], color='darkviolet', zorder=2)
                axt.set_title('{} C{}'.format(phase[i], cues[i][c]), y=0.8)
            axt.set_xticks([])
            axt.set_ylim(bottom=dalim[1], top=dalim[0])

            if i == 0 or i == 3:
                axt.set_ylabel('DA')
            else:
                axt.set_yticks([])

            #axt.axvline(3*1000/20,color='r',linestyle='--')
            #axt.hlines(y=dalim[1]*1.1, xmin=1*1000/20, xmax=3*1000/20, color='k')
            #axt.text(y=dalim[1], x=1*1000/20, s='Cue')

            axt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            fig.add_subplot(axt)

            axb = plt.Subplot(fig, inner[1])
            mpl.rcParams['axes.spines.bottom'] = False
            if ptype == 'mean':
                axb.plot(np.mean(gate[i, :, :time],axis=0), color='darkgreen', zorder=2)
            else:
                axb.plot(gate[i,c,:time], color='darkgreen', zorder=2)

            axb.set_xticks([])
            axb.set_ylim(bottom=gatelim[1], top=gatelim[0])
            if i == 0 or i == 3:
                axb.set_ylabel('P(Update WM)')
            else:
                axb.set_yticks([])

            if i >2:
                axb.set_xlabel('Time (s)')
                axb.set_xticks(labels=np.linspace(0,20,5,dtype=int),ticks=np.linspace(0,time,5))
            else:
                axb.set_xticks([])

            #axb.axvline(3 * 1000 / 20, color='r', linestyle='--')
            axb.hlines(y=gatelim[0]*0.95, xmin=1*1000/20, xmax=3*1000/20, color='r', linewidth=5)
            randst = np.random.uniform(low=0,high=1,size=1)
            axb.hlines(y=gatelim[0] * 0.95, xmin=6 * 1000 / 20, xmax=time, color='dodgerblue', linewidth=3,linestyle=(0, (1, 5)))
            #axb.text(y=gatelim[0], x=1*1000/20, s='Cue')
            fig.add_subplot(axb)

    savefig('./Fig/wkm/da_gate_{}_act'.format(ptype),fig)