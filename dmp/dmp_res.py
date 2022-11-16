import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend.maze import Navex
import multiprocessing as mp
from functools import partial
from backend.model import ResACAgent
from backend.utils import saveload, save_rdyn, get_default_hp, plot_1pa_maps_dmp, get_savings


def singlepa_script(hp, pool):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    trsess = hp['trsess']  # number of training trials per cue, location
    epochs = hp['epochs'] # number of epochs to go through all cue-location combination

    # store performance
    totlat = np.zeros([btstp, epochs, trsess])
    totdgr = np.zeros([btstp, epochs])
    totpath = np.zeros([btstp,epochs, int(hp['probetime'] * (1000 // hp['tstep'])+1),2])

    x = pool.map(partial(sym_singleloc_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpath[b], pweights, alldyn = x[b]

    totlat[totlat == 0] = np.nan

    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 8})
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    ax1 = plt.subplot2grid((6, 9), (0, 0), colspan=4, rowspan=2)
    plt.ylabel('Latency (s)')
    totlat *= hp['tstep']/1000
    #plt.title('Latency, {} trials'.format(trsess))
    plt.errorbar(x=np.arange(epochs*trsess),y=np.mean(totlat,axis=0).reshape(-1),
                 yerr=np.std(totlat,axis=0).reshape(-1)/np.sqrt(btstp), marker='o', color='k', elinewidth=0.5, markersize=5)

    ax2 = plt.subplot2grid((6, 9),(0, 5), colspan=4, rowspan=2)
    plt.ylabel('Visit Ratio per Epoch')
    dgrmean = np.mean(totdgr, axis=0)
    dgrstd = np.std(totdgr, axis=0)
    plt.errorbar(x=np.arange(epochs), y=dgrmean, yerr=dgrstd/np.sqrt(btstp), color='k')
    md, _ = np.polyfit(x=np.arange(epochs), y=dgrmean, deg=1)
    plt.legend(['VrG {:.2g}'.format(md)],loc=2)
    ax3 = plt.twinx()
    savingsm, savingss = get_savings(totlat)
    ms,_ = np.polyfit(x=np.arange(epochs),y=savingsm,deg=1)
    ax3.errorbar(x=np.arange(epochs),y=savingsm, yerr=savingss/np.sqrt(btstp), color='g')
    ax3.set_ylabel('Savings')
    ax3.legend(['SavG {:.2g}'.format(ms)],loc=1)

    mvpath = totpath[0]
    #midx = np.linspace(0,epochs-1,3,dtype=int)
    for i in range(hp['epochs']):
        plt.subplot(6,9,i+19)

        rloc = mvpath[i, -1]
        plt.ylabel('PT{}, R{}'.format(i+1, rloc))
        plt.plot(mvpath[i,:-1,0],mvpath[i,:-1,1],'k')
        plt.axis([-0.8,0.8,-0.8,0.8])
        plt.gca().set_aspect('equal', adjustable='box')
        circle = plt.Circle(rloc, 0.03, color='r')
        plt.gcf().gca().add_artist(circle)
        plt.xticks([])
        plt.yticks([])

    env = Navex(hp)
    env.make()
    agent = ResACAgent(hp=hp)

    # plot weights
    if hp['hidtype'] == 'classic':
        xx, yy = np.meshgrid(np.arange(hp['npc']), np.arange(hp['npc']))
        for i in range(hp['epochs']):
            ax = plt.subplot(6, 9, i + 28)
            actorw = pweights[i][2][:hp['npc'] ** 2].numpy()
            actorpol = np.matmul(agent.actor.aj, actorw.T)
            criticw = pweights[i][1][:, 0][:hp['npc'] ** 2].numpy()
            im = plt.imshow(criticw.reshape(hp['npc'], hp['npc']))
            plt.title('C {}'.format(np.round([np.max(criticw), np.min(criticw)], 1)))
            plt.xlabel('A {}'.format(np.round([np.max(actorw), np.min(actorw)], 1)))
            plt.colorbar(im,fraction=0.046, pad=0.04,ax=ax)
            plt.quiver(xx.reshape(-1), yy.reshape(-1), actorpol[0], actorpol[1], color='w')
            plt.xticks([])
            plt.yticks([])
    else:
        plot_1pa_maps_dmp(alldyn, mvpath, hp, pweights, pltidx=[6, 9, 28])

    # plt.subplot(prow, 9, 4)
    # plt.imshow(pweights[0][-1],aspect='auto')
    # plt.colorbar()
    # plt.title('Mem Weights')

    # plot rho
    sess = list(alldyn[1].keys())
    labels = ['rho', 'goal']
    ridx = [1,3]
    for d in range(2):
        for i in range(hp['epochs']):
            plt.subplot(6, 9, i + 37 + 9*d)
            if i == 0:
                plt.ylabel(labels[d])
            dyn = np.array(alldyn[ridx[d]][sess[i]])
            if d == 1:
                plt.axhline(y=hp['omite'], color='r', linestyle='--')
                plt.plot(dyn)
            else:
                plt.imshow(dyn.T, aspect='auto')  # [:hitr[i,pidx[i]]]
                plt.colorbar()

    plt.show()

    if hp['savefig']:
        plt.savefig('./Fig/fig_{:.2g}s_{:.2g}d_{}.png'.format(ms,md, exptname))
    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}s_{:.2g}d_{}_b{}_{}'.format(ms, md, exptname, btstp, dt.time()),   [totlat, totdgr, totpath])

    return totlat, totdgr, mvpath, pweights, alldyn


def sym_singleloc_expt(hp,b):
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Navex(hp)

    trsess = hp['trsess']
    epochs = hp['epochs']

    # Create nonrewarded probe trial index

    # store performance
    lat = np.zeros([epochs, trsess])
    dgr = np.zeros([epochs])
    totpath = np.zeros([epochs, env.normax + 1, 2])
    alldyn = [{}, {}, {}, {}]
    pweights = []

    # Start experiment
    start = dt.time()
    agent = ResACAgent(hp=hp)
    mdlw = None

    # start training
    for e in range(epochs):

        env.make(noreward=[hp['trsess']])
        rlocs = env.rlocs
        if b == 0:
            print('All Rlocs in Epoch {}: {}'.format(e, rlocs))

        lat[e], dgr[e], mdlw, totpath[e] = run_sym_1rloc_expt(e, b, env, hp, agent, trsess, alldyn, useweight=mdlw)
        pweights.append(mdlw)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.monotonic()), [alldyn,pweights, totpath, lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, pweights, alldyn


def run_sym_1rloc_expt(e, b, env, hp, agent, sessions, alldyn, useweight=None, noreward=None):
    lat = np.zeros(sessions)
    dgr = []

    if useweight:
        agent.set_weights(useweight)

    for t in range(sessions*len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()
        trackg = []
        while not done:

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort or t in env.noct:
                plasticity = False
            else:
                plasticity = True

            cpc, cue, rfr = agent.see(state=state, cue=cue, startbox=env.startbox)

            value, xy, goal, mem = agent.estimate_value_position_goal_memory_td(cpc=cpc,cue=cue,rfr=rfr)

            tderr, tdxy = agent.learn(reward=reward, self_motion=env.dtxy, cpc=cpc, cue=cue, rfr=rfr, xy=xy, goal=goal,
                                      plasticity=plasticity)

            action, rho = agent.get_action(rfr=rfr, xy=xy, goal=goal)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # save lsm & actor dynamics for analysis
            if t in env.nort:
                #save_rdyn(alldyn[0], 'dmp', e, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], 'dmp', e, env.startpos, env.cue, rho)
                save_rdyn(alldyn[2], 'dmp', e, env.startpos, env.cue, np.concatenate([value, tderr],axis=1))
                save_rdyn(alldyn[3], 'dmp', e, env.startpos, env.cue, goal)
            trackg.append(goal)

            if done:
                env.render()
                break

        if env.probe:
            dgr = env.dgr
            mvpath = np.concatenate([np.array(env.tracks[:env.normax]),env.rloc[None,:]],axis=0)
        else:
            lat[t] = env.i

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('{} | D {:4.3f} | S {} | Dgr {} | {} as | g {} | xy{}'.format(
                    t, ds4r, env.i // (1000 // env.tstep), env.dgr,
                np.round(np.mean(agent.actor.avgspeed),3), np.round(goal,2),
                np.round(xy.numpy()[0],2)))

    mdlw = agent.get_weights()
    if hp['platform'] == 'laptop' or b == 0:
        print('Coord max {:.3g}, min {:.3g}'.format(np.max(mdlw[0]), np.min(mdlw[0])))
        print('Critic max {:.3g}, min {:.3g}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
        print('Actor max {:.3g}, min {:.3g}'.format(np.max(mdlw[2]), np.min(mdlw[2])))
        print('Goal max {:.3g}, min {:.3g}'.format(np.max(mdlw[5]), np.min(mdlw[5])))

    return lat, dgr, mdlw, mvpath


if __name__ == '__main__':

    hp = get_default_hp(task='dmp',platform='server')

    hp['btstp'] = 24
    hp['savefig'] = True
    hp['savegenvar'] = True
    hp['savevar'] = False

    ''' agent param '''
    hp['clr'] = 0.0002
    hp['taug'] = 3000
    hp['alr'] = 0.00005

    hp['usenmc'] = True  # confi, neural

    hp['stochlearn'] = True

    ''' env param'''
    hp['Rval'] = 5
    hp['render'] = False  # visualise movement trial by trial

    allcb = [1]

    pool = mp.Pool(processes=hp['cpucount'])

    for cb in allcb:
        hp['contbeta'] = cb

        hp['exptname'] = 'dmp_res_{}cb_{}glr_{}sl_{}ach_{}clr_{}tg_{}alr_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['glr'], hp['stochlearn'],hp['ach'], hp['clr'], hp['taug'],hp['alr'],
            hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totlat, totdgr, mvpath, pw, alldyn = singlepa_script(hp, pool)

    pool.close()
    pool.join()