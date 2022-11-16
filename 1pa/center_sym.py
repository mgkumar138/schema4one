import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from backend.utils import save_rdyn, saveload, get_default_hp, plot_1pa_maps_center
import time as dt
import pandas as pd
from backend.maze import Static_Maze
import multiprocessing as mp
from functools import partial
from backend.model import SymACAgent


def static_pa_script(hp, pool):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    tt = hp['trsess'] * 6  # Number of session X number of trials per session

    # store performance
    totlat = np.zeros([btstp, (tt)])
    totdgr = np.zeros([btstp, 3])
    totpath = np.zeros([btstp, 3, 6, int(hp['probetime'] * 1000 / hp['tstep']), 2])

    x = pool.map(partial(main_single_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpath[b], mdlw, alldyn = x[b]

    totlat = totlat * hp['tstep'] / 1000

    dgrperf = np.round(np.mean(totdgr, axis=0)[-1], 1)
    latperf = np.mean(np.mean(totlat, axis=0)[-16:-6])
    plt.figure(figsize=(12, 6))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(341)
    plt.title('Latency per Trial, {:.1f}'.format(latperf))
    plt.errorbar(x=np.arange(totlat.shape[1]), y=np.mean(totlat, axis=0), yerr=np.std(totlat, axis=0))
    # plt.plot(np.mean(totlat, axis=0), 'k', linewidth=3)

    plt.subplot(345)
    df = pd.DataFrame(np.mean(totdgr, axis=0), index=['1', '2', '3'])
    ds = pd.DataFrame(np.std(totdgr, axis=0), index=['1', '2', '3'])
    df.plot.bar(rot=0, ax=plt.gca(), yerr=ds, legend=False)
    plt.axhline(y=np.mean(totdgr, axis=0)[0], color='r', linestyle='--')

    # create environment
    env = Static_Maze(hp)
    env.make('square', rloc=24)

    hitr = []
    stl = []
    mvpath = totpath[0]
    for i in range(3):
        for p in range(6):
            hitr.append(np.argmax(np.linalg.norm(mvpath[i, p], axis=1) < 0.05) - 1)
            stl.append((mvpath[i, p, 0] != np.array([0, -0.8])).any())
    hitr = np.reshape(hitr, (3, 6))
    stl = np.reshape(stl, (3, 6))
    pidx = []
    comprate = []
    for i in range(3):
        pidx.append(np.argmax(stl[i] * hitr[i] > 0))
        comprate.append(np.sum(stl[i] * hitr[i] > 0))
    comprate = 100 * np.array(comprate) / 6

    # plot traj
    for i in range(3):
        plt.subplot(3, 4, i + 2)
        if hp['obs']:
            for j in range(len(env.obstacles)):
                circle = plt.Circle(env.obstacles[j], env.obssz, color='k')
                plt.gcf().gca().add_artist(circle)
        k = mvpath[i, pidx[i]]
        plt.title('{}'.format(i))
        plt.plot(np.array(k)[:, 0], np.array(k)[:, 1], alpha=1)
        circle = plt.Circle(env.rloc, env.rrad, color='r')
        plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])

    # plot rho
    sess = np.reshape(list(alldyn[1].keys()), (3, 6))
    for i in range(3):
        plt.subplot(3, 4, i + 6)
        rho = np.array(alldyn[1][sess[i, pidx[i]]])
        plt.imshow(rho.T, aspect='auto')  # [:hitr[i,pidx[i]]]
        # plt.plot(np.arange(rho[:hitr[i,pidx[i]]].shape[0]), np.argmax(rho[:hitr[i,pidx[i]]], axis=1), color='r', alpha=0.25,
        #          linewidth=1)
        plt.colorbar()

    # plot weights
    if hp['hidtype'] == 'foster':
        agent = SymACAgent(hp=hp)
        xx, yy = np.meshgrid(np.arange(hp['npc']), np.arange(hp['npc']))
        for i in range(3):
            plt.subplot(3, 4, i + 10)
            actorw = np.array(mdlw[i][2][:hp['npc'] ** 2])
            actorpol = np.matmul(agent.actor.aj, actorw.T)
            criticw = np.array(mdlw[i][1][:, 0][:hp['npc'] ** 2])
            plt.imshow(criticw.reshape(hp['npc'], hp['npc']))
            plt.title('C {}, A {}'.format(np.round([np.max(criticw), np.min(criticw)], 2),
                                          np.round([np.max(actorw), np.min(actorw)], 3)))
            plt.colorbar()
            plt.quiver(xx.reshape(-1), yy.reshape(-1), actorpol[0], actorpol[1], color='w')
            plt.xticks([])
            plt.yticks([])

    else:
        plot_1pa_maps_center(alldyn, mvpath, hp, mdlw)

    plt.tight_layout()

    if hp['savefig']:
        plt.savefig('./Fig/fig_{:.2g}l_{:.2g}d_{}.png'.format(latperf, dgrperf, exptname))
    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}l_{:.2g}d_{}_b{}_{}'.format(latperf, dgrperf, exptname, btstp, dt.monotonic()),
                 [totlat, totdgr, totpath])
    print(exptname)

    return totlat, totdgr, mvpath, mdlw, alldyn


def run_single_expt(b, mtype, env, hp, agent, alldyn, trials, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(trials)
    dgr = []
    mvpath = np.zeros((3, 6, env.normax, 2))
    pweights = []

    env.make(mtype=mtype, rloc=24, noreward=noreward)

    if useweight:
        agent.set_weights(useweight)

    for t in range(trials):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()

        while not done:

            # Plasticity switched off during non-rewarded probe trials
            if t in env.nort:
                plasticity = False
            else:
                plasticity = True

            cpc, cue, rfr = agent.see(state=state, cue=cue, startbox=env.startbox)

            value, xy, goal = agent.estimate_value_position_goal(cpc=cpc,cue=cue,rfr=rfr)

            tderr, tdxy = agent.learn(reward=reward,self_motion=env.dtxy, cpc=cpc, rfr=rfr, plasticity=plasticity)

            action, rho = agent.get_action(rfr=rfr, xy=xy, goal=goal)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            agent.learn_cue_location(cue=cue, xy=xy, goal=goal, reward=reward, done=done,plasticity=plasticity)

            # save lsm & actor dynamics for analysis
            if t in env.nort:
                # save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], mtype, t, env.startpos, 0, rho)
                save_rdyn(alldyn[2], mtype, t, env.startpos, 0, np.concatenate([value, tderr],axis=1))
                save_rdyn(alldyn[3], mtype, t, env.startpos, 0, goal)

            if done:
                env.render()
                break

        if (t == np.array([11, 35, 59])).any():
            pweights.append(agent.get_weights())

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            lat[t] = np.nan
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, t % 6] = env.tracks[:env.normax]
            #print(np.max(trackval), np.max(np.vstack(tracka)))
        else:
            lat[t] = env.i

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | S {} | D {:4.3f} | {}as | g{} | Dgr {}'.format(
                t, env.i // (1000 // env.tstep), ds4r,
                np.round(np.mean(agent.actor.avgspeed), 3), np.round(goal,3), np.round(dgr)))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        dgr = np.mean(dgr)

    mdlw = agent.get_weights()
    if hp['platform'] == 'laptop' or b == 0:
        print('Critic max {:.3g}, min {:.3g}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
        print('Actor max {:.3g}, min {:.3g}'.format(np.max(mdlw[2]), np.min(mdlw[2])))
        print('Coord max {:.3g}, min {:.3g}'.format(np.max(mdlw[0]), np.min(mdlw[0])))
    return lat, dgr, pweights, mvpath


def main_single_expt(hp, b):
    from backend.model import SymACAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']

    # create environment
    env = Static_Maze(hp)

    tt = hp['trsess'] * 6  # Number of session X number of trials per session
    nonr = [2, 6, 10]
    alldyn = [{}, {}, {}, {}]

    start = dt.time()
    agent = SymACAgent(hp=hp)

    # Start Training
    lat, dgr, pweights, mvpath = run_single_expt(b, 'square', env, hp, agent, alldyn, tt, noreward=nonr)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.monotonic()),
                [alldyn, pweights, mvpath, lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, mvpath, pweights, alldyn


if __name__ == '__main__':
    hp = get_default_hp(task='center', platform='laptop')

    hp['btstp'] = 24
    hp['savefig'] = True
    hp['savegenvar'] = True
    hp['savevar'] = False

    hp['alr'] = 0.00001  # 0.0003
    hp['clr'] = 0.0001  # 0.0003
    hp['taug'] = 10000

    hp['usenmc'] = False  # confi, neural

    hp['Rval'] = 1
    hp['obs'] = True
    hp['obstype'] = 'cave'
    hp['render'] = False  # visualise movement trial by trial

    allcb = [0,0.3,1]

    pool = mp.Pool(processes=hp['cpucount'])

    for cb in allcb:
        hp['contbeta'] = cb

        hp['exptname'] = 'center_sym_{}cb_{}mb_{}oe_{}gd{}_{}clr_{}tg_{}alr_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['mcbeta'], hp['omite'], hp['gdecay'],hp['gdist'], hp['clr'], hp['taug'], hp['alr'],
            hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totlat, totdgr, mvpath, mdlw, alldyn = static_pa_script(hp, pool)

    pool.close()
    pool.join()
