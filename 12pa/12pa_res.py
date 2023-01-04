import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend.maze import MultiplePA_Maze
from backend.utils import get_default_hp, save_rdyn, saveload
import multiprocessing as mp
from functools import partial

def npapa_script(hp, pool):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totdgr = np.zeros([btstp, 4])
    totpi = np.zeros_like(totdgr)
    totpath = np.zeros([btstp, 12, 1+hp['probetime'] * (1000 // hp['tstep']), 2])

    x = pool.map(partial(sym_12pa_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totdgr[b], totpi[b], totpath[b], allw = x[b]
        #totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn, agent = control_multiplepa_expt(hp,b)

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)

    plt.subplot(231)
    plt.bar(np.arange(4),np.mean(totdgr,axis=0))
    plt.axhline(100/6,color='r',linestyle='--')
    plt.title(np.round(np.mean(totdgr,axis=0),1))
    plt.ylabel('visit ratio')

    plt.subplot(232)
    plt.bar(np.arange(4),np.mean(totpi,axis=0))
    plt.title(np.round(np.mean(totpi,axis=0),1))
    plt.ylabel('# PA learned')

    import matplotlib.cm as cm
    col = cm.rainbow(np.linspace(0, 1, 12))
    plt.subplot(2, 3, 3)
    k = totpath[0]
    for pt in range(12):
        plt.plot(np.array(k[pt])[:-1, 0], np.array(k[pt])[:-1, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=1)
        plt.scatter(np.array(k[pt])[0, 0], np.array(k[pt])[0, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=1, marker='s',s=50)
        plt.scatter(np.array(k[pt])[-2, 0], np.array(k[pt])[-2, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=2, marker='x', s=50)

        circle = plt.Circle(k[pt][-1], 0.03, color=col[pt], zorder=9)
        plt.gcf().gca().add_artist(circle)
        circle2 = plt.Circle(k[pt][-1], 0.03, color='k', fill=False, zorder=10)
        plt.gcf().gca().add_artist(circle2)

    plt.axis((-0.8, 0.8, -0.8, 0.8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.tight_layout()

    print(exptname)

    plt.tight_layout()

    if hp['savefig']:
        plt.savefig('./Fig/fig_{:.2g}n_{}.png'.format(np.mean(totpi,axis=0)[3],exptname))
    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}n_{}'.format(np.mean(totpi,axis=0)[3],exptname), [totdgr, totpi, totpath])

    return totdgr, totpi, totpath, allw


def sym_12pa_expt(hp,b):
    from backend.model import ResACAgent
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = MultiplePA_Maze(hp)

    trsess = hp['trsess']
    evsess = 2

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # store performance
    dgr = np.zeros(4)
    pi = np.zeros_like(dgr)

    # Start experiment
    alldyn = [{}, {}, {}, {}]

    start = dt.time()
    agent = ResACAgent(hp=hp)

    # Start Training
    _, trw, dgr[:3], pi[:3] = run_sym_12pa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    mvpath,  npa1w, dgr[3], pi[3] = run_sym_12pa_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', 'vars_{}_{}'.format(exptname, dt.time()), [mvpath, dgr, pi, []])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath, [trw[0], npa1w[0]]


def run_sym_12pa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype!='train':
        mvpath = np.zeros((12,env.normax+1,2))
    else:
        mvpath = None

    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.set_weights(useweight)

    for t in range(sessions*env.totr):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()

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

            if t in env.nort:
                #save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)  # rho
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho) # rho
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, np.concatenate([value, tderr],axis=1))  # rho
                save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, goal)

            if done:
                env.render()
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            if mtype != 'train':
                mvpath[env.idx] = np.concatenate([np.array(env.tracks)[:env.normax], env.rloc[None,:]],axis=0)
            dgr.append(env.dgr)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | Dgr {} | gl {} | goal {} | '.format(
                t, np.argmax(env.cue)+1, env.i // (1000 // env.tstep),np.round(env.dgr,1), env.rloc, np.round(goal,2)))

            # Session information
            if (t + 1) % env.totr == 0:
                print('################## {} Session {}/{}, PI {} ################'.format(
                    mtype, (t + 1) // env.totr, sessions, env.sessr))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > 100/3
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.sum(np.array(dgr) > 100/6)
        dgr = np.mean(dgr)

    mdlw = agent.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return mvpath, mdlw, dgr, sesspi


if __name__ == '__main__':

    hp = get_default_hp(task='6pa', platform='laptop')

    hp['probetime'] = 5
    hp['time'] = 1

    hp['btstp'] = 1
    hp['savefig'] = False
    hp['savegenvar'] = False
    hp['savevar'] = False

    hp['alr'] = 0.00005  # 0.0005
    hp['clr'] = 0.0002  # 0.0002
    hp['taug'] = 3000

    hp['stochlearn'] = True  # if false, will use LMS rule

    hp['usenmc'] = True  # confi, neural

    ''' env param'''
    hp['Rval'] = 5
    hp['render'] = False  # visualise movement trial by trial

    allN = 2**np.arange(7,12)
    #allN = [1024]
    pool = mp.Pool(processes=hp['cpucount'])

    for N in allN:
        hp['contbeta'] = 1
        hp['nrnn'] = N
        hp['glr'] = 7.5e-6  # if N>2000 and stochlearn=True, reduce glr to 5e-6

        hp['exptname'] = '12pa_res_{}cb_{}sl_{}ach_{}glr_{}clr_{}tg_{}alr_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['stochlearn'], hp['ach'], hp['glr'], hp['clr'], hp['taug'],
            hp['alr'], hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totdgr, totpi, totpath, allw = npapa_script(hp, pool)

    pool.close()
    pool.join()