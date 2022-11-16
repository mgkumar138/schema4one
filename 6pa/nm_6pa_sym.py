import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend.maze import Maze
from backend.utils import get_default_hp, save_rdyn, saveload
import multiprocessing as mp
from functools import partial

def multiplepa_script(hp, pool):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totlat = np.zeros([btstp, (hp['trsess']*2 + hp['evsess'] * 4)])
    totdgr = np.zeros([btstp, 10])
    totpi = np.zeros_like(totdgr)

    x = pool.map(partial(setup_hebagent_multiplepa_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath, allw, alldyn = x[b]

    f = plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(341)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    plt.subplot(345)
    plt.bar(np.arange(10),np.mean(totdgr,axis=0))
    plt.axhline(100/6,color='r',linestyle='--')
    plt.title(np.round(np.mean(totpi,axis=0),1))

    env = Maze(hp)
    col = ['b', 'g', 'r', 'y', 'm', 'k']
    mtypes = ['train','train','train','opa','2npa','6npa','nm','nm','nm', '6nm']
    pltidx = [2,3,4,6,7,8,9,10,11,12]
    for i in range(10):
        plt.subplot(3, 4, pltidx[i])
        plt.title('{}'.format(mtypes[i]))
        env.make(mtypes[i])
        k = mvpath[i]
        for pt in range(len(mvpath[2])):
            plt.plot(np.array(k[pt])[:, 0], np.array(k[pt])[:, 1], col[pt], alpha=0.5, zorder=2)
            circle = plt.Circle(env.rlocs[pt], env.rrad, color=col[pt], zorder=3)
            plt.gcf().gca().add_artist(circle)
        if hp['obs']:
            for ld in env.obstacles:
                circle = plt.Circle(ld, env.obssz, color='k', zorder=1)
                plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
        plt.gca().set_aspect('equal', adjustable='box')

    print(exptname)

    plt.tight_layout()

    if hp['savefig']:
        f.savefig('./Fig/fig_{:.2g}o_{:.2g}n_{}.png'.format(np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4], exptname))

    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}o_{:.2g}n_{}'.format(
            np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4], exptname), [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, mvpath, allw, alldyn


def setup_hebagent_multiplepa_expt(hp,b):
    from backend.model import SymACAgent
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)

    trsess = hp['trsess']
    evsess = int(trsess*.1)

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # store performance
    lat = []
    dgr = []
    pi = []
    allw = []
    mvpath = []

    # Start experiment
    alldyn = [{}, {}, {}, {}]

    start = dt.time()
    agent = SymACAgent(hp=hp)
    useweight = None

    # Start Training
    mtypes = ['train','opa','2npa','6npa','nm','6nm']
    for mt in range(6):
        mtype = mtypes[mt]
        if mtype == 'train' or mtype == 'nm':
            sess = trsess
            nonprobe = nonrp
        else:
            sess = evsess
            nonprobe = [nonrp[0]]

        latency, path, weight, visitratio, perfidx = run_hebagent_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sess, useweight=useweight, noreward=nonprobe)

        if mtype == 'train' or mtype == 'nm':
            useweight = weight

        lat.append(latency)
        mvpath.append(path)
        allw.append(weight)
        dgr.append(visitratio)
        pi.append(perfidx)

    lat = np.concatenate(lat)
    dgr = np.concatenate(dgr)
    pi = np.concatenate(pi)
    mvpath = np.concatenate(mvpath,axis=0)

    if hp['savevar']:
        saveload('save', './Data/vars_{:.2g}o_{:.2g}n_{}_{}'.format(dgr[3], dgr[4], exptname, dt.time()),
                 [alldyn, allw, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, allw, alldyn


def run_hebagent_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    if mtype=='train' or mtype == 'nm':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((1, 6,env.normax,2))
    dgr = []

    if useweight:
        agent.set_weights(useweight)
        agent.memory.keyvalue[6:] = 0

    env.make(mtype=mtype, nocue=nocue, noreward=noreward)
    if mtype == 'nm':
        agent.pc.flip_pcs()

    for t in range(sessions * len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()

        if t%len(env.rlocs)==0:
            sesslat = []

        while not done:

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort or t in env.noct:
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
            sesslat.append(np.nan)
            if mtype == 'train' or mtype == 'nm':
                sid = np.argmax(np.array(noreward) == (t // 6) + 1)
                mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
            else:
                mvpath[0, env.idx] = env.tracks[:env.normax]

            if mtype == '2npa':
                if (np.argmax(env.cue)+1 == np.array([7, 8])).any():
                    dgr.append(env.dgr)
            else:
                dgr.append(env.dgr)
        else:
            sesslat.append(env.i)

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] != 'server' or b == 0:
            # Trial information
            print('C{} | D {:4.3f} | S {} | Dgr {:.3g} | {} as | g {}'.format(
                    np.argmax(env.cue)+1, ds4r, env.i // (1000 // env.tstep), env.dgr,
                np.round(np.mean(agent.actor.avgspeed),3), np.round(goal,2)))

            # Session information
            if (t + 1) % 6 == 0:
                print('############## {} Session {}/{}, Avg Steps {:5.1f}, ##############'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1]))
                mdlw = agent.get_weights()
                print('Coord max {:.3g}, min {:.3g}'.format(np.max(mdlw[0]), np.min(mdlw[0])))
                print('Critic max {:.3g}, min {:.3g}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
                print('Actor max {:.3g}, min {:.3g}'.format(np.max(mdlw[2]), np.min(mdlw[2])))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > 100/3
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.array([np.sum(np.array(dgr) > 100/3)])
        dgr = np.array([np.mean(dgr)])

    mdlw = agent.get_weights()

    print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


if __name__ == '__main__':

    hp = get_default_hp(task='6pa', platform='server')

    hp['btstp'] = 4
    hp['savefig'] = True
    hp['savegenvar'] = True
    hp['savevar'] = False

    hp['alr'] = 0.00005  # 0.0005
    hp['clr'] = 0.0002  # 0.0002
    hp['taug'] = 3000

    hp['usenmc'] = False  # confi, neural

    ''' env param'''
    hp['Rval'] = 5
    hp['render'] = False  # visualise movement trial by trial

    allcb = [1]
    pool = mp.Pool(processes=hp['cpucount'])

    for cb in allcb:
        hp['contbeta'] = cb

        hp['exptname'] = '6pa_sym_{}cb_{}mb_{}oe_{}gd{}_{}clr_{}tg_{}alr_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['mcbeta'], hp['omite'], hp['gdecay'],hp['gdist'], hp['clr'], hp['taug'], hp['alr'],
            hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp, pool)

    pool.close()
    pool.join()