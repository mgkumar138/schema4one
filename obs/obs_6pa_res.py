import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend.maze import Maze
from backend.utils import get_default_hp, save_rdyn, saveload, plot_single_map
import multiprocessing as mp
from functools import partial

def multiplepa_script(hp, pool):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totlat = np.zeros([btstp, (hp['trsess'] + hp['evsess'] * 3)])
    totdgr = np.zeros([btstp, 6])
    totpi = np.zeros_like(totdgr)

    x = pool.map(partial(setup_hebagent_multiplepa_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath, allw, alldyn = x[b]

    f=plt.figure(figsize=(15, 8))
    f.text(0.01, 0.01, exptname, fontsize=10)

    ax = plt.subplot2grid((7, 7), (0, 0), colspan=3, rowspan=1)
    ax.set_title('Latency')
    ax.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    ax1 = plt.subplot2grid((7, 7), (0, 4), colspan=3, rowspan=1)
    ax1.bar(np.arange(6),np.mean(totdgr,axis=0))
    ax1.axhline(100/6,color='r',linestyle='--')
    ax1.set_title(np.round(np.mean(totpi,axis=0),1))

    sess = list(alldyn[1].keys())
    env = Maze(hp)
    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i,m in enumerate(['train','train','train','opa', '2npa', '6npa']):
        ss = sess[i * 6:i * 6 + 6]
        plt.subplot(7,7, i*7+8)
        plt.ylabel('{}'.format(m))
        env.make(m)
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
        plt.xticks([])
        plt.yticks([])

        for c in ss:
            cue = int(c[-1])-1
            if cue == 6:
                cue = 0
            elif cue == 7:
                cue = 5
            plt.subplot(7, 7, i * 7 + 9 + cue)
            plt.ylabel('C{}'.format(c[-1]))
            plot_single_map(np.vstack(alldyn[1][c]),np.vstack(alldyn[2][c]), k[cue],hp)
    plt.show()

    print(exptname)

    if hp['savefig']:
        f.savefig('./Fig/fig_{:.2g}o_{:.2g}n_{:.2g}m_{}.png'.format(
            np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4],np.mean(totdgr,axis=0)[5], exptname))

    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}o_{:.2g}n_{:.2g}m_{}.png'.format(
            np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4],np.mean(totdgr,axis=0)[5], exptname), [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, mvpath, allw, alldyn


def setup_hebagent_multiplepa_expt(hp,b):
    from backend.model import ResACAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)

    trsess = hp['trsess']
    evsess = 2

    # Create nonrewarded probe trial index
    scl = trsess / 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2, int(9 * scl), int(16 * scl)]  # sessions that are non-rewarded probe trials

    # store performance
    lat = np.zeros(trsess + evsess * 3)
    dgr = np.zeros(6)
    pi = np.zeros_like(dgr)

    # Start experiment
    alldyn = [{}, {}, {}, {}]
    mvpath = np.zeros([6, 6, env.normax, 2])

    start = dt.time()
    agent = ResACAgent(hp=hp)

    # Start Training
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_hebagent_multiplepa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_hebagent_multiplepa_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[2])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_hebagent_multiplepa_expt(b,'2npa', env, hp, agent, alldyn, evsess, trw, noreward=[2])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_hebagent_multiplepa_expt(b, '6npa', env, hp, agent, alldyn, evsess, trw, noreward=[2])

    allw = [trw, opaw, npaw, nmw]

    if hp['savevar']:
        saveload('save', './Data/vars_{:.2g}o_{:.2g}n_{}_{}'.format(dgr[3], dgr[4], exptname, dt.time()),
                 [alldyn, allw, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, allw, alldyn


def run_hebagent_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    if mtype=='train':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    # if mtype=='nm':
    #     agent.pc.flip_pcs()

    if useweight:
        agent.set_weights(useweight)

    mdlw = agent.get_weights()
    if hp['platform'] == 'laptop' or b == 0:
        print('Coord max {:.3g}, min {:.3g}'.format(np.max(mdlw[0]), np.min(mdlw[0])))
        print('Goal max {:.3g}, min {:.3g}'.format(np.max(mdlw[5]), np.min(mdlw[5])))
        print('Critic max {:.3g}, min {:.3g}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
        print('Actor max {:.3g}, min {:.3g}'.format(np.max(mdlw[2]), np.min(mdlw[2])))

    for t in range(sessions * len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()
        trackg = []
        if t%len(env.rlocs)==0:
            sesslat = []

        while not done:

            # Plasticity switched off during non-rewarded probe trials
            if t in env.nort:
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
                # save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, np.concatenate([value, tderr],axis=1))
                save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, goal)
            trackg.append(goal)

            if done:
                env.render()
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            sesslat.append(np.nan)
            if mtype == 'train':
                sid = np.argmax(np.array(noreward) == (t // 6) + 1)
                mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
            else:
                mvpath[env.idx] = env.tracks[:env.normax]

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
            mdlw = agent.get_weights()
            print('T {} | C {} | S {} | D {:4.3f} | st {} | g{} | as {} | Dgr {}'.format(
                t, np.argmax(env.cue)+1, env.i // (1000 // env.tstep), ds4r, env.startpos[0], np.round(goal,3),
                np.round(np.mean(agent.actor.avgspeed),3), np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('############## {} Session {}/{}, Avg Steps {:5.1f}, ##############'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1]))
                mdlw = agent.get_weights()
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
        sesspi = np.sum(np.array(dgr) > 100/3)
        dgr = np.mean(dgr)

    mdlw = agent.get_weights()

    print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


if __name__ == '__main__':

    hp = get_default_hp(task='obs',platform='laptop')

    hp['btstp'] = 24
    hp['savefig'] = True
    hp['savegenvar'] = True
    hp['savevar'] = False

    hp['hidtype'] = 'rnn'

    hp['alr'] = 0.000005  # reduce for obs
    hp['clr'] = 0.0001  # 0.0001
    hp['taug'] = 10000  # 10000

    ''' env param'''
    hp['Rval'] = 1
    hp['obs'] = True
    hp['render'] = False  # visualise movement trial by trial

    allcb = [0,0.4,1]
    pool = mp.Pool(processes=hp['cpucount'])

    for cb in allcb:
        hp['contbeta'] = cb

        hp['exptname'] = 'obs_res_{}cb_{}glr_{}sl_{}ach_{}clr_{}tg_{}alr_{}N_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['glr'], hp['stochlearn'],hp['ach'], hp['clr'], hp['taug'],hp['alr'], hp['nrnn'],
            hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp, pool)

    pool.close()
    pool.join()