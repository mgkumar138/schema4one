#%%
import sys
import os
sys.path.append(os.getcwd())

# add path schema4one
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend.maze import Maze
from backend.utils import get_default_hp, save_rdyn, saveload
import multiprocessing as mp
from functools import partial
import argparse

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

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(331)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    plt.subplot(332)
    plt.bar(np.arange(6),np.mean(totdgr,axis=0))
    plt.axhline(100/6,color='r',linestyle='--')
    plt.title(np.round(np.mean(totpi,axis=0),1))

    env = Maze(hp)
    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i,m in enumerate(['train','train','train','opa', '2npa', '6npa']):

        plt.subplot(3, 3, i+4)
        plt.title('{}'.format(m))
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

    print(exptname)

    plt.tight_layout()

    if hp['savefig']:
        plt.savefig('./Fig/fig_{:.2g}o_{:.2g}n_{:.2g}m_{}.png'.format(
            np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4],np.mean(totdgr,axis=0)[5], exptname))
    if hp['savegenvar']:
        saveload('save', './Data/genvars_{:.2g}o_{:.2g}n_{:.2g}m_{}.png'.format(
            np.mean(totdgr,axis=0)[3],np.mean(totdgr,axis=0)[4],np.mean(totdgr,axis=0)[5], exptname), [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, mvpath, allw, alldyn


def setup_hebagent_multiplepa_expt(hp,b):
    from backend.model_np import SymACAgent
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
    agent = SymACAgent(hp=hp)

    # Start Training
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_hebagent_multiplepa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_hebagent_multiplepa_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_hebagent_multiplepa_expt(b,'2npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_hebagent_multiplepa_expt(b, '6npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

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

    for t in range(sessions * len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.state_reset()

        if t%len(env.rlocs)==0:
            sesslat = []

        while not done:

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
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
            print('T {} | C {} | S {} | D {:4.3f} | st {} | g{} | as {} | Dgr {}'.format(
                t, np.argmax(env.cue)+1, env.i // (1000 // env.tstep), ds4r, env.startpos[0], np.round(goal,3),
                np.round(np.mean(agent.actor.avgspeed),3), np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))
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
    if hp['platform'] == 'laptop' or b == 0:
        print('Critic max {:.3f}, min {:.3f}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
        print('Actor max {:.3f}, min {:.3f}'.format(np.max(mdlw[2]), np.min(mdlw[2])))
        print('Coord max {:.3f}, min {:.3f}'.format(np.max(mdlw[0]), np.min(mdlw[0])))

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        if v.lower() in ("no", "false", "f", "n", "0"): return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='Run observation symmetry experiment')
    parser.add_argument('--cb', type=float, default=1.0, help='beta_control')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--usenmc', type=str2bool, default=True, help='use neural motor controller (bool)')
    args = parser.parse_args()

    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)

    # save data
    data_dir = '/n/netscratch/pehlevan_lab/Lab/mgk/schema/cb_data_glr'
    os.makedirs(data_dir, exist_ok=True)

    print("obs_sym:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    hp = get_default_hp(task='obs', platform='laptop')

    hp['btstp'] = 1
    hp['savefig'] = False
    hp['savegenvar'] = False
    hp['savevar'] = False

    hp['alr'] = 0.000005  # reduce for obs
    hp['clr'] = 0.0001  # 0.0001
    hp['taug'] = 10000  # 10000

    hp['usenmc'] = args.usenmc  # confi, neural

    ''' env param'''
    hp['Rval'] = 1
    hp['obs'] = True
    hp['render'] = False  # visualise movement trial by trial

    allcb = [args.cb]
    pool = mp.Pool(processes=hp['cpucount'])

    for cb in allcb:
        hp['contbeta'] = cb

        hp['exptname'] = 'obs_sym_{}cb_{}mb_{}oe_{}gd_{}clr_{}tg_{}alr_{}N_{}R_{}dt_b{}_{}'.format(
            hp['contbeta'], hp['mcbeta'], hp['omite'], hp['gdecay'], hp['clr'], hp['taug'], hp['alr'], hp['nrnn'],
            hp['Rval'], hp['tstep'], hp['btstp'], dt.monotonic())

        totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp,pool)

    pool.close()
    pool.join()

    # save data

    np.savez(f'{data_dir}/obs_sym_cb{args.cb}_{args.seed}s_{args.usenmc}nmc.npz', totlat=totlat, totdgr=totdgr)
