import sys
import os
sys.path.append(os.getcwd())
from backend.model import RecurrentCells, GoalCells
from backend.utils import get_default_hp, saveload, savefig
from backend.maze import run_Rstep
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial


def run_association(hp,b):
    print(hp['nrnn'])
    # cue-coordiante dataset
    np.random.seed(b)
    ncues = hp['ncues']
    gain = 3
    cues = np.eye(ncues) * gain
    goals = np.random.uniform(low=-1,high=1, size=[ncues,2])

    # model
    if hp['rtype'] == 'fb':
        res = RecurrentCells(hp, ninput=ncues+2)
    else:
        res = RecurrentCells(hp, ninput=ncues)
    target = GoalCells(hp)

    phase = ['learn','unlearn']
    toterr = np.zeros([2, ncues])
    allg = np.zeros([ncues,2*hp['time']*1000//20+50,6])

    for nc in np.arange(ncues, ncues + 1):
        target.reset_goal_weights()

        for i, p in enumerate(phase):
            for c in range(nc):
                terr = []
                cue = cues[c]
                goal = goals[c][None, :]
                res.reset()
                target.reset()
                truegoal = np.concatenate([goal, np.array([[1]])], axis=1)
                gt = np.zeros([1,3])
                runR = run_Rstep(hp)

                ti = 0

                if i == 0:
                    idx = 0
                else:
                    idx = hp['time']*1000//20+50

                for t in range(hp['time'] * 1000 // hp['tstep']):

                    if hp['rtype'] == 'fb':
                        rfr = res.process(np.concatenate([cue[None, :],gt[:,:2]],axis=1))
                    else:
                        rfr = res.process(cue[None, :])

                    gt = target.recall(rfr)
                    allg[c,t+idx]= np.concatenate([gt, target.gnoisy],axis=1)

                    if p == 'learn':
                        terr.append(np.mean((truegoal - gt) ** 2))
                        if t == 2*1000/hp['tstep']:
                            R = hp['Rval']
                        else:
                            R = 0

                        reward, done = runR.step(R)
                        if done is False:
                            target.learn_goal(rfr, reward=reward, target=goal)

                    elif p == 'unlearn' and (c == 0 or c == 1 or c==2):

                        ti +=1
                        if ti < 2*1000/hp['tstep'] or ti > 7*1000/hp['tstep']:
                            terr.append(np.mean((truegoal - gt) ** 2))
                            unlearn = False
                        else:
                            unlearn = True

                        if unlearn:
                            if c == 0:
                                target.ach = 0.1
                            elif c == 1:
                                target.ach = 0.01
                            elif c == 2:
                                target.ach = 0.001
                            target.decay_goal(ract=rfr, reward=0)
                    else:
                        terr.append(np.mean((truegoal - gt) ** 2))

                if b == 0:
                    print('NC{} C{} G{} {} {}'.format(nc, c + 1, np.round(goal[0], 2),p, np.round(gt[0], 2)))
                toterr[i, c] = np.mean(terr)
            if b == 0:
                print('NC{} err {:.2g}'.format(nc,np.mean(toterr[i])))

    return toterr, allg


if __name__ == '__main__':

    hp = get_default_hp('6pa',platform='laptop')
    hp['time'] = 10
    hp['tstep'] = 20
    hp['nrnn'] = 1000
    hp['cp'] = [1,0.1]
    hp['choas'] = 1.5
    hp['gtau'] = 100
    hp['gns'] = 0.05  # lms: 0-0.1, EH: 0.1-0.25
    hp['stochlearn'] = True
    hp['glr'] = 7.5e-6  # 0.01/0.005 lms (e=0.008), 0.01/0.0085 EH (e=0.1/e=0.08) | 1e-5/5e-6
    hp['ach'] = 0.05  # 0.1

    hp['ncues'] = 4
    hp['Rval'] = 1
    hp['taua'] = 250
    hp['taub'] = 100
    hp['tolr'] = 1e-8
    hp['wkm'] = False
    hp['rtype'] = 'none'

    exptname = '{}_gbar_{}sl_{}N_{}lr_{}ta_{}ns_{}ach'.format(hp['tolr'], hp['stochlearn'], hp['nrnn'], hp['glr'],hp['taua'], hp['gns'], hp['ach'])
    print(exptname)

    #allN = 2**np.arange(7,12)
    allN = [1000]
    allerr = []

    btstp = 1
    pool = mp.Pool(processes=1)

    for N in allN:

        hp['nrnn'] = N

        x = pool.map(partial(run_association, hp), np.arange(btstp))

        toterr = []
        for b in range(btstp):
            err, trackg = x[b]
            toterr.append(err[None,:])

        toterr = np.vstack(toterr)
        allerr.append(toterr[None,:])

    allerr = np.vstack(allerr)

    # plot
    trackg[:,hp['time']*1000//20:hp['time']*1000//20+50] = np.nan

    saveload('save', './Data/vars_learn_unlearn_ach_{}cues'.format(hp['ncues']), [allerr, trackg])


    import matplotlib as mpl
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams.update({'font.size': 8})

    f = plt.figure(figsize=(7,3))
    plt.axhline(y=0.6,color='r')
    col = ['tab:blue','tab:orange','tab:green']
    for c in range(hp['ncues']):
        plt.subplot(2,3,c+1)
        for i in range(3):
            #plt.plot(trackg[c,:,i+3],alpha=0.2,color=col[i])
            plt.plot(trackg[c, :,i],alpha=1,color=col[i])
        plt.title('Cue {}'.format(c+1))
        plt.axvline(2*1000/hp['tstep'], color='red', linestyle='--')
        plt.axvline(6.5*1000/hp['tstep'],color='k',linestyle='--')
        if c == 0 or c == 1:
            plt.axvline(12*1000/hp['tstep'], color='magenta', linestyle='--')
            plt.axvline(17*1000/hp['tstep'], color='k', linestyle='--')
    f.tight_layout()

    #saveload('save', './vars_learn_unlearn_ach{}'.format(hp['ach'],btstp),  [allerr, trackg])
    #plt.savefig('./Fig/{}_b{}.png'.format(exptname, btstp))
    #savefig('{}_vert_b{}.png'.format(exptname, btstp),f)



    f = plt.figure(figsize=(7,2.75))
    col = ['tab:blue','tab:orange','tab:green']
    for c in range(hp['ncues']):
        plt.subplot(2,2,c+1)
        for i in range(3):
            #plt.plot(trackg[c,:,i+3],alpha=0.2,color=col[i])
            plt.plot(trackg[c, :,i],alpha=1,color=col[i])
            #plt.xticks([])
            plt.xticks([0,250,550,800,1000], [0,5,0,4,10])
            plt.yticks([0,1])

        plt.ylim([-0.5,1.5])
        plt.hlines(y=1.3,xmin=30,xmax=180,color='k')
        plt.text(90,1.4,'3s')
        plt.title('Cue {}'.format(c+1))
        plt.axvline(10, ymax=0.9, color='red', linestyle='--')
        plt.axvline(238,ymax=0.9,color='k',linestyle='--')
        if c == 0 or c == 1 or c==2:
            plt.axvline(650,ymax=0.9, color='magenta', linestyle='--')
            plt.axvline(900,ymax=0.9, color='k', linestyle='--')
            if c == 0:
                plt.text(650, 1.4, 'Ach=0.1')
            elif c == 1:
                plt.text(650, 1.4, 'Ach=0.01')
            elif c == 2:
                plt.text(650, 1.4, 'Ach=0.001')
    f.tight_layout()
    savefig('./Fig/{}_hori_{}cues.png'.format(exptname, hp['ncues']), f)

