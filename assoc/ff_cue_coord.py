import sys
import os
sys.path.append(os.getcwd())
from backend.model import FeedForwardCells, GoalCells
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
    ff = FeedForwardCells(hp, ninput=ncues)
    target = GoalCells(hp)
    target.reset_goal_weights()

    phase = ['associate','recall']
    toterr = np.zeros([2, ncues])

    for c in range(ncues):

        for i, p in enumerate(phase):
            err = []
            if i == 1:
                totaltrials = c+1
            else:
                totaltrials = 1

            for nc in range(totaltrials):

                if i == 1:
                    cue = cues[nc]
                    goal = goals[nc][None, :]
                    truegoal = np.concatenate([goal, np.array([[1]])], axis=1)
                else:
                    cue = cues[c]
                    goal = goals[c][None, :]
                    truegoal = np.concatenate([goal, np.array([[1]])], axis=1)

                target.reset()
                gt = np.zeros([1, 3])

                runR = run_Rstep(hp)
                trackg = []
                done = False
                reward = 0

                for t in range(hp['time'] * 1000 // hp['tstep']):

                    rfr = ff.process(cue[None, :])
                    gt = target.recall(rfr)

                    trackg.append(gt)
                    if t > 1 * 1000 // hp['tstep']:
                        err.append(np.mean((truegoal - gt) ** 2))

                    if p == 'associate':

                        if t == 1*1000//hp['tstep']: # start associating after 1 second
                            R = hp['Rval']
                        else:
                            R = 0

                        reward, done = runR.step(R)

                        if not done:
                            target.learn_goal(rfr, reward=reward, target=goal)

                    if done:
                        break

                print('C{} G{} {} {}'.format(c + 1, np.round(goal[0], 2),p, np.round(gt[0], 2)))

            if p == 'associate':
                toterr[0, c] = np.mean(err)
            else:
                toterr[1, c] = np.mean(err)

        #if b == 0:
        print('aerr {:.2g}, rerr {:.2g}'.format(toterr[0, c], toterr[1, c]))

    return toterr, np.vstack(trackg)


if __name__ == '__main__':

    hp = get_default_hp('6pa',platform='server')

    hp['time'] = 6
    hp['tstep'] = 20
    hp['nrnn'] = 1000
    hp['gtau'] = 100  # 50/100
    hp['tau'] = 100

    hp['stochlearn'] = True
    hp['glr'] = 7.5e-6  # 0.01/0.005 lms (e=0.008), 0.01/0.0085 EH (e=0.1/e=0.08) | 1e-5/5e-6
    hp['ach'] = 0.0005

    hp['ract'] = 'relu'

    hp['ncues'] = 200
    hp['Rval'] = 1

    hp['btstp'] = 24

    exptname = 'ff_{}_{}sl_{}c_{}b'.format(hp['ract'],hp['stochlearn'],hp['ncues'],hp['btstp'])
    print(exptname)

    allN = 2**np.arange(7,12)
    #allN = [1000]
    allerr = []

    pool = mp.Pool(processes=hp['btstp'])

    for N in allN:

        hp['nrnn'] = N

        x = pool.map(partial(run_association, hp), np.arange(hp['btstp']))

        toterr = []
        tg = []
        for b in range(hp['btstp']):
            err, rg = x[b]
            toterr.append(err[None,:])
            tg.append(rg[None,:])

        toterr = np.vstack(toterr)  # N X associate/recall X Ncues
        allerr.append(toterr[None,:])

    pool.close()
    pool.join()

    allerr = np.vstack(allerr)
    print(np.mean(np.mean(allerr[:,:,1],axis=1),axis=1))
    saveload('save', 'vars_{}'.format(exptname),  [allerr, rg])

    # plot
    f = plt.figure()
    f.text(0.01,0.01,exptname)
    #plt.subplot(211)
    for n in range(len(allN)):
        plt.errorbar(x=np.arange(1,hp['ncues']+1), y=np.mean(allerr[n,:,1],axis=0), yerr=np.std(allerr[n,:,1],axis=0)/np.sqrt(hp['btstp']))
    plt.legend(allN)
    plt.xlabel('Number of cues')
    plt.ylabel('Recall MSE')
    #plt.title('Avg MSE {:.3g}'.format(np.ronp.mean(np.mean(allerr[:,:,1],axis=1),axis=1)))
    plt.savefig('{}.png'.format(exptname))