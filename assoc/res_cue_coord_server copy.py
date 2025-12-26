#%%
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '../'))

# from backend.model import RecurrentCells, GoalCells
from backend.model_np import RecurrentCells, GoalCells
from backend.utils import get_default_hp, saveload, savefig
from backend.maze import run_Rstep
import numpy as np
import matplotlib.pyplot as plt
import argparse

def run_association(hp,b):
    print(hp['nrnn'])
    # cue-coordiante dataset
    np.random.seed(b)
    ncues = hp['ncues']
    gain = 3
    cues = np.eye(ncues) * gain
    goals = np.random.uniform(low=-1,high=1, size=[ncues,2])

    # model
    res = RecurrentCells(hp, ninput=ncues)
    target = GoalCells(hp)
    target.reset_goal_weights()

    phase = ['associate','recall']
    toterr = np.zeros([2, ncues])

    for c in range(ncues):

        for i, p in enumerate(phase):
            err = []
            if i == 1:
                totaltrials = c+1  # if recall, evaluate all previous cues
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

                res.reset()
                target.reset()
                gt = np.zeros([1, 3])

                runR = run_Rstep(hp)
                trackg = []
                done = False
                reward = 0

                for t in range(hp['time'] * 1000 // hp['tstep']):

                    rfr = res.process(cue[None, :])
                    gt = target.recall(rfr)

                    trackg.append(gt)  # record activity

                    if t > 1 * 1000 // hp['tstep']:  # record performance after 1 second
                        err.append(np.mean((truegoal - gt) ** 2))

                    if p == 'associate':

                        if t == 1*1000//hp['tstep']:  # red line, reward given, start associating after 1 second
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


    parser = argparse.ArgumentParser(description='Run observation symmetry experiment')
    parser.add_argument('--prefix', type=str, default='test', help='Prefix for saving the trained model')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--nrnn', type=int, default=1024, help='N')
    parser.add_argument('--ncues', type=int, default=10, help='Number of cues')
    parser.add_argument('--use_stochlearn', action='store_true', help='stochastic learning (bool)', default=True)
    parser.add_argument('--glr', type=float, default=1e-6, help='goal learning rate') # old 7.5e-6
    parser.add_argument('--chaos', type=float, default=1.5, help='chaos')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='relu, phia, tanh')
    args, unknown = parser.parse_known_args()

    seed = args.seed
    np.random.seed(seed)

    # save data
    fig_dir = f'assoc_res_{args.prefix}'
    data_dir = f'/n/netscratch/pehlevan_lab/Lab/mgk/schema/assoc_{args.prefix}'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("assoc_res:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")


    hp = get_default_hp('6pa',platform='laptop')

    hp['time'] = 11
    hp['tstep'] = 20
    hp['gtau'] = 100  # 50/100
    hp['tau'] = 100

    hp['gns'] = 0.05  # 0.1-0.25
    hp['stochlearn'] = args.use_stochlearn  # True - EH, False - LMS
    hp['glr'] = args.glr  # 0.01/0.005 lms (e=0.008), 0.01/0.0085 EH (e=0.1/e=0.08) | 1e-5/5e-6
    hp['ach'] = 0.000
    hp['resns'] = 0.025

    hp['chaos'] = args.chaos
    hp['cp'] = [1,0.1]
    hp['ract'] = args.nonlinearity  #'relu'  #'phia'  #'tanh'  #

    hp['ncues'] = args.ncues  # 200
    hp['Rval'] = 1
    hp['taua'] = 250
    hp['taub'] = 100
    hp['tolr'] = 1e-8

    hp['nrnn'] = args.nrnn

    exptname = f'res_{hp["nrnn"]}N_{hp["ract"]}_{seed}s_{hp["stochlearn"]}sl_{hp["glr"]}glr_{hp["chaos"]}ch'
    print(exptname)


    err, rg = run_association(hp,seed)

    #%%
    # plot
    # if seed ==0:
    f = plt.figure()
    f.text(0.01,0.01,exptname,fontsize=8)
    plt.plot(np.arange(1,hp['ncues']+1), err[1])
    plt.xlabel('Number of cues')
    plt.ylabel('Recall MSE')
    plt.savefig(f'{fig_dir}/{exptname}.png')


    # np.savez(f'{data_dir}/{exptname}.npz', err=err)

