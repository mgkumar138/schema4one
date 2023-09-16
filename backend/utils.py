import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp
import pickle
import matplotlib
import os
import sys
import multiprocessing as mp

def get_default_hp(task, platform='laptop'):
    if task == 'dmp':
        epochs = 9
        trsess = 5
        evsess = None
        time = 300
        probetime = 60
        ach = 0.0005
    elif task =='6pa':
        epochs = None
        trsess = 20
        evsess = 2
        time = 600
        probetime = 60
        ach = 0.00005
    elif task =='obs':
        epochs = None
        trsess = 50
        evsess = 2
        time = 600
        probetime = 60
        ach = 0.00005
    elif task =='wkm':
        epochs = None
        trsess = 20
        evsess = 2
        time = 600
        probetime = 60
        ach = 0.00005
    elif task == 'center':
        epochs = 1
        trsess = 10
        evsess = 0
        time = 300
        probetime = 60
        ach = 0.00005
    else:
        print('No Task selected')
        sys.exit(1)

    hp = {
        # Environment parameters
        'mazesize': 1.6,  # meters
        'task': task,  # type of task determiens number of training, evaluation sessions
        'tstep': 20,  # ms each step taken is every 100 ms
        'time': time,  # seconds total trial time
        'probetime': probetime,  # seconds total probe time
        'render': False,  # dont plot real time movement
        'epochs': epochs,  # only for single displaced location task
        'trsess': trsess,  # number of training trials
        'evsess': evsess,  # number of evaluation trials
        'platform': platform,  # laptop, gpu or server with multiple processors
        'wkm': False,

        'taua': 250,  # reward decay time
        'taub': 100,  # reward rise time
        'Rval': 1,  # magnitude of reward disbursed
        'tolr':1e-8,  # stop trial after Rvalue * (1-totr) reached
        'go2s':1e-4,

        # input parameters
        'ncues': 18, # number of cues
        'npc':7,  # number of place cells across vertical and horizontal axes
        'cuescl': 3,  # gain of cue
        'punish': -0.0, #0
        'obs': False,  # presence of obstacles

        # actor parameters:
        'nact': 40,  # number of actor units
        'actact': 'relu',  # activation of actor units
        'alat': True,  # use lateral connectivity for ring attractor dynamics
        'actns': 0.25,  # exploratory noise for actor
        'maxspeed': 0.03,  # a0 scaling factor for veloctiy
        'actorw-': -1,  # inhibitory scale for lateral connectivity
        'actorw+': 1,  # excitatory scale for lateral connectivity
        'actorpsi': 20,  # lateral connectivity spread
        'tau': 100,  # membrane time constant for all cells
        'awlim': [0.001,-0.001],  # clip weights beyond limit
        'alr': 0.00001,  # actor learning rate
        'contbeta': 0.4,  # proportion of contribution from actor-critic (0) or NAVIGATE schema (1)

        # critic params
        'ncri': 1,  # number of critic
        'criact': 'relu',  # critic activation function
        'crins': 1e-8,  # critic noise
        'eulerm': 1,  # euler method to discretize continuous Temporal difference error, see Kumar et al. (2022) Cerebral cortex
        'cwlim': [0.025,-0.025],  # clip critic weights beyond limit
        'clr': 0.0001,  # critic learning rate
        'taug': 10000,  # 3000 (no obs)/10000 (obs)

        # reservoir parameters
        'hidtype':'rnn',  # feedforward (ff) or reservoir (rnn)
        'ract': 'phia',  # reservoir activation function
        'recact': 'tanh',  # reservoir recurrent activiation function
        'chaos': 1.5,  # chaos gain lambda
        'cp': [1, 0.1],  # connection probability - input to reservoir & within reservoir
        'resns': 0.025,  # white noise in reservoir
        'recwinscl': 1,  # reservoir input weight scale
        'nrnn': 1000,  # number of rnn units
        'sparsity': 3,  # threshold for ReLU activation function

        # motor controller parameters
        'omite': 0.6,  # threshold to omit goal and suppress motor controller
        'mcbeta': 30,  # motor controller beta
        'xylr': 0.01,  # learning rate of self position coordinate network
        'xywlim': [1, -1],
        'xytau': 200,  # eligibility trace to learn path integration
        'recallbeta': 1,  # recall beta within symbolic memory
        'usenmc': True,  # use preptrained neural network motor controller (True) or symbolic function (False)
        'learnxy': True,  # learn metric representation or use symbolic coordinates from environment

        # goal/sym params
        'gdecay': 'both',  # both for symbolic
        'gdist': 0.01,  # 0.01 for symbolic
        'gns': 0.05,  # white noise for goal coordinates
        'glr': 7.5e-6,  # learning rate for association network
        'gwlim': [1,-1],  # clip weights
        'stochlearn': True,  # use Exploratory Hebbian rule (True) or Perceptron rule (False) for one-shot association
        'ach': ach,  # Acetylecholine for synaptic depression single 0.0005, 6pa 0.0001, obs 0.00005
        'achsig': 0,  # if 0, ach is constant. else, ach secreted only when agent is near goal coordinate achsig=0.1

        # working memory/bump attractor
        'nmem':0,  # number of bump neurons
        'memns':0.1,  # bump attractor noise
        'mempsi':300,  # tune width of activation
        'memw+':2,
        'memw-':-10,
        'membeta':2,
        'memact':'relu',
        'memrecact':'bump',
        'bumpf':True,  # bump attractor neuron has self-recurrence (False) or not (True)
        'taum': 10000,  # time constant for eligibility trace * DA. not necessary if using only DA to modulate hebbian rule
        'mwlim':[1,-1],
        'memlearnrule':'da',  # either use dopamine (da) or eligibility trace * dopamine (etda)
        'ndistract':2,  # number of distractors during a trial
        'distfreq': 0.2,  # distractor presentation frequency
        'usegate': False,

        # others
        'savefig': True,  # save output figure
        'savegenvar': True,  # save compiled variables latency, visit ratio
        'savevar': True,  # individual run variables
    }

    if hp['platform'] == 'laptop':
        matplotlib.use('Agg')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        hp['cpucount'] = 1
    elif hp['platform'] == 'server':
        matplotlib.use('tKAgg')
        hp['cpucount'] = 2 #mp.cpu_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif hp['platform'] == 'gpu':
        matplotlib.use('tKAgg')
        hp['cpucount'] = 1
    return hp


def saveload(opt, name, variblelist):
    if opt == 'save':
        name = name + '.pickle'
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var


def savefig(name,f):
    f.savefig('{}.svg'.format(name))
    f.savefig('{}.png'.format(name))
    print('Fig saved as .png & .svg')


def save_rdyn(rdyn, mtype,t,startpos,cue, rfr):
    rfr = np.array(rfr)
    if '{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], np.argmax(cue)+1) in rdyn:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], np.argmax(cue)+1)].append(rfr[0])
    else:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], np.argmax(cue)+1)] = []
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], np.argmax(cue)+1)].append(rfr[0])


def get_savings(latency):
    savings = latency[:,:,0]-latency[:,:,1]
    return np.mean(savings,axis=0), np.std(savings,axis=0)/np.sqrt(latency.shape[0])

def get_savings_ind(latency):
    return latency[:,:,0]-latency[:,:,1]



def plot_1pa_maps_dmp(alldyn, mvpath, hp, pweights, pltidx=[3,4,10]):
    qdyn = alldyn[1]
    cdyn = alldyn[2]
    nonrlen = np.array(qdyn[list(qdyn.keys())[0]]).shape[0]
    bins = 15
    cues = 1
    sess = list(cdyn.keys())
    for p in range(len(pweights)):
        qfr = np.zeros([cues, nonrlen, 40])
        cfr = np.zeros([cues, nonrlen, 2])
        newx = np.zeros([225,2])
        for i in range(15):
            st = i * 15
            ed = st + 15
            newx[st:ed, 0] = np.arange(15)
        for i in range(15):
            st = i * 15
            ed = st + 15
            newx[st:ed, 1] = i * np.ones(15)

        qfr[0] = np.array(qdyn[sess[p]])[-nonrlen:]
        cfr[0] = np.array(cdyn[sess[p]])[-nonrlen:]

        qfr = np.reshape(qfr, newshape=(cues * nonrlen, 40))
        cfr = np.reshape(cfr, newshape=(cues * nonrlen, 2))

        coord = mvpath[p][:nonrlen]

        policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr)

        actorw = pweights[p][2]
        criticw = pweights[p][1]

        plt.subplot(pltidx[0], pltidx[1], p + pltidx[2])
        im = plt.imshow(valuemap.T, aspect='auto', origin='lower')
        plt.quiver(newx[:, 1], newx[:, 0], policymap[0].reshape(bins ** 2), policymap[1].reshape(bins ** 2),
                   units='xy', color='w')
        plt.ylabel('C [{:.2g},{:.2g}]'.format(np.max(criticw), np.min(criticw)))
        plt.title('A [{:.2g},{:.2g}]'.format(np.max(actorw), np.min(actorw)))
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().set_aspect('equal', adjustable='box')


def plot_1pa_maps_center(alldyn, mvpath, hp, pweights, pltidx=[3,4,10]):
    qdyn = alldyn[1]
    cdyn = alldyn[2]
    nonrlen = np.array(qdyn[list(qdyn.keys())[0]]).shape[0]
    bins = 15
    trials = [2, 6, 10]
    cues = 6
    for p in range(len(pweights)):
        trial = trials[p]
        qfr = np.zeros([cues, nonrlen, 40])
        cfr = np.zeros([cues, nonrlen, 2])
        coord = np.zeros([nonrlen * cues, 2])
        newx = np.zeros([225,2])
        for i in range(15):
            st = i * 15
            ed = st + 15
            newx[st:ed, 0] = np.arange(15)
        for i in range(15):
            st = i * 15
            ed = st + 15
            newx[st:ed, 1] = i * np.ones(15)

        sess = [v for v in cdyn.keys() if v.startswith('square_s{}'.format(trial))]
        for c,s in enumerate(sess):
            qfr[c] = np.array(qdyn[s])[-nonrlen:]
            cfr[c] = np.array(cdyn[s])[-nonrlen:]

        qfr = np.reshape(qfr, newshape=(cues * nonrlen, 40))
        cfr = np.reshape(cfr, newshape=(cues * nonrlen, 2))

        for i, s in enumerate(sess):
            st = i * nonrlen
            ed = st + nonrlen
            coord[st:ed] = mvpath[p, i][-nonrlen:]

        policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr)

        actorw = pweights[p][2]
        criticw = pweights[p][1]

        plt.subplot(pltidx[0],pltidx[1],p+pltidx[2])
        im = plt.imshow(valuemap.T,aspect='auto',origin='lower')
        plt.quiver(newx[:, 1], newx[:, 0], policymap[0].reshape(bins ** 2), policymap[1].reshape(bins ** 2),
                         units='xy',color='w')
        plt.ylabel('C [{:.2g},{:.2g}]'.format(np.max(criticw), np.min(criticw)))
        plt.title('A [{:.2g},{:.2g}]'.format(np.max(actorw), np.min(actorw)))
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().set_aspect('equal', adjustable='box')


def get_binned_stat(hp, coord, qfr, cfr, bins=15):
        from backend.model import ActorCells
        from scipy.stats import binned_statistic_2d
        actor = ActorCells(hp)
        qpolicy = np.matmul(actor.aj, qfr.T)

        policymap = np.zeros([2, bins, bins])
        policymap[0] = binned_statistic_2d(coord[:, 0], coord[:, 1], qpolicy[0], bins=bins, statistic='mean')[0]
        policymap[1] = binned_statistic_2d(coord[:, 0], coord[:, 1], qpolicy[1], bins=bins, statistic='mean')[0]
        valuemap = binned_statistic_2d(coord[:, 0], coord[:, 1], cfr[:, 0], bins=bins, statistic='mean')[0]
        return np.nan_to_num(policymap), np.nan_to_num(valuemap)

def plot_single_map(qfr, cfr,coord, hp):
    bins = 15
    newx = np.zeros([225,2])
    for i in range(15):
        st = i * 15
        ed = st + 15
        newx[st:ed, 0] = np.arange(15)
    for i in range(15):
        st = i * 15
        ed = st + 15
        newx[st:ed, 1] = i * np.ones(15)

    policymap, valuemap = get_binned_stat(hp, coord, qfr, cfr)

    im = plt.imshow(valuemap.T,aspect='auto',origin='lower')
    plt.quiver(newx[:, 1], newx[:, 0], policymap[0].reshape(bins ** 2), policymap[1].reshape(bins ** 2),
                     units='xy',color='w')

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.gca().set_aspect('equal', adjustable='box')

def compute_auc(learncurves):
    return np.trapz(learncurves)
