import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import matplotlib
import os


def run_hebagent_multiplepa_expt(mtype, env, agent, sessions, noreward, useweight=None):
    env.make(mtype=mtype, noreward=noreward)
    print(f"{mtype} env created. Training ...")
    lat = np.zeros(sessions)
    dgr = []

    if mtype=='train' or mtype == 'nm':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((1, 6,env.normax,2))
    
    if useweight:
        agent.set_weights(useweight)

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

            value, xy, goal, mem = agent.estimate_value_position_goal_memory_td(cpc=cpc,cue=cue,rfr=rfr)

            tderr, tdxy = agent.learn(reward=reward, self_motion=env.dtxy, cpc=cpc, cue=cue, rfr=rfr, xy=xy, goal=goal,
                                      plasticity=plasticity)

            action, rho = agent.get_action(rfr=rfr, xy=xy, goal=goal)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

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

        # Trial information
        print('C{} | S {} | Dgr {:.3g} | Recall Goal {}'.format(
                np.argmax(env.cue)+1, env.i // (1000 // env.tstep), env.dgr, np.round(goal,2)[0]))

        # Session information
        if (t + 1) % 6 == 0:
            print('############## {} Session {}/{}, Avg Steps {:5.1f}, ##############'.format(
                mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1]))
            mdlw = agent.get_weights()
            # print('Coord max {:.3g}, min {:.3g}'.format(np.max(mdlw[0]), np.min(mdlw[0])))
            # print('Critic max {:.3g}, min {:.3g}'.format(np.max(mdlw[1]), np.min(mdlw[1])))
            # print('Actor max {:.3g}, min {:.3g}'.format(np.max(mdlw[2]), np.min(mdlw[2])))

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

    print('Agent {} training dig rate: {}'.format(mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi

class Maze:
    def __init__(self, hp):

        ''' learn Tse et al., 2007 OPA and NPA tasks '''
        self.hp=hp
        self.tstep = hp['tstep'] # simulation time step
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.normax = hp['probetime'] * (1000 // self.tstep)  # Non-rewarded probe trial max time 60s
        self.au = 1.6  # maze size
        self.rrad = 0.03  # reward location radius
        self.testrad = 0.1  # test location radius
        self.stay = False  # when agent reaches reward location, agent position is fixed
        self.rendercall = hp['render']  # plot realtime movement of agent
        self.bounpen = 0.01  # bounce back from boundary
        self.punish = hp['punish']  # no punishment
        self.Rval = hp['Rval']  # reward value when agent reaches reward location
        self.dtxy = np.zeros(2)  # true self-motion
        self.obs = hp['obs']
        self.obssz = 0.08  # 0.05-0.1
        self.hitobs = 0
        self.workmem = hp['wkm']
        self.habittime = 1 * (1000 // self.tstep)  # habituation for 1 second
        self.cuetime = 3 * (1000 // self.tstep) # cue presentation is 3s - 1 s habituation = 2s
        self.ndistract = hp['ndistract']
        self.distfreq = hp['distfreq']
        self.distract = 0
        self.startbox = True

        ''' Define Reward location '''
        self.sclf = hp['cuescl']  # gain for cue
        self.ncues = hp['ncues']
        self.smell = np.eye(self.ncues) * self.sclf  # sensory cue to be passed to agent
        self.holoc = np.zeros([49,2])  # number of reward locations in maze

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        self.obsloc = np.array(
            [9, 10, 11, 12, 17, 36, 37, 38, 39, 31, 24])  # 5, 14, 34, 43
        self.obstacles = self.holoc[self.obsloc]

        inbt = []
        for i in range(len(self.obstacles)):
            for j in range(len(self.obstacles)):
                dist = np.round(np.linalg.norm(self.obstacles[i] - self.obstacles[j], ord=2), 1)
                if dist == 0.2:
                    newld = (self.obstacles[i] + self.obstacles[j])/2
                    inbt.append(newld)
        inbt = np.unique(np.array(inbt),axis=0)
        #dtld = np.array([[-0.3, -0.3], [0.3, 0.3], [-0.3, 0.3], [0.3, -0.3]])
        self.obstacles = np.concatenate([self.obstacles, inbt],axis=0)

    def make(self, mtype='train', nocue=None, noreward=None):
        # make maze environment with reward locations and respective sensory cues
        self.mtype = mtype
        if mtype =='train' or mtype == 'opa':
            self.rlocs = np.array([self.holoc[8],self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35],self.holoc[40]])
            self.cues = self.smell[:6]
            self.totr = 6

        elif mtype == '2npa':
            self.rlocs = np.array(
                [self.holoc[1], self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35], self.holoc[47]])
            self.cues = np.concatenate([self.smell[6][None, :], self.smell[1:5], self.smell[7][None, :]], axis=0)
            self.totr = 6

        elif mtype == '6npa' or mtype == 'nm':
            self.rlocs = np.array(
                [self.holoc[2], self.holoc[19], self.holoc[23], self.holoc[28], self.holoc[32], self.holoc[46]])
            self.cues = self.smell[10:16]
            self.totr = 6

        elif mtype == '6nm':
            self.rlocs = np.array(
                [self.holoc[6], self.holoc[12], self.holoc[18], self.holoc[30], self.holoc[38], self.holoc[40]])
            self.cues = np.concatenate([self.smell[6:10], self.smell[16][None, :], self.smell[17][None, :]], axis=0)
            self.totr = 6


        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*6, i*6)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*6, i*6))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%6 == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(6, 6, replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%6]
        self.rloc = self.rlocs[self.idx]  # reward location at current trial
        self.cue = self.cues[self.idx]  # cue at current trial
        self.cueidx = np.argmax(self.cue)+1
        if self.obs:
            self.x, self.startpos = chosen_pos(self.au,cue=self.cueidx)
        else:
            self.x, self.startpos = randpos(self.au)  # random start position
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []  # track trajectory
        self.tracks.append(self.x)  # include start location
        self.t = trial
        self.cordig = 0  # visit correct location
        self.totdig = 0  # visit a reward location
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(6))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        self.hitbound = False
        self.distract = 0
        self.di = 0
        self.startbox = True
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        if self.workmem:
            if self.i <= self.habittime:
                cue = np.zeros_like(self.cue)
                at = np.zeros_like(at)
                self.startbox = True
            elif self.habittime < self.i <= self.cuetime:
                cue = self.cue
                at = np.zeros_like(at)
                self.startbox = True
            elif self.reward > 0:
                cue = np.zeros_like(self.cue)
                at = np.zeros_like(at)
                self.startbox = False
            else:
                cue = np.zeros_like(self.cue)
                self.startbox = False

            if self.distract < self.ndistract and self.i > (self.cuetime*2):  # start distractor after cuetime from true cue
                self.dprob = np.random.rand() < (self.distfreq * self.tstep) / 1000  # distractor frequency 0.02Hz
                if self.dprob and self.di<1:
                    self.di = 1
                    distractors = np.arange(16,self.ncues)
                    #distractors = np.concatenate([np.arange(9,11),np.arange(16,18)])
                    didx = np.random.choice(distractors, 1)[0]
                    self.distractor = self.smell[didx]

                    # distractors = np.arange(6)
                    # distractors = distractors[distractors!=self.idx]
                    # didx = np.random.choice(distractors, 1)[0]
                    #self.distractor = self.smell[didx]

                    cue = self.distractor
                    self.distract += 1
                    #print('Dist {} @ t={}'.format(didx+1, self.i))

            if 0 < self.di < 1 * (1000 // self.tstep):  # show distractor for 1 second
                self.di += 1
                cue = self.distractor
            else:
                self.di = 0

        else:
            self.startbox = False
            cue = self.cue
            if self.reward > 0:
                # stay at reward location if reached target
                at = np.zeros_like(at)

        xt1 = self.x + at  # update new location

        if self.workmem and self.i < self.cuetime:
            pass
        else:
            if xt1[0] > self.au/2: # right
                xt1 = self.x - self.bounpen * np.array([1,0])
                R = self.punish
                self.hitobs += 1
            elif xt1[0] < -self.au/2: # left
                xt1 = self.x - self.bounpen * np.array([-1, 0])
                R = self.punish
                self.hitobs += 1
            elif xt1[1] < -self.au/2: # bottom
                xt1 = self.x - self.bounpen * np.array([0, -1])
                R = self.punish
                self.hitobs += 1
            elif xt1[1] > self.au/2: # top
                xt1 = self.x - self.bounpen * np.array([0, 1])
                R = self.punish
                self.hitobs += 1

        if self.obs:
            crossb = 0
            if ((np.sum((self.obstacles - self.obssz) < xt1, axis=1) == 2) * (
                        np.sum(xt1 < (self.obstacles + self.obssz), axis=1) == 2)).any():
                crossb += 1

            intc = []
            if np.linalg.norm(at,ord=2)>0:
                for ldmk in self.obstacles:
                    #if np.linalg.norm(ldmk-xt1,2)<self.obssz:
                    intc.append(detect_in_circle(A=self.x, B=xt1, C=ldmk, r=self.obssz))

                if np.sum(intc)>0:
                    crossb += 1

            if crossb > 0:
                xt1 = self.x #- self.bounpen * at / np.linalg.norm(at,ord=2)
                R = self.punish
                self.hitobs += 1

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1
            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.totr == 6:
                    # visit ratio to correct target compared to other targets
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-5)
                else:
                    # visit ratio at correct target over total time
                    self.dgr = np.round(100 * self.cordig / (self.normax), 5)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            # training trial
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, start reward disbursement
                R = self.Rval
                self.stay = True
                self.sessr += 1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2)  # eucledian distance from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1
        self.reward = reward

        return self.x, cue, reward, self.done, distr

    def render(self):
        if self.rendercall:
            #plt.ion()
            fig = plt.figure(figsize=(5, 5))
            col = ['tab:green', 'tab:gray', 'gold', 'tab:orange', 'tab:blue', 'tab:red']
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])
            if self.obs:
                for j in range(len(self.obstacles)):
                    circle = plt.Circle(self.obstacles[j], self.obssz, color='k')
                    self.ax.add_artist(circle)
            # for i in range(49):
            #     self.ax.text(self.holoc[i, 0], self.holoc[i, 1], i, color='r')

            for i in range(len(self.rlocs)):
                circle = plt.Circle(self.rlocs[i], self.rrad, color=col[i])
                self.ax.add_artist(circle)
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
            self.ax.set_title('T{}'.format(self.t+1))
            plt.show(block=False)
            plt.pause(1)

class run_Rstep:
    def __init__(self,hp):
        '''continuous reward function'''
        self.rat = 0
        self.rbt = 0
        self.rt = 0
        self.taua = hp['taua']
        self.taub = hp['taub']
        self.tstep = hp['tstep']
        self.Rval = hp['Rval']
        self.tolr = hp['tolr']
        self.go2s = hp['go2s']
        self.totR = 0
        self.done = False
        self.gotostart = False
        assert self.taua > self.tstep and self.taub > self.tstep, 'Reward time constant less than timestep'

    def convR(self,rat, rbt):
        rat = (1 - (self.tstep / self.taua)) * rat
        rbt = (1 - (self.tstep / self.taub)) * rbt
        rt = (rat - rbt) / (self.taua - self.taub)
        return rat, rbt, rt

    def step(self, R):
        self.rat += R
        self.rbt += R
        self.rat, self.rbt, self.rt = self.convR(self.rat, self.rbt)
        self.totR += self.rt * self.tstep
        if self.totR >= self.Rval * (1-self.tolr): # end after fullR reached
            self.done = True
        if self.totR >= self.Rval * (1-self.go2s):
            self.gotostart = True
        return self.rt, self.done


def randpos(au):
    # to randomly chose a start position
    stpos = (au/2)*np.concatenate([np.eye(2),-1*np.eye(2)],axis=0)
    idx = np.random.choice(4,1, replace=True) # east, north, west, south
    randst = stpos[idx]
    return randst.reshape(-1), idx


def chosen_pos(au, cue):
    stpos = (au/2)*np.concatenate([np.eye(2),-1*np.eye(2)],axis=0)
    # to randomly chose a start position
    allpos = []
    if (cue == np.array([1,4,5,7,11,13,14])).any():
        allpos.append(0)
    if (cue == np.array([3,4,5,6,8,15,16])).any():
        allpos.append(1)
    if (cue == np.array([2,3,6,8,12,15,16])).any():
        allpos.append(2)
    if (cue == np.array([1,2,3,4,7,11,12,13,15])).any():
        allpos.append(3)

    idx = np.random.choice(allpos,1, replace=False) # east, north, west, south
    randst = stpos[idx]
    return randst.reshape(-1), idx

        
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
        #matplotlib.use('tKAgg')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        hp['cpucount'] = 1
    elif hp['platform'] == 'server':
        matplotlib.use('Agg')
        hp['cpucount'] = 2 #mp.cpu_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif hp['platform'] == 'gpu':
        matplotlib.use('tKAgg')
        hp['cpucount'] = 1
    return hp




class ResACAgent:
    def __init__(self, hp):
        ''' Init agent '''

        self.hp = hp
        self.startbox = True
        self.pc = PlaceCells(hp)
        self.coord = PositionCells(hp)
        self.target = GoalCells(hp)
        self.critic = ValueCells(hp)
        self.actor = ActorCells(hp)

        if hp['nmem']!=0:
            self.workmem = BumpAttractorCells(hp)  # with bump attractor
            self.rec = RecurrentCells(hp, ninput=hp['npc']**2+hp['ncues']+hp['nmem'])
        else:
            self.rec = RecurrentCells(hp, ninput=hp['npc']**2+hp['ncues'])

    def see(self, state, cue, startbox):
        self.startbox = startbox
        cpc = tf.cast(self.pc.sense(state)[None,:],dtype=tf.float32)  # place cell activity
        cue = tf.cast(cue[None,:],dtype=tf.float32)  # sensory cue

        if self.startbox: # if in startbox, silence place cells
            pcact = tf.zeros_like(cpc)
        else:
            pcact = cpc

        if self.hp['nmem']!=0:
            I = tf.cast(tf.concat([pcact, cue, self.workmem.bumpfr], axis=1), dtype=tf.float32)
        else:
            I = tf.cast(tf.concat([pcact, cue], axis=1), dtype=tf.float32)

        rfr = self.rec.process(I)

        return cpc, cue, rfr

    def estimate_value_position_goal_memory_td(self, cpc, cue, rfr):
        value = self.critic.est_value(rfr)
        xy = self.coord.est_position(cpc)
        goal = self.target.recall(rfr)

        if self.hp['nmem'] != 0:
            chi = self.workmem.get_gate_activity(ract=rfr)
            mrho = self.workmem.gate_bump_activity(cue=cue, chi=chi)
        else:
            mrho = None

        return value, xy, goal, mrho

    def get_action(self, rfr, xy, goal):
        if self.startbox:  # if startbox, silence actions
            action,rho = np.zeros([2]), tf.zeros([1, self.actor.nact])
        else:
            action, rho = self.actor.move(ract=rfr, xy=xy, goal=goal)
        return action, rho

    def learn(self, reward, self_motion, cpc, cue, rfr, xy, goal, plasticity):
        tderr = self.critic.compute_td(reward)
        tdxy = self.coord.compute_td(self_motion)
        if plasticity:
            # update position coordinates
            self.coord.learn_position(pcact=cpc, tdxy=tdxy)

            # update critic
            self.critic.learn_value(ract=rfr, tderr=tderr)

            # update actor
            self.actor.learn_policy(ract=rfr, tderr=tderr)

            # update target
            self.target.decay_goal(ract=rfr, reward=reward, current=xy)
            self.target.learn_goal(ract=rfr, reward=reward, target=xy)

            # update working memory gate weights
            if self.hp['nmem'] != 0 and self.hp['mlr'] != 0:
                self.workmem.learn_gate(ract=rfr, modulation=tderr, etrace=False)

        return tderr, tdxy

    def state_reset(self):
        self.coord.reset()
        self.rec.reset()
        self.critic.reset()
        self.actor.reset()
        self.target.reset()

        if self.hp['nmem']!=0:
            self.workmem.reset()

    def get_weights(self):
        win = self.rec.win
        wrec = self.rec.wrec
        wcoord = self.coord.wxy
        wgoal = self.target.wgoal
        wact = self.actor.wact
        wcri = self.critic.wval

        if self.hp['nmem']!=0:
            wgate = self.workmem.wgate
            wlat = self.workmem.wlat
            return [wcoord, wcri, wact, win, wrec, wgoal, wgate, wlat]
        else:
            return [wcoord, wcri, wact, win, wrec, wgoal]

    def set_weights(self, mdlw):
        self.coord.wxy = mdlw[0]
        self.critic.wval = mdlw[1]
        self.actor.wact = mdlw[2]
        self.rec.win = mdlw[3]
        self.rec.wrec = mdlw[4]
        self.target.wgoal = mdlw[5]
        if self.hp['nmem']!=0:
            self.workmem.wmem = mdlw[6]
            self.workmem.wlat = mdlw[7]


class GoalCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.ngoal = 3  # [X, Y, StepFunc(Reward)]
        self.galpha = self.tstep / hp['tau']
        self.gns = np.sqrt(1 / self.galpha) * hp['gns']
        self.gnoisy = tf.random.normal(shape=(1, self.ngoal)) * self.gns
        self.gfr = tf.zeros([1,self.ngoal])  # initialise goal units to 0
        self.wgoal = tf.zeros([hp['nrnn'], self.ngoal])
        self.Phat = 0
        self.stochlearn = hp['stochlearn']
        self.goallr = hp['glr']
        self.ach = hp['ach']
        self.achsig = hp['achsig']
        self.bigtheta = choose_activation('bigtheta',hp)

    def reset(self):
        self.gnoisy = tf.random.normal(shape=(1, self.ngoal)) * self.gns
        self.Phat = 0

    def recall(self, ract):
        self.gnoisy = tf.matmul(ract, self.wgoal) + tf.random.normal(shape=(1, self.ngoal)) * self.gns
        self.gfr += self.galpha * (-self.gfr + self.gnoisy)
        return self.gfr

    def learn_goal(self, ract, reward, target):
        StepR = self.bigtheta(reward)

        if StepR != 0:  # dont update weights if R==0, unnecessary compute time
            target_reward = tf.cast(tf.concat([target, np.array([[StepR]])], axis=1),
                                    dtype=tf.float32)  # concatenate current coordiantes & reward
            if self.stochlearn:
                P = -tf.reduce_sum((target_reward - self.gnoisy) ** 2) # compute scalar performance metric
                self.Phat += self.galpha * (-self.Phat + P)  # low pass filtered performance

                if P > self.Phat:  # modulatory factor
                    M = 1
                else:
                    M = 0
                eg = tf.matmul(ract, (self.gnoisy - self.gfr), transpose_a=True)  # EH rule:trace pre x (post - lowpass)
                dwg = eg * M * StepR
            else:
                eg = tf.matmul(ract, (target_reward - self.gfr), transpose_a=True)  # LMS rule
                dwg = eg * StepR  # trace * reward
            self.wgoal += self.tstep * self.goallr * dwg

    def decay_goal(self, ract, reward, current=None):
        ed = tf.matmul(ract, self.gfr, transpose_a=True)
        if self.achsig !=0:
            gd = self.goal_exp(xy=current,goal=self.gfr)
        else:
            gd = 1

        self.wgoal += self.tstep * self.goallr * -self.ach * ed * (self.bigtheta(reward)==0) * gd

    def goal_exp(self, xy, goal):  # increase Ach when near goal location. not necessary
        d = np.linalg.norm(goal[0,:2]-xy)
        gd = np.exp(-d**2/(2*self.achsig**2)) # if agent within 0.01m of reward, ach close to 1
        scalegd = gd * 500  # Ach approximately 0.1
        return scalegd

    def reset_goal_weights(self):
        self.wgoal = tf.zeros_like(self.wgoal)
        print('reset goal synapses')


class PositionCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.nxy = 2  # X,Y coordinate cells
        self.alpha = self.tstep / hp['tau']
        self.xyalpha = self.tstep / hp['xytau']
        self.xyns = np.sqrt(1 / self.xyalpha) * hp['crins']
        self.wxy = tf.zeros([hp['npc']**2, 2])
        self.xylr = hp['xylr']
        self.xywlim = hp['xywlim']
        self.xyfr = tf.zeros((1, self.nxy))  # initialise actor units to 0
        self.pastxy = tf.zeros((1, self.nxy))
        self.tdxy = 0
        self.etxy = 0

    def reset(self):
        self.xyfr = tf.zeros((1, self.nxy))  # reset actor units to 0
        self.tdxy = 0
        self.etxy = 0

    def est_position(self, pcact):
        self.pastxy = tf.identity(self.xyfr)
        I = tf.matmul(pcact, self.wxy)
        sigxy = tf.random.normal(mean=0, stddev=self.xyns, shape=(1, self.nxy), dtype=tf.float32)
        self.xyfr += self.alpha * (-self.xyfr + I + sigxy)
        return self.xyfr

    def compute_td(self, dat):
        tdxy = -dat[None, :] + self.xyfr - self.pastxy
        self.tdxy = tf.reshape(tdxy,(1,self.nxy))
        return self.tdxy

    def learn_position(self, pcact, tdxy):
        self.etxy += self.xyalpha * (-self.etxy + pcact)
        dwxy = tf.matmul(self.etxy, tdxy,transpose_a=True)
        self.wxy += self.xylr * dwxy
        self.wxy = tf.clip_by_value(self.wxy, self.xywlim[1], self.xywlim[0])

    def plot_map(self, hpc=7, vpc=7):
        plt.figure()
        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.imshow(self.wxy[:,i].numpy().reshape(hpc,vpc))
            plt.colorbar()
        plt.show()


class BumpAttractorCells:
    def __init__(self, hp):
        # gating mechanism
        self.tstep = hp['tstep']
        self.wgate = tf.zeros([hp['nrnn'], 2])
        self.gatelr = hp['mlr']

        self.em = tf.zeros([hp['nrnn'], 2])
        self.discount = 1 - self.tstep / hp['taum']
        self.mbeta = hp['membeta']
        self.gatex = tf.zeros([1, 2])
        self.gatefr = tf.zeros((1, 2))

        # ring attractor
        self.nbump = hp['nmem']
        self.alpha = self.tstep / hp['tau']
        self.memns = np.sqrt(1 / self.alpha) * hp['memns']
        self.bumpx = tf.zeros((1, self.nbump))
        self.bumpfr = tf.zeros((1, self.nbump))

        wminus = hp['memw-']  # -10
        wplus = hp['memw+']  # 2
        psi = hp['mempsi']  # 300
        thetaj = (2 * np.pi * np.arange(1, self.nbump + 1)) / self.nbump
        thetadiff = np.tile(thetaj[None, :], (self.nbump, 1)) - np.tile(thetaj[:, None], (1, self.nbump))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nbump)  # normalise self recurrence
        norm = np.sum(f, axis=0)[0]
        self.wlat = tf.cast((wminus / self.nbump) + wplus * f / norm, dtype=tf.float32)
        self.memact = choose_activation(hp['memact'], hp)
        self.memrecact = choose_activation(hp['memrecact'], hp)

        self.winbump = np.zeros([hp['ncues'], hp['nmem']])
        j = 1
        for i in range(18):  # create loading weights from 18 sensory cues to 54 bump neurons
            self.winbump[i, j] = 1
            j += 54 // 18
        self.winbump = tf.cast(self.winbump,dtype=tf.float32)

    def reset(self):
        # reset gate dynamics
        self.gatex = tf.random.normal(shape=(1,2)) * self.memns
        self.gatefr = tf.zeros((1, 2))

        # reset bump attractor dynamics
        self.bumpx = tf.random.normal(shape=(1, self.nbump)) * self.memns  # reset memory units to random state
        self.bumpfr = self.memact(self.bumpx)

    def get_gate_activity(self, ract):
        self.gatex += self.alpha * (-self.gatex + tf.matmul(ract, self.wgate))
        mprob = tf.nn.softmax(self.mbeta * self.gatex)
        chi = np.random.choice(np.arange(mprob.shape[1]), p=np.array(mprob[0]) / np.sum(mprob[0]))
        gatefr = np.zeros_like(self.gatex)
        gatefr[0,chi] = 1  # select either close or open using stochastic policy
        self.gatefr = tf.cast(gatefr,dtype=tf.float32)
        return chi

    def gate_bump_activity(self, cue, chi):
        I = tf.matmul(cue, self.winbump) * chi
        sigq = tf.random.normal(mean=0, stddev=self.memns, shape=(1, self.nbump), dtype=tf.float32)
        lat = tf.matmul(self.memrecact(self.bumpx), self.wlat)
        self.bumpx += self.alpha * (-self.bumpx + I + lat + sigq)
        self.bumpfr = self.memact(self.bumpx)
        return self.bumpfr

    def learn_gate(self, ract, modulation, etrace):
        if etrace:
            self.em += self.discount * (-self.em + tf.matmul(ract, self.gatefr, transpose_a=True))
        else:
            self.em = tf.matmul(ract, self.gatefr,transpose_a=True)
        self.wgate += self.tstep * self.gatelr * self.em * modulation

class ActorCells:
    def __init__(self, hp):
        self.nact = hp['nact']
        self.alat = hp['alat']
        self.tstep = hp['tstep']
        self.astep = hp['maxspeed'] * self.tstep
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        self.aj = tf.cast(self.astep * np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
        self.alpha = self.tstep / hp['tau']
        self.qstate = tf.zeros((1, self.nact))  # initialise actor units to 0
        self.actns = np.sqrt(1 / self.alpha) * hp['actns']
        self.avgspeed = deque(maxlen=int(10 * 1000 / self.tstep))

        wminus = hp['actorw-']  # -1
        wplus = hp['actorw+']  # 1
        psi = hp['actorpsi']  # 20
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        thetadiff = np.tile(thetaj[None, :], (self.nact, 1)) - np.tile(thetaj[:, None], (1, self.nact))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nact)
        norm = np.sum(f, axis=0)[0]
        self.wlat = tf.cast((wminus / self.nact) + wplus * f / norm, dtype=tf.float32)
        self.actact = choose_activation(hp['actact'], hp)
        self.wact = tf.zeros([hp['nrnn'], hp['nact']])
        self.omite = hp['omite']
        self.rho = tf.zeros((1, self.nact))
        self.alr = hp['alr']
        self.awlim = hp['awlim']
        self.contbeta = hp['contbeta']
        self.usenmc = hp['usenmc']
        self.beta = hp['mcbeta']

        if hp['usenmc']:
            self.nmc = tf.keras.models.load_model('../motor_controller/mc_2h128_linear_30mb_31sp_0.6oe_20e_2022-10-08')

    def reset(self):
        self.qstate = tf.zeros((1, self.nact))  # reset actor units to 0
        self.rho = tf.zeros((1, self.nact))

    def move(self, ract, xy, goal):
        Ihid = tf.matmul(ract, self.wact)

        if self.usenmc:
            Imc = self.nmc(tf.concat([goal, xy], axis=1))
        else:
            Imc = self.symbolic_motor_controller(goal=goal, xy=xy)

        I = (1-self.contbeta) * Ihid + self.contbeta * Imc  # combine model-free & schema actions
        sigq = tf.random.normal(mean=0, stddev=self.actns, shape=(1, self.nact), dtype=tf.float32)
        lat = tf.matmul(self.actact(self.qstate), self.wlat)

        self.qstate += self.alpha * (-self.qstate + I + lat + sigq)
        self.rho = self.actact(self.qstate)
        at = tf.matmul(self.aj, self.rho, transpose_b=True).numpy()[:, 0] / self.nact

        movedist = np.linalg.norm(at, 2) * 1000 / self.tstep  # m/s
        self.avgspeed.append(movedist)
        return at, self.rho

    def symbolic_motor_controller(self, goal, xy):
        ''' motor controller to decide direction to move with current position and goal location '''
        if goal[0, -1] > self.omite:  # omit goal if goal is less than threshold
            dircomp = tf.cast(goal[:, :2] - xy, dtype=tf.float32)  # vector subtraction
            qk = tf.matmul(dircomp, self.aj)  # choose action closest to direction to move
            qhat = tf.nn.softmax(self.beta * qk)  # scale action with beta and get probability of action
        else:
            qhat = tf.zeros([1, self.nact])  # if goal below threshold, no action selected by motor controller
        return qhat

    def learn_policy(self, ract, tderr):
        dwa = tf.matmul(ract, self.rho,transpose_a=True) * tderr
        self.wact += self.tstep * self.alr * dwa
        self.wact = tf.clip_by_value(self.wact, self.awlim[1], self.awlim[0])


class ValueCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.ncri = hp['ncri']
        self.alpha = self.tstep / hp['tau']
        self.vstate = tf.zeros((1, self.ncri))  # initialise actor units to 0
        self.crins = np.sqrt(1 / self.alpha) * hp['crins']
        self.criact = choose_activation(hp['criact'], hp)
        self.wval = tf.zeros([hp['nrnn'], hp['ncri']])
        self.pastvalue = tf.zeros((1, self.ncri))
        self.value = tf.zeros((1, self.ncri))
        self.eulerm = hp['eulerm']
        self.tderr = tf.zeros((1, 1))
        self.tdalpha = self.tstep / hp['taug']
        self.clr = hp['clr']
        self.cwlim = hp['cwlim']

    def reset(self):
        self.vstate = tf.zeros((1, self.ncri))  # reset actor units to 0
        self.value = tf.zeros((1, self.ncri))
        self.tderr = tf.zeros((1, 1))

    def est_value(self, ract):
        self.pastvalue = tf.identity(self.value)
        I = tf.matmul(ract, self.wval)
        sigv = tf.random.normal(mean=0, stddev=self.crins, shape=(1, self.ncri), dtype=tf.float32)
        self.vstate += self.alpha * (-self.vstate + I + sigv)
        self.value = self.criact(self.vstate)
        return self.value

    def compute_td(self, reward):
        # forward euler method from Kumar et al. (2022) Cerebral cortex
        tderr = reward + (self.value - (1 + self.tdalpha) * self.pastvalue) / self.tstep
        self.tderr = tf.reshape(tderr,(1,1))
        return self.tderr

    def learn_value(self, ract, tderr):
        # dwv = tf.matmul(ract, self.value,transpose_a=True) * self.tderr 3 factor hebbian doesnt work for critic
        dwv = tf.transpose(ract) * self.tderr
        self.wval += self.tstep * self.clr * dwv
        self.wval = tf.clip_by_value(self.wval, self.cwlim[1], self.cwlim[0])


class RecurrentCells:
    def __init__(self, hp, ninput=None):
        ''' reservoir definition '''
        if ninput:
            self.ninput = ninput
        else:
            self.ninput = hp['npc']**2 + hp['ncues']
        self.nrnn = hp['nrnn']  # number of units
        self.tstep = hp['tstep']
        self.alpha = hp['tstep'] / hp['tau']  # time constant
        self.recns = np.sqrt(1 / self.alpha) * hp['resns']  # white noise
        self.resact = choose_activation(hp['ract'], hp)  # reservoir activation function
        self.resrecact = choose_activation(hp['recact'], hp)  # recurrent activation function
        self.rx = tf.zeros((1, self.nrnn))  # reset recurrent units to 0
        self.rfr = tf.zeros((1, self.nrnn))
        self.cp = hp['cp']  # connection probability
        self.recwinscl = hp['recwinscl']  # input weight scale

        ''' input weight'''
        winconn = np.random.uniform(-self.recwinscl, self.recwinscl, (self.ninput, self.nrnn))  # uniform dist [-1,1]
        self.winprob = np.random.choice([0, 1], (self.ninput, self.nrnn), p=[1 - self.cp[0], self.cp[0]])
        self.win = np.multiply(winconn, self.winprob)  # cater to different input connection probabilities

        ''' recurrent weight '''
        connex = np.random.normal(loc=0, scale=1/np.sqrt(self.cp[1] * self.nrnn), size=(self.nrnn, self.nrnn))
        self.prob = np.random.choice([0, 1], (self.nrnn, self.nrnn), p=[1 - self.cp[1], self.cp[1]])
        # initialise random network with connection probability & gain
        self.wrec = hp['chaos'] * np.multiply(connex, self.prob)  # Hoerzer 2012
        self.wrec *= (np.eye(self.nrnn) == 0)  # remove self recurrence Sompolinsky 1988

    def reset(self):
        self.rx = tf.random.normal(shape=(1, self.nrnn)) * np.sqrt(1 / self.alpha) * 0.1 # reset & kick network into chaotic regime
        self.rfr = self.resact(self.rx)

    def process(self, inputs):
        I = tf.matmul(tf.cast(inputs,dtype=tf.float32), self.win)  # get input current
        rj = tf.matmul(self.resrecact(self.rx), self.wrec)  # get recurrent activity
        rsig = tf.random.normal(shape=(1, self.nrnn)) * self.recns  # white noise

        self.rx += self.alpha * (-self.rx + I + rj + rsig)  # new membrane potential
        self.rfr = self.resact(self.rx)  # reservoir firing rate
        return self.rfr


class PlaceCells:
    def __init__(self, hp):
        self.sigcoeff = 2  # larger coeff makes distribution sharper
        self.npc = hp['npc']  # vpcn * hpcn  # square maze
        self.au = hp['mazesize']
        hori = np.linspace(-self.au / 2, self.au / 2, self.npc)
        vert = np.linspace(-self.au / 2, self.au / 2, self.npc)
        self.pcsig = hori[1] - hori[0]  # distance between each place cell

        self.pcs = np.zeros([self.npc * self.npc, 2])  #  coordinates place cells are selective for
        i = 0
        for x in hori[::-1]:
            for y in vert:
                self.pcs[i] = np.array([y, x])
                i += 1

    def sense(self, s):
        ''' to convert coordinate s to place cell activity '''
        norm = np.sum((s - self.pcs) ** 2, axis=1)
        pcact = np.exp(-norm / (self.sigcoeff * self.pcsig ** 2))
        return pcact

    def check_pc(self, showpc='n'):
        ''' to show place cell distribution on Maze '''
        if showpc == 'y':
            plt.figure()
            plt.scatter(self.pcs[:, 0], self.pcs[:, 1], s=20, c='r')
            plt.axis((-self.au / 2, self.au / 2, -self.au / 2, self.au / 2))
            for i in range(self.npc**2):
                circ = plt.Circle(self.pcs[i], (self.sigcoeff * self.pcsig ** 2), color='g', fill=False)
                plt.gcf().gca().add_artist(circ)
            plt.show()

    def flip_pcs(self):
        #self.pcidx = np.arange(len(self.pcs))
        #np.random.shuffle(self.pcidx)
        #self.pcs = self.pcs[self.pcidx]
        np.random.shuffle(self.pcs)  # randomly shuffle place cell selectivity


def choose_activation(actname, hp=None):
    if actname == 'tanh':
        act = tf.tanh
    elif actname == 'relu':
        act = tf.nn.relu
    elif actname == 'bump':
        def bump_activation(x):
            g01 = (x * tf.cast(tf.keras.backend.greater(x, 0), tf.float32) * tf.cast(
                tf.keras.backend.less_equal(x, 0.5),
                tf.float32)) ** 2
            has_nans = tf.cast(tf.sqrt(2 * x - 0.5), tf.float32) * tf.cast(tf.keras.backend.greater(x, 0.5), tf.float32)
            g1 = tf.where(tf.math.is_nan(has_nans), tf.zeros_like(has_nans), has_nans)
            return g01 + g1
        act = bump_activation
    elif actname == 'phia':
        act = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=hp['sparsity'])
    elif actname == 'softmax':
        act = tf.nn.softmax
    elif actname == 'retanh':
        def retanh(x):
            return tf.tanh(tf.nn.relu(x))
        act = retanh
    elif actname == 'exp':
        act = tf.math.exp
    elif actname == 'bigtheta':
        def bigtheta(x):
            return 1*(x>0)
        act = bigtheta
    else:
        def no_activation(x):
            return x
        act = no_activation
    return act

