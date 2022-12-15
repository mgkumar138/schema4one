import numpy as np
import matplotlib.pyplot as plt


class MultiplePA_Maze:
    def __init__(self, hp):

        ''' Learn 12 NPAs after learn 6 PAs from Tse et al. (2007) '''
        self.hp=hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.normax = hp['probetime'] * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        self.au = 1.6
        self.rrad = 0.03
        self.testrad = 0.1
        self.stay = False
        self.rendercall = hp['render']
        self.bounpen = 0.01
        self.punish = 0  # no punishment
        self.Rval = hp['Rval']
        self.dtxy = np.zeros(2)
        self.startbox = False

        ''' Define Reward location '''
        self.ncues = hp['ncues']
        sclf = hp['cuescl']  # gain for cue
        self.smell = np.eye(self.ncues) * sclf
        self.holoc = np.zeros([49,2])

        self.loci = 0
        self.opaloc = np.array([8, 13, 18, 30, 35, 40])  # to exclude OPA locations from new NPA
        self.npaloc = np.arange(49)
        self.npaloc = np.array(list(set(self.npaloc)-set(self.opaloc)))
        # choose 12 NPA locations randomly for every new agent instantiation
        self.npaloc = np.random.choice(self.npaloc, 36, replace=False)[:36].reshape(3,12)

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

    def make(self, mtype='train', nocue=None, noreward=None):
        self.mtype = mtype

        if mtype =='train':
            self.rlocs = np.array(
                [self.holoc[8], self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35], self.holoc[40]])
            self.totr = 6
            self.cues = self.smell[:6]

        elif mtype == '12npa':
            rlocidx = self.npaloc[self.loci]
            self.rlocs = []
            for r in rlocidx:
                self.rlocs.append(self.holoc[r])
            self.rlocs = np.array(self.rlocs)
            self.cues = self.smell[6:]
            self.totr = 12
            self.loci += 1

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*self.totr, i*self.totr)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*self.totr, i*self.totr))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%self.totr == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(self.totr, self.totr, replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%self.totr]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(self.totr))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        self.hitbound = False
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        cue = self.cue
        if self.reward > 0:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        if xt1[0] > self.au/2: # right
            xt1 = self.x - self.bounpen * np.array([1,0])
            R = self.punish
        elif xt1[0] < -self.au/2: # left
            xt1 = self.x - self.bounpen * np.array([-1, 0])
            R = self.punish
        elif xt1[1] < -self.au/2: # bottom
            xt1 = self.x - self.bounpen * np.array([0, -1])
            R = self.punish
        elif xt1[1] > self.au/2: # top
            xt1 = self.x - self.bounpen * np.array([0, 1])
            R = self.punish

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
                # visit ratio to correct target compared to other targets
                self.dgr = 100 * self.cordig / (self.totdig + 1e-5)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:

            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                R = self.Rval
                self.stay = True
                self.sessr +=1

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
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])
            # for i in range(49):
            #     self.ax.text(self.holoc[i, 0], self.holoc[i, 1], i, color='r')

            for i in range(len(self.rlocs)):
                circle = plt.Circle(self.rlocs[i], self.rrad)
                self.ax.add_artist(circle)
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show(block=False)
        plt.pause(0.001)


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
            
            
class Static_Maze:
    def __init__(self, hp):

        ''' Define Env Parameters '''
        self.hp=hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.normax = hp['probetime'] * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        self.au = 1.6
        self.rrad = 0.03
        self.testrad = 0.1
        self.stay = False
        self.obs = hp['obs']
        self.rendercall = hp['render']
        self.bounpen = 0.01
        self.obssz = 0.08
        self.punish = hp['punish']  # no punishment
        self.Rval = hp['Rval']
        self.t = 0
        self.dtxy = np.zeros(2)
        self.startbox = False

        ''' Define Reward location '''
        ncues = 18
        sclf = hp['cuescl']  # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])
        self.hitobs = 0

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        if hp['obstype'] == 'cave':
            self.obstacles = np.array([[-0.3,0],[-0.3,-0.3],[-0.3,0.3],
                                      [0.3,0], [0.3,0.3],[0.3,-0.3],
                                      [0,0.3]])
            inbt = []
            for i in range(len(self.obstacles)):
                for j in range(len(self.obstacles)):
                    dist = np.round(np.linalg.norm(self.obstacles[i] - self.obstacles[j], ord=2), 1)
                    if dist == 0.3:
                        newld = (self.obstacles[i] + self.obstacles[j]) / 2
                        inbt.append(newld)
            inbt = np.unique(np.array(inbt), axis=0)
            # dtld = np.array([[-0.3, -0.3], [0.3, 0.3], [-0.3, 0.3], [0.3, -0.3]])
            self.obstacles = np.concatenate([self.obstacles, inbt], axis=0)
        elif hp['obstype'] == 'center':
            self.obsloc = np.array(
                [9, 10, 11, 12, 17, 36, 37, 38, 39, 31, 24])  # 5, 14, 34, 43, 26, 33, 22, 15
            self.obstacles = self.holoc[self.obsloc]

            inbt = []
            for i in range(len(self.obstacles)):
                for j in range(len(self.obstacles)):
                    dist = np.round(np.linalg.norm(self.obstacles[i] - self.obstacles[j], ord=2), 1)
                    if dist == 0.2:
                        newld = (self.obstacles[i] + self.obstacles[j]) / 2
                        inbt.append(newld)
            inbt = np.unique(np.array(inbt), axis=0)
            # dtld = np.array([[-0.3, -0.3], [0.3, 0.3], [-0.3, 0.3], [0.3, -0.3]])
            self.obstacles = np.concatenate([self.obstacles, inbt], axis=0)

        elif hp['obstype'] == 'split':
            self.obsloc = np.arange(21,28) # 5, 14, 34, 43
            self.obstacles = self.holoc[self.obsloc]
            dtld = np.array([[-0.8, 0], [0.8,0]])
            self.obstacles = np.concatenate([self.obstacles, dtld], axis=0)

            inbt = []
            for i in range(len(self.obstacles)):
                for j in range(len(self.obstacles)):
                    dist = np.round(np.linalg.norm(self.obstacles[i] - self.obstacles[j], ord=2), 1)
                    if dist == 0.2:
                        newld = (self.obstacles[i] + self.obstacles[j]) / 2
                        inbt.append(newld)
            inbt = np.unique(np.array(inbt), axis=0)
            self.obstacles = np.sort(np.concatenate([self.obstacles, inbt], axis=0),axis=0)

    def make(self, mtype='square', rloc=24, noreward=None):
        self.mtype = mtype
        if rloc is None:
            self.rloc = np.array([1,1])
        else:
            self.rloc = self.holoc[rloc]

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*6, i*6))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if self.mtype == 'square':
            self.x, _ = cave_pos(self.au)
        else:
            self.x = np.zeros(2)
        if self.hp['obstype'] == 'split':
            startpos = np.array([[0,-0.8], [0,0.8]])
            if trial < 3:
                sidx = 0
            # elif 3 <= trial < 6:
            #     sidx=1
            else:
                sidx = np.random.random_integers(low=0,high=1)
                self.obstacles = self.obstacles[:12]
            self.x = startpos[sidx]
        self.startpos = self.x.copy()
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        self.runR = run_Rstep(self.hp)
        return self.x, self.smell[0], self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        # stay at reward location if reached target
        if self.reward > 0:
            self.stay = True
            at = np.zeros_like(at)

        # update new location
        xt1 = self.x + at

        if self.mtype == 'square':
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
        else:
            if np.linalg.norm(xt1) > self.au/2:
                xt1 = self.x - (self.bounpen * at)/np.linalg.norm(at)

        if self.obs:
            crossb = 0
            if ((np.sum((self.obstacles - self.obssz) < xt1, axis=1) == 2) * (
                        np.sum(xt1 < (self.obstacles + self.obssz), axis=1) == 2)).any():
                crossb += 1

            intc = []
            if np.linalg.norm(at,ord=2)>0:
                for bd in self.obstacles:
                    #if np.linalg.norm(ldmk-xt1,2)<self.obssz:
                    intc.append(detect_in_circle(A=self.x, B=xt1, C=bd, r=self.obssz))
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

            if self.i == self.normax:
                self.done = True
                self.dgr = np.round(100 * self.cordig / (self.normax), 5)

        else:  # trianing trial
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, start reward disbursement
                R = self.Rval
                #self.stay = True

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2)  # eucledian distance from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1
        self.reward = reward

        return self.x, self.smell[0], self.reward, self.done, distr

    def render(self, saveplot=False):
        if self.rendercall or saveplot:
            try:
                plt.close(self.fig)
            except AttributeError: print('Maze not saved')

            self.fig = plt.figure(figsize=(5, 5))
            self.ax = self.fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])
            if self.obs:
                for j in range(len(self.obstacles)):
                    circle = plt.Circle(self.obstacles[j], self.obssz, color='gray')
                    self.ax.add_artist(circle)

            circle = plt.Circle(self.rloc, self.rrad, color='r')
            self.ax.add_artist(circle)
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
            self.ax.set_title('T{}'.format(self.t+1))
            plt.show(block=False)
            plt.pause(1)
            if saveplot:
                from backend.utils import savefig
                savefig('{}_{}t_{}i'.format(self.hp['obstype'], self.t, self.normax))
            #plt.close()


class Navex:
    def __init__(self,hp):
        ''' Learning single displaced locations '''
        self.hp = hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # Train max time, 1hr
        self.normax = hp['probetime'] * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        self.au = 1.6
        self.rrad = 0.03
        self.bounpen = 0.01
        self.testrad = 0.1
        self.stay = False
        self.rendercall = hp['render']
        self.Rval = hp['Rval']
        self.punish = hp['punish']
        self.hitobs = 0
        self.dtxy = np.zeros(2)
        self.workmem = hp['wkm']
        self.cuetime = 1 * (1000 // self.tstep)
        self.habittime = 1 * (1000 // self.tstep)
        self.startbox = False

        ''' Define Reward location '''
        ncues = hp['ncues']
        holes = np.linspace((-self.au/2)+0.2,(self.au/2)-0.2,7) # each reward location is 20 cm apart
        sclf = hp['cuescl'] # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        ''' sequence '''
        self.locseq = np.random.choice(np.arange(49),49,replace=False)
        self.loci = 0

        ''' create dig sites '''
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

    def make(self, mtype=1, nocue=None, noreward=None):
        self.mtype = mtype
        assert isinstance(mtype,int)

        rlocsidx = self.locseq[self.loci]

        self.rlocs = []
        for r in range(mtype):
            self.rlocs.append(self.holoc[rlocsidx])
            #self.rlocs.append(self.holoc[24]) (0,0) coordinate
        self.rlocs = np.array(self.rlocs)
        self.cues = np.tile(self.smell[0],(len(self.smell),1))  # np.zeros_like(self.smell)
        self.loci += 1

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*len(self.rlocs), i*len(self.rlocs))) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*len(self.rlocs), i*len(self.rlocs)))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%len(self.rlocs) == 0: # reset order of cues presented after NR trials
            self.ridx = np.random.choice(np.arange(len(self.rlocs)), len(self.rlocs), replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%len(self.rlocs)]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(len(self.rlocs)))
        self.mask.remove(self.idx)
        self.hitbound = False
        if trial in self.nort:
            self.probe = True
        else:
            self.probe = False
        return self.x,self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        if self.workmem:
            if self.habittime < self.i < self.cuetime:
                cue = self.cue
                at = np.zeros_like(at)
            elif self.reward > 0:
                cue = np.zeros_like(self.cue)
                at = np.zeros_like(at)
            else:
                cue = np.zeros_like(self.cue)
        else:
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

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            # time spent = location within 0.1m near reward location with no overlap of other locations
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1

            if self.mtype>1:
                for orl in self.rlocs[self.mask]:
                    if np.linalg.norm(orl-xt1,2)<self.testrad:
                        self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.mtype == 1:
                    self.dgr = np.round(100 * self.cordig / self.normax, 5)
                else:
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-10)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, start reward disbursement
                R = self.Rval
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2) # eucledian distance away from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1
        self.reward = reward

        return self.x, cue, reward, self.done, distr

    def render(self):
        if self.rendercall:
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

            circle = plt.Circle(self.rloc, self.rrad, color='r')
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


def cave_pos(au):
    stpos = (au/2)*np.concatenate([np.eye(2),-1*np.eye(2)],axis=0)
    allpos = [0,1,2]
    idx = np.random.choice(allpos,1, replace=False) # east, north, west, south
    randst = stpos[idx]
    return randst.reshape(-1), idx


def detect_in_circle( A, B, C, r ):

    # First, let's express each vector as a complex number.
    # This simplifies the rest of the code because we can then subtract them
    # from each other in one statement, or find their length with one statement.
    # (Downside: it does not allow us to generalize the code to spheres in 3D.)
    OA = complex( *A )
    OB = complex( *B )
    OC = complex( *C )

    # Now let's translate into a coordinate system where A is the origin
    AB = OB - OA
    AC = OC - OA

    # Before we go further let's cover one special case:  if either A or B is actually in
    # the circle,  then mark it as a detection
    BC = OC - OB
    if abs( BC ) < r or abs( AC ) < r: return True

    # Project C onto the line to find P, the point on the line that is closest to the circle centre
    AB_normalized = AB / abs( AB )
    AP_distance = AC.real * AB_normalized.real  +  AC.imag * AB_normalized.imag    # dot product (scalar result)
    AP = AP_distance * AB_normalized   # actual position of P relative to A (vector result)

    # If AB intersects the circle, and neither A nor B itself is in the circle,
    # then P, the point on the extended line that is closest to the circle centre, must be...

    # (1) ...within the segment AB:
    AP_proportion = AP_distance / abs(AB)  # scalar value: how far along AB is P?
    in_segment = 0 <= AP_proportion <= 1

    # ...and (2) within the circle:
    CP = AP - AC
    in_circle = abs(CP) < r

    detected = in_circle and in_segment

    return detected


