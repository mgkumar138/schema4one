import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf

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

        if hp['nmem'] != 0:
            self.workmem = BumpAttractorCells(hp)  # with bump attractor
            self.rec = RecurrentCells(hp, ninput=hp['npc']**2 + hp['ncues'] + hp['nmem'])
        else:
            self.rec = RecurrentCells(hp, ninput=hp['npc']**2 + hp['ncues'])

    def see(self, state, cue, startbox):
        self.startbox = startbox
        cpc = self.pc.sense(state)[None, :].astype(np.float32)  # place cell activity
        cue = cue[None, :].astype(np.float32)  # sensory cue

        if self.startbox:  # if in startbox, silence place cells
            pcact = np.zeros_like(cpc)
        else:
            pcact = cpc

        if self.hp['nmem'] != 0:
            I = np.concatenate([pcact, cue, self.workmem.bumpfr], axis=1).astype(np.float32)
        else:
            I = np.concatenate([pcact, cue], axis=1).astype(np.float32)

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
            action, rho = np.zeros([2]), np.zeros([1, self.actor.nact])
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

        if self.hp['nmem'] != 0:
            self.workmem.reset()

    def get_weights(self):
        win = self.rec.win
        wrec = self.rec.wrec
        wcoord = self.coord.wxy
        wgoal = self.target.wgoal
        wact = self.actor.wact
        wcri = self.critic.wval

        if self.hp['nmem'] != 0:
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
        if self.hp['nmem'] != 0:
            self.workmem.wmem = mdlw[6]
            self.workmem.wlat = mdlw[7]


class GoalCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.ngoal = 3  # [X, Y, StepFunc(Reward)]
        self.galpha = self.tstep / hp['tau']
        self.gns = np.sqrt(1 / self.galpha) * hp['gns']
        self.gnoisy = np.random.normal(size=(1, self.ngoal)) * self.gns
        self.gfr = np.zeros([1, self.ngoal])  # initialise goal units to 0
        self.wgoal = np.zeros([hp['nrnn'], self.ngoal])
        self.Phat = 0
        self.stochlearn = hp['stochlearn']
        self.goallr = hp['glr']
        self.ach = hp['ach']
        self.achsig = hp['achsig']
        self.bigtheta = choose_activation('bigtheta', hp)

    def reset(self):
        self.gnoisy = np.random.normal(size=(1, self.ngoal)) * self.gns
        self.Phat = 0

    def recall(self, ract):
        self.gnoisy = np.dot(ract, self.wgoal) + np.random.normal(size=(1, self.ngoal)) * self.gns
        self.gfr += self.galpha * (-self.gfr + self.gnoisy)
        return self.gfr

    def learn_goal(self, ract, reward, target):
        StepR = self.bigtheta(reward)

        if StepR != 0:  # dont update weights if R==0, unnecessary compute time
            target_reward = np.concatenate([target, np.array([[StepR]])], axis=1).astype(np.float32)
            if self.stochlearn:
                P = -np.sum((target_reward - self.gnoisy) ** 2)  # compute scalar performance metric
                self.Phat += self.galpha * (-self.Phat + P)  # low pass filtered performance

                if P > self.Phat:  # modulatory factor
                    M = 1
                else:
                    M = 0
                eg = np.dot(ract.T, (self.gnoisy - self.gfr))  # EH rule:trace pre x (post - lowpass)
                dwg = eg * M * StepR
            else:
                eg = np.dot(ract.T, (target_reward - self.gfr))  # LMS rule
                dwg = eg * StepR  # trace * reward
            self.wgoal += self.tstep * self.goallr * dwg

    def decay_goal(self, ract, reward, current=None):
        ed = np.dot(ract.T, self.gfr)
        if self.achsig != 0:
            gd = self.goal_exp(xy=current, goal=self.gfr)
        else:
            gd = 1

        self.wgoal += self.tstep * self.goallr * -self.ach * ed * (self.bigtheta(reward) == 0) * gd

    def goal_exp(self, xy, goal):  # increase Ach when near goal location. not necessary
        d = np.linalg.norm(goal[0, :2] - xy)
        gd = np.exp(-d**2 / (2 * self.achsig**2))  # if agent within 0.01m of reward, ach close to 1
        scalegd = gd * 500  # Ach approximately 0.1
        return scalegd

    def reset_goal_weights(self):
        self.wgoal = np.zeros_like(self.wgoal)
        print('reset goal synapses')


class PositionCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.nxy = 2  # X,Y coordinate cells
        self.alpha = self.tstep / hp['tau']
        self.xyalpha = self.tstep / hp['xytau']
        self.xyns = np.sqrt(1 / self.xyalpha) * hp['crins']
        self.wxy = np.zeros([hp['npc']**2, 2])
        self.xylr = hp['xylr']
        self.xywlim = hp['xywlim']
        self.xyfr = np.zeros((1, self.nxy))  # initialise actor units to 0
        self.pastxy = np.zeros((1, self.nxy))
        self.tdxy = 0
        self.etxy = 0

    def reset(self):
        self.xyfr = np.zeros((1, self.nxy))  # reset actor units to 0
        self.tdxy = 0
        self.etxy = 0

    def est_position(self, pcact):
        self.pastxy = np.copy(self.xyfr)
        I = np.dot(pcact, self.wxy)
        sigxy = np.random.normal(0, self.xyns, size=(1, self.nxy)).astype(np.float32)
        self.xyfr += self.alpha * (-self.xyfr + I + sigxy)
        return self.xyfr

    def compute_td(self, dat):
        tdxy = -dat[None, :] + self.xyfr - self.pastxy
        self.tdxy = np.reshape(tdxy, (1, self.nxy))
        return self.tdxy

    def learn_position(self, pcact, tdxy):
        self.etxy += self.xyalpha * (-self.etxy + pcact)
        dwxy = np.dot(self.etxy.T, tdxy)
        self.wxy += self.xylr * dwxy
        self.wxy = np.clip(self.wxy, self.xywlim[1], self.xywlim[0])

    def plot_map(self, hpc=7, vpc=7):
        plt.figure()
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.imshow(self.wxy[:, i].reshape(hpc, vpc))
            plt.colorbar()
        plt.show()


class BumpAttractorCells:
    def __init__(self, hp):
        # gating mechanism
        self.tstep = hp['tstep']
        self.wgate = np.zeros([hp['nrnn'], 2])
        self.gatelr = hp['mlr']

        self.em = np.zeros([hp['nrnn'], 2])
        self.discount = 1 - self.tstep / hp['taum']
        self.mbeta = hp['membeta']
        self.gatex = np.zeros([1, 2])
        self.gatefr = np.zeros((1, 2))

        # ring attractor
        self.nbump = hp['nmem']
        self.alpha = self.tstep / hp['tau']
        self.memns = np.sqrt(1 / self.alpha) * hp['memns']
        self.bumpx = np.zeros((1, self.nbump))
        self.bumpfr = np.zeros((1, self.nbump))

        wminus = hp['memw-']  # -10
        wplus = hp['memw+']  # 2
        psi = hp['mempsi']  # 300
        thetaj = (2 * np.pi * np.arange(1, self.nbump + 1)) / self.nbump
        thetadiff = np.tile(thetaj[None, :], (self.nbump, 1)) - np.tile(thetaj[:, None], (1, self.nbump))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nbump)  # normalise self recurrence
        norm = np.sum(f, axis=0)[0]
        self.wlat = ((wminus / self.nbump) + wplus * f / norm).astype(np.float32)
        self.memact = choose_activation(hp['memact'], hp)
        self.memrecact = choose_activation(hp['memrecact'], hp)

        self.winbump = np.zeros([hp['ncues'], hp['nmem']])
        j = 1
        for i in range(18):  # create loading weights from 18 sensory cues to 54 bump neurons
            self.winbump[i, j] = 1
            j += 54 // 18
        self.winbump = self.winbump.astype(np.float32)

    def reset(self):
        # reset gate dynamics
        self.gatex = np.random.normal(size=(1, 2)) * self.memns
        self.gatefr = np.zeros((1, 2))

        # reset bump attractor dynamics
        self.bumpx = np.random.normal(size=(1, self.nbump)) * self.memns  # reset memory units to random state
        self.bumpfr = self.memact(self.bumpx)

    def get_gate_activity(self, ract):
        self.gatex += self.alpha * (-self.gatex + np.dot(ract, self.wgate))
        mprob = softmax(self.mbeta * self.gatex)
        chi = np.random.choice(np.arange(mprob.shape[1]), p=np.array(mprob[0]) / np.sum(mprob[0]))
        gatefr = np.zeros_like(self.gatex)
        gatefr[0, chi] = 1  # select either close or open using stochastic policy
        self.gatefr = gatefr.astype(np.float32)
        return chi

    def gate_bump_activity(self, cue, chi):
        I = np.dot(cue, self.winbump) * chi
        sigq = np.random.normal(0, self.memns, size=(1, self.nbump)).astype(np.float32)
        lat = np.dot(self.memrecact(self.bumpx), self.wlat)
        self.bumpx += self.alpha * (-self.bumpx + I + lat + sigq)
        self.bumpfr = self.memact(self.bumpx)
        return self.bumpfr

    def learn_gate(self, ract, modulation, etrace):
        if etrace:
            self.em += self.discount * (-self.em + np.dot(ract.T, self.gatefr))
        else:
            self.em = np.dot(ract.T, self.gatefr)
        self.wgate += self.tstep * self.gatelr * self.em * modulation


class SymACAgent:
    def __init__(self, hp):
        ''' env step, '''
        self.hp = hp
        self.startbox = True
        self.pc = PlaceCells(hp)
        self.coord = PositionCells(hp)
        self.critic = ValueCells(hp)
        self.actor = ActorCells(hp)
        self.memory = SymMemory(hp)
        if self.hp['hidtype'] == 'ff':
            self.hidden = FeedForwardCells(hp)
        else:
            self.rec = RecurrentCells(hp)

    def see(self, state, cue, startbox):
        self.startbox = startbox
        cpc = self.pc.sense(state)[None, :].astype(np.float32)
        cue = cue[None, :].astype(np.float32)

        if self.startbox:
            pcact = np.zeros_like(cpc)
        else:
            pcact = cpc

        if self.hp['hidtype'] == 'ff':
            rfr = self.hidden.process(np.concatenate([pcact, cue], axis=1).astype(np.float32))
        else:
            rfr = self.rec.process(np.concatenate([pcact, cue], axis=1).astype(np.float32))

        return cpc, cue, rfr

    def estimate_value_position_goal(self, cpc, cue, rfr):
        value = self.critic.est_value(rfr)
        xy = self.coord.est_position(cpc)
        goal = self.memory.recall(cue)  # or can pass rfr as query
        return value, xy, goal

    def get_action(self, rfr, xy, goal):
        if self.startbox:
            action, rho = np.zeros([2]), np.zeros([1, self.actor.nact])
        else:
            action, rho = self.actor.move(ract=rfr, xy=xy, goal=goal)
        return action, rho

    def learn(self, reward, self_motion, cpc, rfr, plasticity):
        tderr = self.critic.compute_td(reward)
        tdxy = self.coord.compute_td(self_motion)
        if plasticity:
            # update critic
            self.critic.learn_value(ract=rfr, tderr=tderr)

            # update actor
            self.actor.learn_policy(ract=rfr, tderr=tderr)

            # update position
            self.coord.learn_position(pcact=cpc, tdxy=tdxy)
        return tderr, tdxy

    def learn_cue_location(self, cue, xy, goal, reward, done, plasticity):
        if plasticity:
            # update memory
            self.memory.store(query=cue, value=xy, reward=reward)

            if self.hp['gdecay'] == 'dist' or self.hp['gdecay'] == 'both':
                self.memory.delete_goal_dist(xy=xy, goal=goal, query=cue, reward=reward)

            if self.hp['gdecay'] == 'done' or self.hp['gdecay'] == 'both':
                self.memory.delete_goal(query=cue, reward=reward, done=done)

    def state_reset(self):
        self.coord.reset()
        if self.hp['hidtype'] != 'ff':
            self.rec.reset()
        self.critic.reset()
        self.actor.reset()

    def get_weights(self):
        wcoord = self.coord.wxy
        wact = self.actor.wact
        wcri = self.critic.wval
        return [wcoord, wcri, wact]

    def set_weights(self, mdlw):
        self.coord.wxy = mdlw[0]
        self.critic.wval = mdlw[1]
        self.actor.wact = mdlw[2]


class SymMemory:
    def __init__(self, hp):
        self.ncues = hp['ncues']
        self.recallbeta = hp['recallbeta']
        self.keyvalue = np.zeros([50, self.ncues+3], dtype=np.float32)  # combine key & value matrix
        self.gdist = hp['gdist']
        self.omite = hp['omite']
        self.bigtheta = choose_activation('bigtheta', hp)

    def store(self, query, value, reward):
        ''' store current position as goal if reward is positive at cue indexed row '''
        if reward > 0:
            memidx = np.argmax(query)
            StepR = self.bigtheta(reward)
            self.keyvalue[memidx] = np.concatenate([query[None, :], value, np.array([[StepR]])], axis=1)

    def delete_goal(self, query, reward, done):
        ''' delete current position as goal since no reward at end of trial '''
        if done and reward == 0:
            memidx = np.argmax(query)
            self.keyvalue[memidx] = 0

    def delete_goal_dist(self, xy, goal, query, reward):
        ''' delete current goal as recalled goal no reward'''
        if np.linalg.norm(goal[0, :2] - xy, ord=2) < self.gdist and reward == 0 and goal[0, -1] > self.omite:
            memidx = np.argmax(query)
            self.keyvalue[memidx] = 0

    def recall(self, query):
        ''' attention mechanism to query memory to retrieve goal coord'''
        qk = np.dot(query, self.keyvalue[:, :self.ncues].T)  # use cue to query memory
        At = softmax(self.recallbeta * qk)  # attention weight
        value = np.dot(At, self.keyvalue[:, self.ncues:])  # goalxy
        return value.astype(np.float32)

class NumpyMotorController:
    def __init__(self, npz_path):
        w = np.load(npz_path)
        # Set up shapes: (input_dim,128), (128,128), (128,40)
        self.W1 = w['h1_w']  # (5,128)
        self.b1 = w['h1_b']  # (128,)
        self.W2 = w['h2_w']  # (128,128)
        self.b2 = w['h2_b']  # (128,)
        self.Wout = w['action_w']  # (128,40)
        self.bout = w['action_b']  # (40,)

    def relu(self, x):
        return np.maximum(0, x)
    
    def __call__(self, x):
        """Expects x of shape (batch, 5)"""
        # Ensure float32
        x = np.array(x, dtype=np.float32)
        z1 = np.dot(x, self.W1) + self.b1
        h1 = self.relu(z1)
        z2 = np.dot(h1, self.W2) + self.b2
        h2 = self.relu(z2)
        out = np.dot(h2, self.Wout) + self.bout  # (batch, 40)
        return out

    def predict(self, x):
        return self.__call__(x)

class ActorCells:
    def __init__(self, hp):
        self.nact = hp['nact']
        self.alat = hp['alat']
        self.tstep = hp['tstep']
        self.astep = hp['maxspeed'] * self.tstep
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        self.aj = (self.astep * np.array([np.sin(thetaj), np.cos(thetaj)])).astype(np.float32)
        self.alpha = self.tstep / hp['tau']
        self.qstate = np.zeros((1, self.nact))  # initialise actor units to 0
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
        self.wlat = ((wminus / self.nact) + wplus * f / norm).astype(np.float32)
        self.actact = choose_activation(hp['actact'], hp)
        self.wact = np.zeros([hp['nrnn'], hp['nact']])
        self.omite = hp['omite']
        self.rho = np.zeros((1, self.nact))
        self.alr = hp['alr']
        self.awlim = hp['awlim']
        self.contbeta = hp['contbeta']
        self.usenmc = hp['usenmc']
        self.beta = hp['mcbeta']

        if hp['usenmc']:
            # self.nmc = NumpyMotorController('../motor_controller/motor_controller_weights_128_40.npz')
            self.nmc = tf.keras.models.load_model('../motor_controller/mc_2h128_linear_30mb_31sp_0.6oe_20e_2022-10-08')
            print('Load TF motor controller')

    def reset(self):
        self.qstate = np.zeros((1, self.nact))  # reset actor units to 0
        self.rho = np.zeros((1, self.nact))

    def move(self, ract, xy, goal):
        Ihid = np.dot(ract, self.wact)

        if self.usenmc:
            # Imc = self.nmc(np.concatenate([goal, xy], axis=1))
            Imc = np.array(self.nmc(tf.concat([goal, xy], axis=1)))
        else:
            Imc = self.symbolic_motor_controller(goal=goal, xy=xy)

        I = (1-self.contbeta) * Ihid + self.contbeta * Imc  # combine model-free & schema actions
        sigq = np.random.normal(0, self.actns, size=(1, self.nact)).astype(np.float32)
        lat = np.dot(self.actact(self.qstate), self.wlat)

        self.qstate += self.alpha * (-self.qstate + I + lat + sigq)
        self.rho = self.actact(self.qstate)
        at = np.dot(self.aj, self.rho.T)[:, 0] / self.nact

        movedist = np.linalg.norm(at, 2) * 1000 / self.tstep  # m/s
        self.avgspeed.append(movedist)
        return at, self.rho

    def symbolic_motor_controller(self, goal, xy):
        ''' motor controller to decide direction to move with current position and goal location '''
        if goal[0, -1] > self.omite:  # omit goal if goal is less than threshold
            dircomp = (goal[:, :2] - xy).astype(np.float32)  # vector subtraction
            qk = np.dot(dircomp, self.aj)  # choose action closest to direction to move
            qhat = softmax(self.beta * qk)  # scale action with beta and get probability of action
        else:
            qhat = np.zeros([1, self.nact])  # if goal below threshold, no action selected by motor controller
        return qhat

    def learn_policy(self, ract, tderr):
        dwa = np.dot(ract.T, self.rho) * tderr
        self.wact += self.tstep * self.alr * dwa
        self.wact = np.clip(self.wact, self.awlim[1], self.awlim[0])


class ValueCells:
    def __init__(self, hp):
        self.tstep = hp['tstep']
        self.ncri = hp['ncri']
        self.alpha = self.tstep / hp['tau']
        self.vstate = np.zeros((1, self.ncri))  # initialise actor units to 0
        self.crins = np.sqrt(1 / self.alpha) * hp['crins']
        self.criact = choose_activation(hp['criact'], hp)
        self.wval = np.zeros([hp['nrnn'], hp['ncri']])
        self.pastvalue = np.zeros((1, self.ncri))
        self.value = np.zeros((1, self.ncri))
        self.eulerm = hp['eulerm']
        self.tderr = np.zeros((1, 1))
        self.tdalpha = self.tstep / hp['taug']
        self.clr = hp['clr']
        self.cwlim = hp['cwlim']

    def reset(self):
        self.vstate = np.zeros((1, self.ncri))  # reset actor units to 0
        self.value = np.zeros((1, self.ncri))
        self.tderr = np.zeros((1, 1))

    def est_value(self, ract):
        self.pastvalue = np.copy(self.value)
        I = np.dot(ract, self.wval)
        sigv = np.random.normal(0, self.crins, size=(1, self.ncri)).astype(np.float32)
        self.vstate += self.alpha * (-self.vstate + I + sigv)
        self.value = self.criact(self.vstate)
        return self.value

    def compute_td(self, reward):
        # forward euler method from Kumar et al. (2022) Cerebral cortex
        tderr = reward + (self.value - (1 + self.tdalpha) * self.pastvalue) / self.tstep
        self.tderr = np.reshape(tderr, (1, 1))
        return self.tderr

    def learn_value(self, ract, tderr):
        dwv = ract.T * self.tderr
        self.wval += self.tstep * self.clr * dwv
        self.wval = np.clip(self.wval, self.cwlim[1], self.cwlim[0])


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
        self.rx = np.zeros((1, self.nrnn))  # reset recurrent units to 0
        self.rfr = np.zeros((1, self.nrnn))
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
        self.rx = np.random.normal(size=(1, self.nrnn)) * np.sqrt(1 / self.alpha) * 0.1 # reset & kick network into chaotic regime
        self.rfr = self.resact(self.rx)

    def process(self, inputs):
        I = np.dot(inputs.astype(np.float32), self.win)  # get input current
        rj = np.dot(self.resrecact(self.rx), self.wrec)  # get recurrent activity
        rsig = np.random.normal(size=(1, self.nrnn)) * self.recns  # white noise

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

        self.pcs = np.zeros([self.npc * self.npc, 2])  # coordinates place cells are selective for
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
        np.random.shuffle(self.pcs)  # randomly shuffle place cell selectivity


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def choose_activation(actname, hp=None):
    if actname == 'tanh':
        act = np.tanh
    elif actname == 'relu':
        def relu(x):
            return np.maximum(0, x)
        act = relu
    elif actname == 'bump':
        def bump_activation(x):
            g01 = (x * (x > 0) * (x <= 0.5)) ** 2
            g1 = np.sqrt(2 * x - 0.5) * (x > 0.5)
            g1[np.isnan(g1)] = 0
            return g01 + g1
        act = bump_activation
    elif actname == 'phia':
        def phia(x):
            return np.maximum(0, x - hp['sparsity'])
        act = phia
    elif actname == 'softmax':
        act = softmax
    elif actname == 'retanh':
        def retanh(x):
            return np.tanh(np.maximum(0, x))
        act = retanh
    elif actname == 'exp':
        act = np.exp
    elif actname == 'bigtheta':
        def bigtheta(x):
            return 1*(x>0)
        act = bigtheta
    else:
        def no_activation(x):
            return x
        act = no_activation
    return act


class FeedForwardCells:
    def __init__(self, hp, ninput=67):
        self.nff = hp['nrnn']
        self.fact = choose_activation(hp['ract'], hp)
        self.win = np.random.uniform(-1, 1, size=(ninput, self.nff)).astype(np.float32)

    def process(self, inputs):
        x = np.dot(inputs.astype(np.float32), self.win)
        h = self.fact(x)
        return h


class FFACAgent:
    def __init__(self, hp):
        ''' Init agent '''
        self.hp = hp
        self.pc = PlaceCells(hp)
        self.coord = PositionCells(hp)
        self.target = GoalCells(hp)
        self.critic = ValueCells(hp)
        self.actor = ActorCells(hp)
        self.ff = FeedForwardCells(hp, ninput=hp['npc'] ** 2 + hp['ncues'])

    def see(self, state, cue, startbox):
        self.startbox = startbox
        cpc = self.pc.sense(state)[None, :].astype(np.float32)  # place cell activity
        cue = cue[None, :].astype(np.float32)  # sensory cue

        if self.startbox:  # if in startbox, silence place cells
            pcact = np.zeros_like(cpc)
        else:
            pcact = cpc

        I = np.concatenate([pcact, cue], axis=1).astype(np.float32)
        rfr = self.ff.process(I)
        return cpc, cue, rfr

    def estimate_value_position_goal_memory_td(self, cpc, cue, rfr):
        value = self.critic.est_value(rfr)
        xy = self.coord.est_position(cpc)
        goal = self.target.recall(rfr)

        return value, xy, goal, None

    def get_action(self, rfr, xy, goal):
        if self.startbox:  # if startbox, silence actions
            action, rho = np.zeros([2]), np.zeros([1, self.actor.nact])
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

        return tderr, tdxy

    def state_reset(self):
        self.coord.reset()
        self.critic.reset()
        self.actor.reset()
        self.target.reset()

    def get_weights(self):
        win = self.ff.win
        wcoord = self.coord.wxy
        wgoal = self.target.wgoal
        wact = self.actor.wact
        wcri = self.critic.wval

        return [wcoord, wcri, wact, win, wgoal]

    def set_weights(self, mdlw):
        self.coord.wxy = mdlw[0]
        self.critic.wval = mdlw[1]
        self.actor.wact = mdlw[2]
        self.ff.win = mdlw[3]
        self.target.wgoal = mdlw[4]