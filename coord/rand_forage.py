import numpy as np
import matplotlib.pyplot as plt
from backend.model import ResACAgent#, #GridCells
from backend.utils import get_default_hp, savefig, saveload
from backend.maze import Static_Maze
import multiprocessing as mp
from functools import partial
from scipy import ndimage, stats
import matplotlib as mpl


def random_forage(agent,env, tidx):
    #[agent,env] = args
    sxytrue = []
    sxyest = []
    sxyerr = []
    swxy = []
    alltd = []
    print('Random Foraging started ...')
    for t in range(env.hp['trsess']):
        agent.state_reset()
        state, cue, reward, done = env.reset(trial=t)
        txytrue = []
        txyest = []
        txyerr = []
        px = np.zeros(2)
        std = []

        while not done:
            # if t in tidx:
            #     plasticitiy = False
            # else:
            #     plasticitiy = True

            cpc, cue, rfr = agent.see(state=state, cue=cue, startbox=env.startbox)

            value, xy, goal, mem = agent.estimate_value_position_goal_memory_td(cpc=cpc,cue=cue,rfr=rfr)

            tderr, tdxy = agent.learn(reward=reward, self_motion=env.dtxy, cpc=cpc, cue=cue, rfr=rfr, xy=xy, goal=goal, plasticity=True)

            action, rho = agent.get_action(rfr=rfr, xy=xy, goal=goal)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # path integrate with self motion alone
            px += env.dtxy

            # store estimate
            txytrue.append(state)
            txyest.append(xy)
            txyerr.append(np.linalg.norm(state-xy))
            std.append(tdxy)

            if done:
                # if t in env.nort:
                #     saveplot = True
                # else:
                #     saveplot = False
                # env.render(saveplot)
                break

        alltd.append(np.mean(np.mean(np.vstack(std) ** 2, axis=0)))
        #alltd.append(np.linalg.norm(np.vstack(std),axis=1))
        sxyerr.append(np.vstack(txyerr)[:env.maxstep])
        if t in tidx:
            sxytrue.append(np.vstack(txytrue))
            sxyest.append(np.vstack(txyest))
            swxy.append(agent.coord.wxy)
        print('Trial {} | Steps {} | Start {} | End {} | dW {:.3g} | TD {:.3g}'.format(
            t+1, env.i, env.startpos, np.round(state,1), np.max(agent.coord.wxy), alltd[-1]))
    return np.array(sxytrue), np.array(sxyest), np.array(sxyerr), np.array(swxy), np.array(alltd)


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='server')
    hp['xylr'] = 0.01
    hp['alr'] = 0
    hp['clr'] = 0
    hp['glr'] = 0
    hp['nrnn'] = 2
    hp['tstep'] = 20
    hp['xytau'] = 200
    hp['shufflepc'] = False
    hp['trsess'] = 20
    hp['time'] = 30
    hp['probetime'] = 10
    hp['obs'] = False
    hp['obstype'] = 'none'  # nil. cave, center, split
    hp['render'] = False

    tidx = [0,4,19]

    # start training
    btstp = 1
    #alltau = [100,200,400,750,1000] #np.logspace(2,3,num=5)
    alltau = [200]

    for xytau in alltau:
        hp['xytau'] = xytau

        for b in range(btstp):
            # set agent
            agent = ResACAgent(hp=hp)
            # set task
            env = Static_Maze(hp)
            env.make('square', noreward=None, rloc=None)

            xytrue, xyestimate, xyerror, xyweights, td = random_forage(agent,env, tidx)
            # saveload('save', './Data/vars_learncoord_{}tau_{}'.format(hp['xytau'], b+10),
            #          [xytrue, xyestimate, xyerror, xyweights, td])

            # plot
            f, ax = plt.subplots(nrows=4, ncols=3,figsize=(5,5))
            #f.text(0.01, 0.01, exptname)
            sttraj = 1 * 1000 //hp['tstep']
            endtraj = hp['probetime'] * 1000 //hp['tstep']
            err = np.mean(xyerror[tidx][:,:,0],axis=1)
            for i in range(3):
                ax[2,i].plot(xytrue[i, sttraj:endtraj,0], xytrue[i, sttraj:endtraj,1], color='k')
                ax[2, i].scatter(xytrue[i, sttraj, 0], xytrue[i, sttraj, 1], color='k', marker='s', s=20)
                ax[2, i].scatter(xytrue[i, endtraj, 0], xytrue[i, endtraj, 1], color='k', marker='x',s=20)
                ax[2, i].spines.right.set_visible(False)
                ax[2, i].spines.top.set_visible(False)
                #ax[2,i].set_aspect('equal', adjustable='box')

                ax[3, i].plot(xyestimate[i, sttraj:endtraj, 0], xyestimate[i, sttraj:endtraj, 1],color='r')
                ax[3, i].scatter(xyestimate[i, sttraj, 0], xyestimate[i, sttraj, 1], color='r', marker='s', s=20)
                ax[3, i].scatter(xyestimate[i, endtraj, 0], xyestimate[i, endtraj, 1], color='r', marker='x',s=20)
                ax[3, i].spines.right.set_visible(False)
                ax[3, i].spines.top.set_visible(False)
                #ax[3, i].set_aspect('equal', adjustable='box')
                #ax[3, i].set_title('Δ={:.1g}'.format(err[i]), fontsize=8)

                im = ax[0,i].imshow(xyweights[i][:,0].reshape(hp['npc'],hp['npc']))
                cbar = plt.colorbar(im,ax=ax[0,i])
                cbar.ax.tick_params(labelsize=8)
                im2 = ax[1, i].imshow(xyweights[i][:, 1].reshape(hp['npc'], hp['npc']))
                cbar = plt.colorbar(im2,ax=ax[1,i])
                cbar.ax.tick_params(labelsize=8)

                ax[0,i].set_title('Trial '+ str(tidx[i]+1))
                if i == 0:
                    ax[2,0].set_ylabel('True')
                    ax[3, 0].set_ylabel('Estimate')
                    ax[0, 0].set_ylabel('$W^{coord}$ to X')
                ax[1, 0].set_ylabel('$W^{coord}$ to Y')
            for i in range(3):
                for j in range(4):
                    ax[j, i].set_xticks([])
                    ax[j, i].set_yticks([])
            f.tight_layout()

            savefig('./Fig/true_est_coord_{}tau_{}obs_{}t_b{}_10'.format(hp['xytau'], hp['obstype'], hp['time'],btstp), f)
            # plt.close('all')

    # if btstp >1:
    #     N = 20
    #     alltau = [20,100,200,1000]
    #     col = ['k','b','r','g']
    #     alltd = np.zeros([len(alltau),N, 20])
    #     import glob
    #     for i,tau in enumerate(alltau):
    #         allvars = glob.glob('./Data/vars_learncoord_{}tau*'.format(tau))
    #
    #         if len(allvars) !=0:
    #             for j in range(N):
    #                 [xytrue, xyestimate, xyerror, xyweights, td] = saveload('load',allvars[j],1)
    #
    #                 alltd[i,j] = td
    #
    #     f = plt.figure(figsize=(6,4))
    #     ax = plt.subplot(111)
    #     for i in range(4):
    #         mtd = np.mean(alltd[i],axis=0)
    #         std = np.std(alltd[i],axis=0)/np.sqrt(N)
    #         ax.plot(np.arange(hp['trsess']), mtd, color=col[i], marker='o',ms=5)
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 1, box.height])
    #     ax.legend(['20ms','100ms','200ms','1000ms'],frameon=False,fontsize=8, loc='center left',bbox_to_anchor=(1, 0.5))
    #     for i in range(4):
    #         mtd = np.mean(alltd[i],axis=0)
    #         std = np.std(alltd[i],axis=0)/np.sqrt(N)
    #         ax.fill_between(x=np.arange(hp['trsess']), y1=mtd-std, y2=mtd+std, alpha=0.2, color=col[i])
    #
    #     ax.set_xlabel('Trial')
    #     ax.set_ylabel('Average TD MSE')
    #     ax.set_title('Path integration TD error')
    #     ax.set_xticks(np.linspace(1,20,5,dtype=int),np.linspace(1,20,5,dtype=int))
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #     ax.spines.right.set_visible(False)
    #     ax.spines.top.set_visible(False)
    #     f.tight_layout()
    #     savefig('./Fig/all_tau_coord',f)
    #



