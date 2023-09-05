# One-shot learning of multiple new paired associations using biologically plausible schemas

This repository contains the code to obtain the results described in the [paper](https://arxiv.org/abs/2106.03580)  (Instructions to run the code is given below).
      
The main outcome of the paper is to propose a biologically plausible reinforcement learning model that demonstrates one-shot learning
of multiple flavour cue - goal location paired associations, like the rodent experiment (Tse et al. 2007). 
The task comprises of a continuous square arena in which agents start from either the north, south, east or west and have to 
navigate to the goal location that is associated to the flavour cue given during that trial. Agents are only rewarded
when they reach the correct goal location.  

Agents were evaluated on their learning abilities on 5 tasks:
- **dmp**: Navigation to a single goal that is displaced to a new location every 4 trials
- **1pa**: Navigation past a U-shaped obstacle to a single goal in the center of the arena
- **6pa**: Navigation to 6 flavour-location paired association (PA) over 17 sessions followed by navigation to 2, 6 or 12 new paired associations for 1 session 
- **obs**: Navigation past obstacles to 6 PAs over 47 session followed by navigation to 2 or 6 new PAs for 1 session 
- **wkm**: Navigation to 6 PAs over 17 sessions followed by navigation to 2 or 6 new PAs but flavour cue is given only at the start of the trial followed by distractors

2 types of agents were evaluated in all tasks and script ends with the following nomenclature:
- Symbolic schema agent                                                        (**_sym**)
- Reservoir based neural agent                              (**_res**)

Both types of agents have both the schema networks and the actor-critic networks. A hyperparameter hp['contbeta'] controls
the proportion of contribution between the schema networks (hp['contbeta'] = 1) and the actor-critic network (hp['contbeta'] = 0)

### Installation
The code was created and tested on 
- OS: Windows 10, Windows 11
- Python version: Python 3.7

To install the necessary dependencies, create a virtual environment using either venv or a new environment in conda (use pip in conda env) and run

```setup
pip install -r requirements.txt
```

### Training details

Since the outcome of the paper is to demonstrate one-shot learning, there are no pretrained models except for the neural motor controller. 
The learning potential of each agent can be observed by running the respective scripts.
Training for each agent takes about 45 minutes for single reward task and 90 minutes for multiple paired association and 12PA task depending on the hardware specifications. 
A GPU is not needed as the models are relatively small. 

To use one processor to run X number of agents in sequence, set hp['platform'] = 'laptop' and hp['btstp'] = X.
To run multiple agents in parallel, set hp['platform'] = 'server' and specify the number of processors available. 

Hyperparameters can be found in get_default_hp function in ./backend_scripts/utils.py. 
Specific hyperparameters can be found in each *.py script.

E.g. if you would want the dmp_res.py agent to use the symbolic or neural motor controller, set hp['usesmc'] to True or False respectively.

Each code generates figures that is saved to the Fig directory in working directory. 

## Experiments
### Associating multiple cues to coordinates
To train the networks to associate cues to coordinates, ensure working directory is /schema4one.

To associate 200 cues, set hp['ncues'] = 200. 
To use the Exploratory Hebbian rule, set hp['stochlearn'] = True and to use the Perceptron rule, set hp['stochlearn'] = False.
To change the size of the association network, set hp['nrnn'] with the number of units to be in the layer.
To change the activation function of the association layer, set hp['ract'] = 'relu' or 'phia' or 'tanh' etc.


To use the association network with a feedforward layer
```train
python assoc/ff_cue_coord.py
```
To use the association network with a reservoir
```train
python assoc/res_cue_coord.py
```

To learn and unlearn the association using reward modulation and acetylcholine modulation using a network with a feedforward layer
```train
python assoc/ff_learn_unlearn.py
```
To learn and unlearn the association using reward modulation and acetylcholine modulation using a network with a reservoir
```train
python assoc/res_learn_unlearn.py
```


### Single displaced location task

To run each agent described in the paper in the single displaced location task, ensure working directory is /schema4one.

To use either the Actor-Critic or the Symbolic schema agent, set hp['contbeta'] = 0 or 1 respectively
```train
python dmp/dmp_sym.py
```
To use only the Neural schema agent, set hp['contbeta'] = 1:
```train
python dmp/dmp_res.py
```

### Single goal with obstacle task

To run each agent described in the paper in the single goal with obstacle task, ensure working directory is /schema4one.

To use either the Actor-Critic, Symbolic, or hybrid Actor-Critic-Symbolic agent, set hp['contbeta'] = 0 or 1 or 0.3 respectively:
```train
python 1pa/center_sym.py
```
To use either the Neural schema or the hybrid Actor-Critic-Neural agent, set hp['contbeta'] = 1 or 0.3 respectively:
```train
python 1pa/center_res.py
```

### Multiple paired association task: 20 sessions of OPA, 2 sessions of OPA, 2NPA, 6NPA, 20 sessions of NM, 2 sessions of 6NPANM 

To run each agent in the multiple paired association task, ensure working directory is /schema4one.

To use either the Actor-Critic or the Symbolic schema agent, set hp['contbeta'] = 0 or 1 respectively
```train
python 6pa/6pa_sym.py
```
To use only the Neural schema agent, set hp['contbeta'] = 1:
```train
python 6pa/6pa_res.py
```

### One-shot learning of 12 novel paired associates 
To run each agent in the 12NPA task, ensure working directory is /schema4one.

To use either the Actor-Critic or the Symbolic schema agent, set hp['contbeta'] = 0 or 1 respectively
```train
python 12pa/12pa_sym.py
```

To use the Exploratory Hebbian rule, set hp['stochlearn'] = True and to use the Perceptron rule, set hp['stochlearn'] = False.
To change the size of the association network, set hp['nrnn'] with the number of units to be in the layer.

To use the Neural schema agent with a feedforward layer based association network 
```train
python 12pa/12pa_ff.py
```
To use the Neural schema agent with a reservoir based association network
```train
python 12pa/12pa_res.py
```

### One-shot learning of 2NPA and 6NPA while navigating past obstacles

To run each agent described in the paper in the multiple paired association task with obstacle, ensure working directory is /schema4one.

To use either the Actor-Critic, Symbolic, or hybrid Actor-Critic-Symbolic agent, set hp['contbeta'] = 0 or 1 or 0.4 respectively:
```train
python obs/obs_sym.py
```
To use either the Neural schema or the hybrid Actor-Critic-Neural agent, set hp['contbeta'] = 1 or 0.4 respectively:
```train
python obs/obs_res.py
```

### Learning to gate working memory for one-shot learning

To run each agent described in the paper in multiple paired association task with transient cue and multiple distractors, ensure working directory is /schema4one.

To introduce none, one or two distractors during a trial, set hp['ndistract'] = 0 or 1 or 2. The number of distractors can be increased to increase task difficulty. 
To set the frequency (Hz) of distractor presentation, change hp['distfreq'] where default is 0.2.
To change the learning rate for the gating mechanism, set hp['mlr'] where default is 0.0001.

To train the hybrid Actor-Critic-Neural agent, set hp['contbeta'] = 0.7-0.9 where default is 0.8
```train
python wkm/wkm_da_dist_res.py
```


## Results

Our agents achieve the following performance for single displaced location task :

- Savings in latency by all agents:

![Latency_1pa](https://user-images.githubusercontent.com/35286288/120445898-a76a6300-c3bb-11eb-8dd8-50068163b657.png)


Our agents achieve the following performance when learning the multiple paired association task :
- Latency reached by all agents:

![Latency_6pa](https://user-images.githubusercontent.com/35286288/120445947-b224f800-c3bb-11eb-88a8-239e2e325099.png)

- Average visit ratio at during each probe session:

![Dgr_train_6pa](https://user-images.githubusercontent.com/35286288/120445966-b94c0600-c3bb-11eb-9c6c-6a676cf70c4d.png)

- One shot learning results obtained for session 22 (OPA), 24 (2NPA), 26 (6NPA), 28 (NM)

![Dgr_eval_6pa](https://user-images.githubusercontent.com/35286288/120445974-bcdf8d00-c3bb-11eb-9159-abe9d18fc23d.png)

- One shot learning results for 12 random paired associations with varying size

![PI_12pa_se](https://user-images.githubusercontent.com/35286288/120446029-c79a2200-c3bb-11eb-8f2d-b782f1a727ce.png)

Our agents achieve the following one-shot learning performance when learning the multiple paired association task with obstacles:

- One shot learning results obtained for session 52 (OPA), 54 (2NPA), 56 (6NPA)


Our agents achieve the following one-shot learning performance when learning the multiple paired association task with transient cue and multiple distractors:

- One shot learning results obtained for session 22 (OPA), 24 (2NPA), 26 (6NPA), 28 (NM)



## References
If you have questions about the work or code, please drop me an [email](m_ganeshkumar@u.nus.edu) or visit [website](https://mgkumar138.github.io/).
Please cite the relevant work if the code is used for academic purposes. 

```citation
@article{kumar2023oneshot,
  title={One-shot learning of paired association navigation with biologically plausible schemas},
  author={Kumar, M Ganesh and Tan, Cheston and Libedinsky, Camilo and Yen, Shih-Cheng and Tan, Andrew Yong-Yi},
  journal={arXiv preprint arXiv:2106.03580},
  year={2023}
}
```
