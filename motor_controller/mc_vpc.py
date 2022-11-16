import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from backend.model import PlaceCells
from backend.utils import get_default_hp, saveload
import datetime

hp = get_default_hp(task='6pa',platform='laptop')

#gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)


pc = PlaceCells(hp)

nhid = 128
nact = 40
beta = 25
lr = 0.001  #0.001
res = 31
epochs = 20
omitg = 0.6
modelname = 'mc_2h{}_linear_{}mb_{}sp_{}oe_{}e_{}'.format(nhid,beta, res, omitg,epochs, str(datetime.date.today()))
print(modelname)

checkpoint_path = '{}/cp.ckpt'.format(modelname)
checkpoint_dir = os.path.dirname(checkpoint_path)

pos = np.linspace(-1,1,res)
xx, yy = np.meshgrid(pos, pos)
g = np.concatenate([xx.reshape([res**2,1]),yy.reshape([res**2,1])],axis=1)
xy = np.copy(g)
x = []
dircomp = []
nogidx = []
i=0
for goal in range(res**2):
    for curr in range(res**2):
        gns = g[goal]
        xyns = xy[curr]

        x.append(np.concatenate([gns, xyns]))

        dircomp.append(gns - xyns)
        i+=1

x = np.array(x)
dircomp = np.array(dircomp)

thetaj = (2 * np.pi * np.arange(1, nact + 1)) / nact
akeys = np.array([np.sin(thetaj), np.cos(thetaj)])
qk = tf.matmul(tf.cast(dircomp,dtype=tf.float32),tf.cast(akeys,dtype=tf.float32))
q = np.array(tf.nn.softmax(beta * qk))
print('{} Data created . . . '.format(q.shape[0]))

# activate motor controller
allinputs = []
alloutputs = []
btstp = 5
for b in range(btstp):
    supeps = np.random.uniform(0, omitg, len(q))  # suppress schema if confidience below threhold omitg
    acteps = np.random.uniform(omitg, 1.25, len(q))  # activate schema if confi above threshold
    alleps = np.concatenate([supeps, acteps])
    np.random.shuffle(alleps)

    alleps = alleps[:len(q)][:,None]
    g = x[:,:2]
    xy = x[:,2:]
    u = np.concatenate([g,alleps,xy],axis=1)

    sidx = alleps > omitg
    z = q*sidx

    allinputs.append(u)
    alloutputs.append(z)

    print(b)

allinputs = np.vstack(allinputs)
alloutputs = np.vstack(alloutputs)
print('Randomised {} data . . . '.format(alloutputs.shape[0]))

tslen = int(len(allinputs) * 0.1)
s = np.arange(len(allinputs))
np.random.shuffle(s)
trainx = allinputs[s][tslen:]
trainy = alloutputs[s][tslen:]
testx = allinputs[s][:tslen]
testy = alloutputs[s][:tslen]

print('Train {}, Test {}'.format(trainx.shape[0],testx.shape[0]))

''' model definition x --> q'''

class motor_controller(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h1 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=True, kernel_initializer='glorot_uniform', name='h1')
        self.h2 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=True, kernel_initializer='glorot_uniform', name='h2')
        self.action = tf.keras.layers.Dense(units=nact, activation='linear',
                                                use_bias=True, kernel_initializer='zeros', name='action')


    def call(self, x):
        a = self.action(self.h2(self.h1(x)))
        #a = self.action(self.h1(x))
        return a


model = motor_controller()

#model.load_weights(checkpoint_path)
#ls, acc = model.evaluate(allinputs[randidx], alloutputs[randidx], verbose=2)

loss = tf.keras.losses.mean_squared_error
model.compile(run_eagerly=False,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=loss, metrics=['accuracy'])

batch_size = 64
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(trainx, trainy, validation_data=(testx,testy),epochs=epochs, batch_size=batch_size,
                    validation_split=0.0, shuffle=True, callbacks=[cp_callback])
model.summary()

qpred = model.predict_on_batch(testx[:23])

plt.figure()
plt.subplot(461)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mse loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

for i in range(23):
    plt.subplot(4,6,i+2)
    plt.plot(testy[i])
    plt.plot(qpred[i])

plt.show()

if history.history['val_loss'][-1] < 1e-5:
    model.save(modelname)
    plt.savefig(modelname+'.png')
    saveload('save','var_'+modelname,[history.history])








