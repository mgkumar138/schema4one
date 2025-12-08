#%%
import tensorflow as tf
import numpy as np

# Adjust this to your real model class if not in same file:
class motor_controller(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h1 = tf.keras.layers.Dense(units=128, activation='relu', name='h1')
        self.h2 = tf.keras.layers.Dense(units=128, activation='relu', name='h2')
        self.action = tf.keras.layers.Dense(units=40, activation='linear', name='action')
    def call(self, x):
        a = self.action(self.h2(self.h1(x)))
        return a

model = tf.keras.models.load_model('/n/home04/mgkumar/schema4one/motor_controller/mc_2h128_linear_30mb_31sp_0.6oe_20e_2022-10-08')
weights_dict = {}
for layer in model.layers:
    if hasattr(layer, 'get_weights'):
        w = layer.get_weights()
        if w:  # skip empty
            weights_dict[layer.name+'_w'] = w[0]
            weights_dict[layer.name+'_b'] = w[1]

np.savez('motor_controller_weights_128_40.npz', **weights_dict)
print('Weights exported!')

