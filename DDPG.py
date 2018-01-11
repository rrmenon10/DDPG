import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque

class ReplayBuffer():

	def __init__(self, max_samples):
		self.buffer_size = max_samples
		self.buffer = deque(maxlen=max_samples)

	def store_sample(self, s_t, a_t, r_t, s_t_1):

		transition = (s_t, a_t, r_t, s_t_1)
		self.buffer.append(transition)

	def get_sample(self, minibatch_size):

		minibatch_size = min(minibatch_size, self.size)
		samples = random.sample(self.buffer, minibatch_size)
		return zip(*samples)

	def _clear_buffer(self):
		self.buffer.clear()

	@property
	def size(self):
		return len(self.buffer)

class OrnsteinUhlenbeckNoise:

	def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.reset()

	def noise_sample(self):
		self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		return self.x

	def reset(self):
	    self.x = np.zeros_like(self.mu)

class Actor:

	def __init__(self, action_dim, state_dim, action_bound, actor_lr, batch_size, scope):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self._scope = scope
		
		self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name= "State")

		with tf.variable_scope(self._scope):
			output = layers.fully_connected(self.state, num_outputs=400, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output = layers.fully_connected(output, num_outputs=300, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output = layers.fully_connected(output, num_outputs=self.action_dim, activation_fn=tf.nn.tanh, weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
		
		self._prediction = tf.multiply(output, action_bound)

		self.critic_gradient = tf.placeholder(tf.float32, shape=[None, self.action_dim])
		self.actor_gradient = tf.gradients(self.prediction, self.vars, -self.critic_gradient)
		self.actor_gradient = list(map(lambda x : tf.div(x, batch_size), self.actor_gradient))
		self._optimize = tf.train.AdamOptimizer(learning_rate=actor_lr).apply_gradients(zip(self.actor_gradient, self.vars))

	@property
	def optimize(self):
		return self._optimize

	@property
	def prediction(self):
		return self._prediction

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)

class Critic:

	def __init__(self, action_dim, state_dim, critic_lr, scope):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self._scope = scope

		self.state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name= "State")
		self.action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="Action")

		with tf.variable_scope(self._scope):
			state_feat = layers.fully_connected(self.state, num_outputs=400, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			state_action = tf.concat([state_feat, self.action], axis=-1)
			output = layers.fully_connected(state_action, num_outputs=300, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output = layers.fully_connected(output, num_outputs=1, activation_fn=None, weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
		
		self._prediction = output

		if "Target" not in self._scope:
			self.target_output = tf.placeholder(tf.float32, shape=[None, 1], name="Target_Q")
			self._loss = tf.reduce_mean(tf.pow(self.prediction-self.target_output, 2))
			self._optimize = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.loss)
		else:
			self._loss = None

		self._action_gradient = tf.gradients(self.prediction, self.action)

	@property
	def optimize(self):
		return self._optimize

	@property
	def loss(self):
		return self._loss

	@property
	def action_grad(self):
		return self._action_gradient

	@property
	def prediction(self):
		return self._prediction

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)

class DDPG:

	def __init__(self, params, sess):
		self.params = params
		self.explore = True
		self.sess = sess
		self.actor = Actor(self.params.action_dims, self.params.state_dims, self.params.action_bound, self.params.actor_lr, self.params.minibatch_size, scope="Actor_Online")
		self.critic = Critic(self.params.action_dims, self.params.state_dims, self.params.critic_lr, scope="Critic_Online")

		self.actor_target = Actor(self.params.action_dims, self.params.state_dims, self.params.action_bound, self.params.actor_lr, self.params.minibatch_size, scope="Actor_Target")
		self.critic_target = Critic(self.params.action_dims, self.params.state_dims, self.params.critic_lr, scope="Critic_Target")


		self.update_target_ops = []
		for target_var, online_var in zip(self.actor_target.vars, self.actor.vars):
			self.update_target_ops.append(tf.assign(target_var, tf.multiply(target_var, 1.-self.params.tau) + tf.multiply(online_var, self.params.tau)))
		for target_var, online_var in zip(self.critic_target.vars, self.critic.vars):
			self.update_target_ops.append(tf.assign(target_var, tf.multiply(target_var, 1.-self.params.tau) + tf.multiply(online_var, self.params.tau)))

		self.buffer = ReplayBuffer(self.params.max_samples)
		self.noise = OrnsteinUhlenbeckNoise(self.params.mu, self.params.sigma, self.params.theta, self.params.dt)

		saver = tf.train.Saver()

	def sample_action(self, state):

		action = self.sess.run(self.actor.prediction, feed_dict={self.actor.state : state})
		if self.explore:
			action = action + self.noise.noise_sample()
		return action

	def update_target(self):
		self.sess.run(self.update_target_ops)

	def train_batch(self):
		s_t, a_t, r_t, s_tp1 = self.buffer.get_sample(self.params.minibatch_size)

		a_pred_tp1 = self.sess.run(self.actor_target.prediction, feed_dict={self.actor_target.state : s_tp1})
		q_pred_tp1 = self.sess.run(self.critic_target.prediction, feed_dict={self.critic_target.state : s_tp1, self.critic_target.action : a_pred_tp1})
		r_t += self.params.gamma * q_pred_tp1.reshape([self.params.minibatch_size, ])
		r_t = r_t.reshape([self.params.minibatch_size, 1])

		q_pred_t = self.sess.run(self.critic.prediction, feed_dict={self.critic.state : s_t, self.critic.action : a_t})

		loss = self.sess.run(self.critic.loss, feed_dict={self.critic.target_output : r_t, self.critic.state : s_t, self.critic.action : a_t})
		self.sess.run(self.critic.optimize, feed_dict={self.critic.target_output : r_t, self.critic.state : s_t, self.critic.action : a_t})

		a_pred_t = self.sess.run(self.actor.prediction, feed_dict={self.actor.state : s_t})
		q_pred_t = self.sess.run(self.critic.prediction, feed_dict={self.critic.state : s_t, self.critic.action : a_pred_t})
		action_grad = self.sess.run(self.critic.action_grad, feed_dict={self.critic.state : s_t, self.critic.action : a_t})
		action_grad = np.array(action_grad).reshape([-1, 1])
		self.sess.run(self.actor.optimize, feed_dict={self.actor.state : s_t, self.actor.critic_gradient : action_grad})

		self.update_target()

	def save_model(self, sess):

		save_path = saver.save(sess, "./models/ddpg_model.ckpt")
		print("Model saved in file: %s" % save_path)