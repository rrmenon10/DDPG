import numpy as np
import tensorflow as tf
import gym
from DDPG import DDPG


def main(sess, env, params):

	ddpg = DDPG(params, sess)

	sess.run(tf.global_variables_initializer())

	ddpg.update_target()

	for i in range(params.max_episodes):

		s = env.reset()
		ep_reward = 0

		for j in range(params.max_episode_length):

			a = ddpg.sample_action(s.reshape([1, params.state_dims]))
			s2, r, terminal, info = env.step(a[0])

			ddpg.buffer.store_sample(s.reshape([params.state_dims, ]), a.reshape([params.action_dims, ]), r, s2.reshape([params.state_dims, ]))
			s = s2
			ep_reward += r


			if ddpg.buffer.size > params.minibatch_size:
				ddpg.train_batch()

			if terminal:
				break

		print ep_reward




if __name__== '__main__':
	flags = tf.app.flags

	flags.DEFINE_float("actor_lr", 1e-4, "Learning rate for Actor network")
	flags.DEFINE_float("critic_lr", 1e-3, "Learning rate for Critic network")
	flags.DEFINE_float("gamma", 0.99, "Discount Factor")
	flags.DEFINE_float("tau", 0.01, "Soft update parameter")
	flags.DEFINE_float("sigma", 0.3, "Sigma for Noise")
	flags.DEFINE_float("theta", 0.15, "Theta for Noise")
	flags.DEFINE_float("dt", 1e-2, "dt for Noise")
	flags.DEFINE_integer("max_samples", 1e6, "Maximum number of samples in replay memory")
	flags.DEFINE_integer("minibatch_size", 64, "Minibatch size for training")
	flags.DEFINE_string("env", "Pendulum-v0", "Environment")
	flags.DEFINE_integer("max_episodes", 50000, "Maximum number of episodes to be played")
	flags.DEFINE_integer("max_episode_length", 200, "Maximum number of steps in each episode")

	params = flags.FLAGS

	sess = tf.Session()

	env = gym.make(params.env)
	setattr(params, 'state_dims', env.observation_space.shape[0])
	setattr(params, 'action_dims', env.action_space.shape[0])
	setattr(params, 'action_bound', env.action_space.high)
	setattr(params, 'mu', np.zeros(params.action_dims))

	main(sess, env, params)
