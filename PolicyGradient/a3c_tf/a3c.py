from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    Given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal,
                 features)

Batch = namedtuple('Batch', ['si', 'a', 'adv', 'r', 'terminal', 'features'])

class PartialRollout(object):
    """
    A piece of a complete rollout. We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

def env_runner(env, policy, num_local_steps, summary_writer, render):
    """
    The logic of the thread runner. In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    timestep_limit = env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')
    semantics_autoreset = env.metadata.get('semantics.autoreset')
    length = 0
    rewards = 0
    running_mean = None
    episode = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal,
                        last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            if terminal or length >= timestep_limit:
                running_mean = rewards if running_mean is None else 0.99 * running_mean + 0.01 * rewards
                summary = tf.Summary()
                terminal_end = True
                if length >= timestep_limit or not semantics_autoreset:
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode {} finished. Sum of rewards: {} Length: {} Running Mean {:.3f}".format(episode, rewards, length, running_mean))
                summary.value.add(tag="running_mean", simple_value=float(running_mean))
                summary_writer.add_summary(summary, episode)
                summary_writer.flush()
                length = 0
                rewards = 0
                episode += 1
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, env, task, visualise):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned
        for the VNC environments. Below, we will have a modest amount of
        complexity due to the way TensorFlow handles data parallelism. But
        overall, we'll define the model, specify its inputs, and describe how
        the policy gradients step should be computed.
        """

        self.env = env
        self.task = task
        self.visualise = visualise
        obs_shape = env.observation_space.shape
        worker_device = '/job:worker/task:{}/cpu:0'.format(task)
        with tf.device(tf.train.replica_device_setter(1,
                worker_device=worker_device)):
            with tf.variable_scope('global'):
                self.network = LSTMPolicy(obs_shape,
                                          env.action_space.n)
                self.global_step = tf.get_variable('global_step', [], tf.int32,
                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                        trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(obs_shape,
                                                     env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss: its derivative is precisely the
            # policy gradient notice that self.ac is a placeholder that is
            # provided externally. adv will contain the advantages, as
            # calculated in process_rollout
            self.pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1])
                                      * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = self.pi_loss + 0.5 * vf_loss - entropy * 0.01

            grads = tf.gradients(self.loss, pi.var_list)

            tf.summary.scalar("model/policy_loss", self.pi_loss)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in
                                   zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars),
                                     inc_step)
            self.summary_writer = None
            self.local_steps = 0


    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer
        self.rollout_provider = env_runner(self.env,
                                           self.local_network,
                                           20, #TODO: Move to args
                                           self.summary_writer,
                                           self.visualise)

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner, and
        updates the parameters. The update is then sent to the parameter
        server.
        """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = next(self.rollout_provider)
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]),
                                                                  fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
