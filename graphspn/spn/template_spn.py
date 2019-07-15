import sys
from abc import abstractmethod
import tensorflow as tf
import libspn as spn
import numpy as np
from numpy import float32
import random
import copy
import os

from graphspn.spn.spn_model import SpnModel, mod_compute_graph_up
from graphspn.graphs.template import NodeTemplate


class TemplateSpn(SpnModel):

    def __init__(self, template, *args, **kwargs):
        """
        Initialize an TemplateSpn.

        template (Template): a class of the modeled template.

        **kwargs
           seed (int): seed for the random generator. Set before generating
                       the structure.
           num_vals (int): number of different values a variable (on a node) can take;
                       the values are discretized from 0 to num_vals-1.

        """
        super().__init__(*args, **kwargs)

        self._template = template
        self._dense_gen = spn.DenseSPNGeneratorLayerNodes(num_decomps=self._num_decomps, num_subsets=self._num_subsets,
                                                          num_mixtures=self._num_mixtures, input_dist=self._input_dist,
                                                          num_input_mixtures=self._num_input_mixtures)
        self._expanded = False
        self._rnd = random.Random()
        self._seed = kwargs.get('seed', None)
        self._saved_path = None
        

    @property
    def template(self):
        return self._template


    @property
    def expanded(self):
        return self._expanded


    @property
    @abstractmethod
    def root(self):
        pass

    @property
    def dense_gen(self):
        return self._dense_gen

    
    @property
    def rnd(self):
        return self._rnd

    
    @property
    def seed(self):
        return self._seed

    
    @property
    def num_nodes(self):
        """
        Number of semantic variables (i.e. number of nodes represented by the modeled template).
        
        Note: for edge templates, num_nodes equals to the number of covered edges times 2, because
              each edge has two nodes.
        """
        return self._num_nodes

    def generate_random_weights(self, trainable=True):
        """
        Generates random weights for this spn.
        """
        weight_init_value = spn.ValueType.RANDOM_UNIFORM(self._weight_init_min, self._weight_init_max)
        spn.generate_weights(self._root, init_value=weight_init_value, trainable=trainable)
        

    def init_weights_ops(self):
        print("Generating weight initialization Ops...")
        init_weights = spn.initialize_weights(self._root)
        self._initialize_weights = init_weights


    def init_learning_ops(self):
        print("Initializing learning Ops...")
        if self._learning_algorithm == spn.GDLearning:
            learning = spn.GDLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      learning_rate=self._learning_rate,
                                      learning_type=self._learning_type,
                                      learning_inference_type=self._learning_inference_type,
                                      use_unweighted=True)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.learn(optimizer=self._optimizer)
            
        elif self._learning_algorithm == spn.EMLearning:
            learning = spn.EMLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      additive_smoothing=self._additive_smoothing_var,
                                      use_unweighted=True,
                                      initial_accum_value=self._init_accum)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.accumulate_updates()
            self._update_spn = learning.update_spn()

        self._log_likelihood_root = learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._log_likelihood_root)

        # ops for inference
        

    def initialize_weights(self, sess):
        print("Initializing weights...")
        sess.run(self._initialize_weights)


    def expand(self):
        """
        Expand input to include likelihoods for semantics.

        Do nothing if already expanded.
        """
        if not self._expanded:
            print("Expanding...")
            
            self._likelihood_inputs = spn.RawInput(num_vars=self._num_nodes * self._num_vals,
                                                   name=self.vn['LH_CONT'])
            self._semantic_inputs = spn.IVs(num_vars=self._num_nodes, num_vals=self._num_vals,
                                            name=self.vn['SEMAN_IVS'])
            prods = []
            for i in range(self._num_nodes):
                for j in range(self._num_vals):
                    prod = spn.Product(
                        (self._likelihood_inputs, [i*self._num_vals + j]),
                        (self._semantic_inputs, [i*self._num_vals + j])
                    )
                    prods.append(prod)
            self._conc_inputs.set_inputs(*map(spn.Input.as_input, prods))
            self._expanded = True

    def _compute_mle_loss(self, samples, func_feed_samples, dgsm_lh=None, batch_size=100):
        batch = 0
        likelihood_values = []
        stop = min(batch_size, len(samples))
        while stop < len(samples):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, len(samples))
            print("    BATCH", batch, "SAMPLES", start, stop)

            likelihood_val = func_feed_samples(self, samples, start, stop, dgsm_lh=dgsm_lh,
                                               ops=[self._avg_train_likelihood])
            likelihood_values.append(likelihood_val)
            batch += 1
        return -np.mean(likelihood_values)
    

    def _start_training(self, samples, batch_size, likelihood_thres,
                        sess, func_feed_samples, num_epochs=None, dgsm_lh=None,
                        shuffle=True, dgsm_lh_test=None, samples_test=None):
        """
        Helper for train() in subclasses. Weights should have been initialized.

        `samples` (numpy.ndarray) numpy array of shape (D, ?).
        `func_feed_samples` (function; start, stop) function that feeds samples into the network.
                            It runs the train_likelihood, avg_train_likelihood, and accumulate_updates
                            Ops.
        `lh` (numpy.ndarray) dgsm likelihoods
        """
        print("Resetting accumulators...")
        sess.run(self._reset_accumulators)

        batch_likelihoods = []  # record likelihoods within an epoch
        train_likelihoods = []  # record likelihoods for training samples
        test_likelihoods = []   # record likelihoods for testing samples

        prev_likelihood = 100   # previous likelihood
        likelihood = 0          # current likelihood
        epoch = 0
        batch = 0

        # Shuffle
        if shuffle:
            print("Shuffling...")
            p = np.random.permutation(len(samples))
            smaples = samples[p]
            if dgsm_lh is not None:
                dgsm_lh = dgsm_lh[p]

        print("Starts training. Maximum epochs: %s   Likelihood threshold: %.3f" % (num_epochs, likelihood_thres))
        while (num_epochs and epoch < num_epochs) \
              or (not num_epochs and (abs(prev_likelihood - likelihood) > likelihood_thres)):
            start = (batch)*batch_size
            stop = min((batch+1)*batch_size, samples.shape[0])
            print("EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop, "  prev likelihood", prev_likelihood, "likelihood", likelihood)

            if self._learning_algorithm == spn.EMLearning:
                ads = max(np.exp(-epoch*self._smoothing_decay)*self._additive_smoothing,
                          self._min_additive_smoothing)
                sess.run(self._additive_smoothing_var.assign(ads))
                print("Smoothing: ", sess.run(self._additive_smoothing_var))

                train_likelihoods_arr, likelihood_train, _, = \
                    func_feed_samples(self, samples, start, stop, dgsm_lh=dgsm_lh,
                                      ops=[self._log_likelihood_root, self._avg_train_likelihood,
                                           self._learn_spn])
                sess.run(self._update_spn)

            batch_likelihoods.append(likelihood_train)
            batch += 1
            if stop >= samples.shape[0]:  # epoch finishes
                epoch += 1
                batch = 0

                # Shuffle
                if shuffle:
                    print("Shuffling...")
                    p = np.random.permutation(len(samples))
                    smaples = samples[p]
                    if dgsm_lh is not None:
                        dgsm_lh = dgsm_lh[p]

                if samples_test is not None:
                    print("Computing train, (test) likelihoods...")
                    likelihood_train = -np.mean(batch_likelihoods)
                    train_likelihoods.append(likelihood_train)
                    print("Train likelihood: %.3f  " % likelihood_train)

                    likelihood_test = self._compute_mle_loss(samples_test, func_feed_samples, dgsm_lh=dgsm_lh_test)
                    test_likelihoods.append(likelihood_test)
                    print("Test likelihood: %.3f  " % likelihood_test)

                prev_likelihood = likelihood
                likelihood = likelihood_train
                batch_likelihoods = []

        return train_likelihoods, test_likelihoods


    @staticmethod
    def dup_fun_up(inpt, *args,
                    conc=None, tmpl_num_vars=[0], tmpl_num_vals=[0], graph_num_vars=[0], labels=[[]], tspn=None):
        """
        Purely for template spn copying only. Supports template with multiple types of IVs.
        Requires that the template SPN contains only one concat node where all inputs go through.

        labels: (2D list) variable's numerical label, used to locate the variable's position in the big IVs.
                If there are multiple types of IVs, then this should be a 2D list, where each inner
                list is the label (starting from 0) for one type of IVs, and each outer list represents
                one type of IVs.
        """
        # Know what range of indices each variable takes
        node, indices = inpt
        if node.is_op:
            if isinstance(node, spn.Sum):
                # [2:] is to skip the weights node and the explicit IVs node for this sum.
                return spn.Sum(*args[2:], weights=args[0])
            elif isinstance(node, spn.ParSums):
                return spn.ParSums(*args[2:], weights=args[0], num_sums=tspn._num_mixtures)
            elif isinstance(node, spn.Product):
                return spn.Product(*args)
            elif isinstance(node, spn.PermProducts):
                return spn.PermProducts(*args)
            elif isinstance(node, spn.Concat):
                # The goal is to map from index on the template SPN's concat node to the index on
                # the instance SPN's concat node.
                
                # First, be able to tell which type of iv the index has
                ranges_tmpl = [0]  # stores the start (inclusive) index of the range of indices taken by a type of iv on template SPN
                ranges_instance = [0]  # stores the start (inclusive) index of the range of indices taken by a type of iv on instance SPN
                for i in range(len(tmpl_num_vars)):
                    ranges_tmpl.append(ranges_tmpl[-1] + tmpl_num_vars[i]*tmpl_num_vals[i])
                    ranges_instance.append(ranges_instance[-1] + graph_num_vars[i]*tmpl_num_vals[i])

                big_indices = []
                for indx in indices:
                    iv_type = -1
                    for i, start in enumerate(ranges_tmpl):
                        if indx < start + tmpl_num_vars[i]*tmpl_num_vals[i]:
                            iv_type = i
                            break
                    if iv_type == -1:
                        raise ValueError("Oops. Something wrong. Index out of range.")

                    # Then, figure out variable index and offset (w.r.t. template Concat node)
                    varidx = (indx - ranges_tmpl[iv_type]) // tmpl_num_vals[iv_type]
                    offset = (indx - ranges_tmpl[iv_type]) - varidx * tmpl_num_vals[iv_type]
                    # THIS IS the actual position of the variable's inputs in the big Concat.
                    varlabel = labels[iv_type][varidx]
                    big_indices.append(ranges_instance[iv_type] + varlabel * tmpl_num_vals[iv_type] + offset)
                return spn.Input(conc, big_indices)
        elif isinstance(node, spn.Weights):
            return node
        else:
            raise ValueError("Unexpected node %s. We don't intend to deal with IVs here. Please remove them from the concat." % node)
    # END fun_up
# -- END TemplateSpn -- #



class NodeTemplateSpn(TemplateSpn):

    """
    Spn on top of an n-node template.
    (The connectivity is not considered; it is
    implicit in the training data)
    """

    def __init__(self, template, *args, **kwargs):
        """
        Initialize an NodeTemplateSpn.

        template (NodeTemplate): a subclass of NodeTemplate.

        **kwargs:
           seed (int): seed for the random generator. Set before generating
                       the structure.
           num_vals (int): number of different values a variable (on a node) can take;
                       the values are discretized from 0 to num_vals-1.
        """
        super().__init__(template, *args, **kwargs)

        self._num_nodes = self.template.num_nodes()
        self._num_vals = kwargs.get("num_vals", -1)
        if self._num_vals <= 0:
            raise ValueError("num_vals must be positive!")

        # Don't use the layered generator for now
        if self._num_nodes == 1:
            self._input_dist = spn.DenseSPNGenerator.InputDist.RAW
            self._dense_gen = spn.DenseSPNGenerator(num_decomps=self._num_decomps, num_subsets=self._num_subsets,
                                                    num_mixtures=self._num_mixtures, input_dist=self._input_dist,
                                                    num_input_mixtures=self._num_input_mixtures)

        # Initialize structure and learning ops
        self._init_struct(rnd=self._rnd, seed=self._seed)


    @property
    def vn(self):
        return {
            'CATG_IVS':"Catg_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'CONC': "Conc_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'SEMAN_IVS': "Exp_Catg_%s_%d" % (self.__class__.__name__, self._num_nodes),
            'LH_CONT': "Exp_Lh_%s_%d" % (self.__class__.__name__, self._num_nodes)
        }
        
    @property
    def root(self):
        return self._root


    def _init_struct(self, *args, rnd=None, seed=None):
        """
        Initialize the structure for training.  (private method)

        rnd (Random): instance of a random number generator used for
                      dense generator. 
        """
        # An input to spn goes through a concat node before reaching the network. This makes
        # it convenient to expand the network downwards, by swapping the inputs to the concat
        # nodes. 

        # Inputs to the spn.
        self._catg_inputs = spn.IVs(num_vars =self._num_nodes,
                                    num_vals = self._num_vals,
                                    name = self.vn['CATG_IVS'])

        # Concat nodes, used to connect the inputs to the spn.
        self._conc_inputs = spn.Concat(spn.Input.as_input((self._catg_inputs, list(range(self._num_nodes*self._num_vals)))),
                                       name=self.vn['CONC'])        

        # Generate structure, weights, and generate learning operations.
        print("Generating SPN structure...")
        if seed is not None:
            print("[Using seed %d]" % seed)
            rnd = random.Random(seed)

        self._root = self._dense_gen.generate(self._conc_inputs, rnd=rnd)
        

    def train(self, sess, *args, **kwargs):
        """
        Train the SPN. Weights should have been initialized.

        sess (tf.Session): Tensorflow session object
        *args:
          samples (numpy.ndarray): A numpy array of shape (D,n) where D is the
                                   number of data samples, and n is the number
                                   of nodes in template modeled by this spn.

        **kwargs:
          shuffle (bool): shuffles `samples` before training. Default: False.
          num_batches (int): number of batches to split the training data into.
                             Default: 1
          likelihood_thres (float): threshold of likelihood difference between
                                    interations to stop training. Default: 0.05
        """
        def feed_samples(self, samples, start, stop, dgsm_lh=None, ops=[]):
            if dgsm_lh is None:
                return sess.run(ops, feed_dict={self._catg_inputs: samples[start:stop]})
                                
            else:
                return sess.run(ops,
                                feed_dict={self._semantic_inputs: np.full((stop-start, self._num_nodes),-1),
                                           self._likelihood_inputs: dgsm_lh[start:stop]})

        
        samples = args[0]
        D, n = samples.shape
        if n != self._num_nodes:
            raise ValueError("Invalid shape for `samples`." \
                             "Expected (?,%d), got %s)" % (self._num_nodes, samples.shape))
        
        shuffle = kwargs.get('shuffle', False)
        batch_size = kwargs.get('batch_size', 200)
        num_epochs = kwargs.get('num_epochs', None)
        likelihood_thres = kwargs.get('likelihood_thres', 0.05)
        dgsm_lh = kwargs.get('dgsm_lh', None)
        dgsm_lh_test = kwargs.get('dgsm_lh_test', None)
        samples_test = kwargs.get('samples_test', None)

        # Starts training
        return self._start_training(samples, batch_size, likelihood_thres,
                                    sess, feed_samples, num_epochs=num_epochs, dgsm_lh=dgsm_lh,
                                    shuffle=shuffle, samples_test=samples_test, dgsm_lh_test=dgsm_lh_test)
        
    
    def evaluate(self, sess, *args, **kwargs):
        """
        Feeds inputs into the network and return the output of the network. Returns the
        output of the network.

        sess (tf.Session): a session.

        *args:
          sample (numpy.ndarray): an (n,) numpy array as a sample.
          
          <if expanded>
          sample_lh ()numpy.ndarray): an (n,m) numpy array, where m is the number of categories,
                                      so [?,c] is a likelihood for class c (float).
        """
        # To feed to the network, we need to reshape the samples.
        sample = np.array([args[0]], dtype=int)
        
        if not self._expanded:
            likelihood_val = sess.run(self._log_likelihood_root, feed_dict={self._catg_inputs: sample})
        else:
            sample_lh = np.array([args[1].flatten()], dtype=float32)
            likelihood_val = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: sample,
                                                                            self._likelihood_inputs: sample_lh})
            
        return likelihood_val  # TODO: likelihood_val is now a numpy array. Should be a float.

            
    def infer_marginals(self, sess, query, query_lh=None, normalize=False):
        """
        Performs marginal inference on node template SPN. Returns the marginal distribution of the
        variables being inferred, in the form of a dictionary: i -> list (variable index -> likelihoods).

        `query` is an numpy array of shape (n,). The value at index i represents the assignment of Yi. If
            this value is -1, then the marginal distribution of this variable will be returned.
        `query_lh` is an numpy array of shape (n,m), used to supply likelihood values (local evidence) per node per class.
        `masked_only` is True if only perform marginal inference on nodes that are
                      masked, i.e. their value in the `query` is -1. Useful when there's no likelihoods and
                      we are just inferring missing semantics.
        """
        marginals = {}
        samples = []
        samples_lh = []
        for i in range(len(query)):
            if query[i] == -1:
                marginals[i] = []
                for v in range(self._num_vals):
                    sample = np.copy(query)
                    sample[i] = v
                    samples.append(sample)
                    if query_lh is not None:
                        samples_lh.append(query_lh.flatten())
        if query_lh is None:
            likelihood_y_vals = sess.run(self._log_likelihood_root, feed_dict={self._catg_inputs: np.array(samples)})

        else:
            likelihood_xy_vals = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: np.array(samples),
                                                                                self._likelihood_inputs: np.array(samples_lh)})

            # prob x
            likelihood_x_val = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: np.full((1, len(query)), -1),
                                                                              self._likelihood_inputs: np.array([query_lh.flatten()], dtype=float32)})[0]

            # prob y|x = prob xy / prob x
            likelihood_y_vals = np.array([likelihood_xy_vals[i] - likelihood_x_val for i in range(len(likelihood_xy_vals))])


        for c, i in enumerate(marginals.keys()):
            likelihoods_yi = likelihood_y_vals[c*self._num_vals:(c+1)*self._num_vals]
            if normalize:
                likelihoods_yi = np.exp(likelihoods_yi -   # plus and minus the max is to prevent overflow
                                        (np.log(np.sum(np.exp(likelihoods_yi - np.max(likelihoods_yi)))) + np.max(likelihoods_yi)))
            marginals[i] = likelihoods_yi.flatten()
        return marginals
    

    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        spn.JSONSaver(path, pretty=pretty).save(self._root, save_param_vals=True, sess = sess)
        self._saved_path = path

    
    def load(self, path, sess):
        """
        Loads the SPN structure and parameters. Replaces the existing structure
        of this SPN.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        loader = spn.JSONLoader(path)
        self._root = loader.load(load_param_vals=True, sess=sess)

        self._catg_inputs = loader.find_node(self.vn['CATG_IVS'])
        self._conc_inputs = loader.find_node(self.vn['CONC'])
        
        if self._expanded:
            self._likelihood_inputs = loader.find_node(self.vn['LH_CONT'])
            self._semantic_inputs = loader.find_node(self.vn['SEMAN_IVS'])
        self._saved_path = path

# -- END NodeTemplateSpn -- #
