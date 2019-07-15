import sys
from abc import abstractmethod
import tensorflow as tf
import libspn as spn
import numpy as np
from numpy import float32
import random
import copy
import os

from graphspn.spn.template_spn import TemplateSpn
from graphspn.spn.spn_model import SpnModel, mod_compute_graph_up
from graphspn.graphs.template import NodeTemplate


class InstanceSpn(SpnModel):
    """
    Given a topological map instance, create an spn from given template spns for this instance.
    """
    def __init__(self, graph, sess, *spns, **kwargs):
        """
        Initializes the structure and learning operations of an instance spn.

        Note that all ivs variables are ordered by sorted(graph.nodes). You can retrieve the
        value of the MPE result of a node by following this order.

        graph (TopologicalMap): a topo map instance.
        num_partitions (int): number of partitions (children for root node)
        spns (list): a list of tuples (TemplateSpn, Template). Note that assume
                     all templates are either node templates or edge templates.
        sess (tf.Session): a session that contains all weights.

        **kwargs:
           num_partitions (int) number of child for the root sum node.
           graph_name (str): sequence id for the given topo map instance. Used as identified when
                         saving the instance spn. Default: "default_1"
           no_init (bool): True if not initializing structure; user might want to load structure
                           from a file.
           visualize_partitions_dirpath (str): Path to save the visualization of partitions on each child.
                                            Default is None, meaning no visualization is saved.
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
           partitions (list):  list of pre-specified partitions of the graph on which the graphspn is
                               built. Therefore, no more partition is produced when building the model.
           super_node_class (SuperNode) used when partitioning the graph. See graph.partition()
           super_edge_class (SuperEdge) used when partitioning the graph. See graph.partition()

           If template is EdgeTemplate or EdgeRelationTemplate, then:
             divisions (int) number of views per place
        """
        super().__init__(**kwargs)
        self._graph_name = kwargs.get("graph_name", "default_1")
        self._super_node_class = kwargs.get("super_node_class")
        self._super_edge_class = kwargs.get("super_edge_class")
        # sort spns by template size.
        self._spns = sorted(spns, key=lambda x: x[1].size(), reverse=True)
        self._num_vals = self._spns[0][0]._num_vals
        self._template_mode = self._spns[0][1].code()
        self._graph = graph
        self._expanded = False

    @property
    def graph(self):
        return self._graph

    @property
    @abstractmethod
    def vn(self):
        pass

    @property
    def root(self):
        return self._root

    @property
    def templates(self):
        return [t[1] for t in self._spns]


    @abstractmethod
    def _init_struct(self, sess, divisions=-1, num_partitions=1, partitions=None,
                     extra_partition_multiplyer=1):
        """
        Initialize the structure for training. (private method)

        sess: (tf.Session): a session that contains all weights.

        **kwargs:
           num_partitions (int): number of partitions (children for root node)
           If template is EdgeTemplate, then:
             divisions (int) number of views per place
           spn_paths (dict): a dictionary from Template to path. For loading the spn for Template at
                             path.
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
        """
        pass

    def _init_ops_basics(self):
        print("Initializing learning Ops...")
        if self._learning_algorithm == spn.GDLearning:
            learning = spn.GDLearning(self._root, log=True,
                                      value_inference_type=self._value_inference_type,
                                      learning_rate=self._learning_rate,
                                      learning_type=self._learning_type,
                                      learning_inference_type=self._learning_inference_type)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.learn(optimizer=self._optimizer)
            
        elif self._learning_algorithm == spn.EMLearning:
            learning = spn.EMLearning(self._root, log=True, value_inference_type = self._value_inference_type,
                                  additive_smoothing = self._additive_smoothing_var, use_unweighted=True)
            self._reset_accumulators = learning.reset_accumulators()
            self._learn_spn = learning.accumulate_updates()
            self._update_spn = learning.update_spn()

        self._log_likelihood_root = learning.value.values[self._root]
        self._avg_train_likelihood = tf.reduce_mean(self._log_likelihood_root)
        

    @abstractmethod
    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        pass

    @abstractmethod
    def infer_marginals(self, sess, query_nids, query, query_lh=None):
        """
        Computes marginal distribution of queried variables.

        Now, only uses the DUMMY method. Iterate over all possible assignments of
        all inquired variables, evaluate the network and use the result as the likelihood
        for that assignment. Note that the marginals are in log space.

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.
        """
        pass

    @abstractmethod
    def evaluate(self, sess, sample, sample_lh=None):
        """
        sess (tf.Session): a session.
        """
        pass


    @abstractmethod
    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.
        """
        pass


    def expand(self, use_cont_vars=False):
        """
        Custom method.

        Replaces the IVs inputs with a product node that has two children: a continuous
        input for likelihood, and a discrete input for semantics category.

        Do nothing if already expanded.
        """
        
        if not self._expanded:

            print("Expanding...")

            num_vars = len(self._graph.nodes)
            self._semantic_inputs = spn.IVs(num_vars=num_vars, num_vals=self._num_vals)
            # Note: use RawInput when doing cold database experiments with dgsm input. Use ContVars for synthetic experiments
            if use_cont_vars:
                self._likelihood_inputs = spn.ContVars(num_vars=num_vars*self._num_vals) #spn.RawInput(num_vars=num_vars*self._num_vals)
            else:
                self._likelihood_inputs = spn.RawInput(num_vars=num_vars*self._num_vals) #spn.RawInput(num_vars=num_vars*self._num_vals)

            prods = []
            for i in range(num_vars):
                for j in range(self._num_vals):
                    prod = spn.Product(
                        (self._likelihood_inputs, [i*self._num_vals + j]),
                        (self._semantic_inputs, [i*self._num_vals + j])
                    )
                    prods.append(prod)
            self._conc_inputs.set_inputs(*map(spn.Input.as_input, prods))
            self._expanded = True


    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        raise NotImplemented


    def load(self, path, sess):
        """
        Loads the SPN structure and parameters.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        raise NotImplemented


    def train(self, sess, *args, **kwargs):
        """
        Train the SPN.

        sess (tf.Session): Tensorflow session object
        """
        raise NotImplemented


class NodeTemplateInstanceSpn(InstanceSpn):

    
    def __init__(self, graph, sess, *spns, **kwargs):
        """
        partitions (list): list of partitions to build the GraphSPN on, if not None.
                           Each partition is a dictionary, mapping from template class
                           to a graph (TopologicalMap object) that covers the nodes and
                           edges that were grouped by that template.
        """
        super().__init__(graph, sess, *spns, **kwargs)

        assert self._template_mode == NodeTemplate.code()
        
        self._id_incr = 0
        num_partitions = kwargs.get('num_partitions', 1)
        self._partitions = []
        self._init_struct(sess, divisions=kwargs.get('divisions', -1),
                          num_partitions=kwargs.get('num_partitions', 1),
                          partitions=kwargs.get('partitions', None),
                          extra_partition_multiplyer=kwargs.get('extra_partition_multiplyer', 1))
        self._inputs_size = len(self._graph.nodes)


    @property
    def vn(self):
        return {
            'CATG_IVS': "Catg_%d_%s" % (self._template_mode, self._graph_name),
            'VIEW_IVS':"View_%d_%s" % (self._template_mode, self._graph_name),
            'CONC': "Conc_%d_%s" % (self._template_mode, self._graph_name),
            'SEMAN_IVS': "Exp_Catg_%d_%s" % (self._template_mode, self._graph_name),
            'LH_CONT': "Exp_Lh_%d_%s" % (self._template_mode, self._graph_name)
        }

    @property
    def partitions(self):
        return self._partitions
        
    def _inputs_list(self, input_type):
        """
        input_type can be 'catg' or 'conc', or 'view' (edge only)
        """
        inputs = [self._template_iv_map[tid][input_type] for tid in sorted(self._template_iv_map)]
        return inputs

        
    def _init_struct(self, sess, divisions=-1, num_partitions=1, partitions=None,
                     extra_partition_multiplyer=1):
        """
        Initialize the structure for training. (private method)

        sess: (tf.Session): a session that contains all weights.

        **kwargs:
           num_partitions (int): number of partitions (children for root node)
           If template is EdgeTemplate, then:
             divisions (int) number of views per place
           extra_partition_multiplyer (int): Used to multiply num_partitions so that more partitions
                                             are tried and ones with higher coverage are picked.
        """

        for tspn, template in self._spns:
            # remove inputs; this is necessary for the duplication to work - we don't want
            # the indicator variables to the template spns because the instance spn has its
            # own indicator variable inputs.
            tspn._conc_inputs.set_inputs()
            
        # Create vars and maps
        self._catg_inputs = spn.IVs(num_vars=len(self._graph.nodes), num_vals=self._num_vals)
        self._conc_inputs = spn.Concat(self._catg_inputs)
        self._template_nodes_map = {}  # map from template id to list of node lds
        self._node_label_map = {}  # key: node id. Value: a number (0~num_nodes-1)
        self._label_node_map = {}  # key: a number (0~num_nodes-1). Value: node id
        _i = 0
        for nid in self._graph.nodes:
            self._node_label_map[nid] = _i
            self._label_node_map[_i] = nid
            _i += 1

        if partitions is None:

            """Try partition the graph `extra_partition_multiplyer` times more than what is asked for. Then pick the top `num_partitions` with the highest
            coverage of the main template."""
            print("Partitioning the graph... (Selecting %d from %d attempts)" % (num_partitions, extra_partition_multiplyer*num_partitions))
            partitioned_results = {}
            main_template = self._spns[0][1]
            for i in range(extra_partition_multiplyer*num_partitions):
                """Note: here, we only partition with the main template. The results (i.e. supergraph, unused graph) are stored
                and will be used later. """
                unused_graph, supergraph = self._graph.partition(main_template, get_unused=True,
                                                                 super_node_class=self._super_node_class,
                                                                 super_edge_class=self._super_edge_class)
                if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                    coverage = len(supergraph.nodes)*main_template.size() / len(self._graph.nodes)
                    partitioned_results[(i, coverage)] = (supergraph, unused_graph)
            used_coverages = set({})
            for i, coverage in sorted(partitioned_results, reverse=True, key=lambda x:x[1]):
                used_coverages.add((i, coverage))
                sys.stdout.write("%.3f  " % coverage)
                if len(used_coverages) >= num_partitions:
                    break
            sys.stdout.write("\n")

            """Keep partitioning the used partitions, and obtain a list of partitions in the same format as the `partitions` parameter"""
            partitions = []
            for key in used_coverages:
                supergraph, unused_graph = partitioned_results[key]
                partition = {main_template: supergraph}
                # Keep partitioning the unused_graph using smaller templates
                for _, template in self._spns[1:]:  # skip main template
                    unused_graph_2nd, supergraph_2nd = unused_graph.partition(template, get_unused=True,
                                                                              super_node_class=self._super_node_class,
                                                                              super_edge_class=self._super_edge_class)
                    partition[template] = supergraph_2nd
                    unused_graph = unused_graph_2nd
                partitions.append(partition)

        """Building instance spn"""
        print("Building instance spn...")
        pspns = []
        tspns = {}
        for template_spn, template in self._spns:
            tspns[template.__name__] = template_spn

        """Making an SPN"""
        """Now, partition the graph, copy structure, and connect self._catg_inputs appropriately to the network."""
        # Main template partition
        _k = 0

        self._partitions = partitions

        for _k, partition in enumerate(self._partitions):
            print("Partition %d" % (_k+1))
            nodes_covered = set({})
            template_spn_roots = []
            for template_spn, template in self._spns:
                supergraph = partition[template]
                print("Will duplicate %s %d times." % (template.__name__, len(supergraph.nodes)))
                template_spn_roots.extend(NodeTemplateInstanceSpn._duplicate_template_spns(self, tspns, template, supergraph, nodes_covered))

                ## TEST CODE: COMMENT OUT WHEN ACTUALLY RUNNING
                # original_tspn_root = tspns[template.__name__].root
                # duplicated_tspn_root = template_spn_roots[-1]
                # original_tspn_weights = sess.run(original_tspn_root.weights.node.get_value())
                # duplicated_tspn_weights = sess.run(duplicated_tspn_root.weights.node.get_value())
                # print(original_tspn_weights)
                # print(duplicated_tspn_weights)
                # print(original_tspn_weights == duplicated_tspn_weights)
                # import pdb; pdb.set_trace()
                
            assert nodes_covered == self._graph.nodes.keys()
            p = spn.Product(*template_spn_roots)
            assert p.is_valid()
            pspns.append(p) # add spn for one partition
        ## End for loop ##
        
        # Sum up all
        self._root = spn.Sum(*pspns)
        assert self._root.is_valid()
        self._root.generate_weights(trainable=True)
        # initialize ONLY the weights node for the root
        sess.run(self._root.weights.node.initialize())
                
    #---- end init_struct ----#

    @classmethod    
    def _duplicate_template_spns(cls, ispn, tspns, template, supergraph, nodes_covered):
        """
        Convenient method for copying template spns. Modified `nodes_covered`.
        """
        sys.stdout.write("Duplicating %s... " % template.__name__)
        roots = []
        __i = 0
        for compound_nid in supergraph.nodes:
            nids = [node.id for node in supergraph.nodes[compound_nid].nodes_list()]
            ispn._template_nodes_map[ispn._id_incr] = nids

            # Make the right indices (with respect to the full conc node)
            labels = []
            for nid in nids:
                # The ivs is arranged like: [...(num_catgs)] * num_nodes
                label = ispn._node_label_map[nid]
                nodes_covered.add(nid)
                labels.append(label)

            tspn = tspns[template.__name__]
            sys.stdout.write("%d " % (__i+1))
            sys.stdout.flush()
            copied_tspn_root = mod_compute_graph_up(tspn.root,
                                                    TemplateSpn.dup_fun_up,
                                                    tmpl_num_vars=[len(nids)],
                                                    tmpl_num_vals=[tspn._num_vals],
                                                    graph_num_vars=[len(ispn._graph.nodes)],
                                                    conc=ispn._conc_inputs,
                                                    tspn=tspn,
                                                    labels=[labels])
            assert copied_tspn_root.is_valid()
            roots.append(copied_tspn_root)
            __i+=1
            ispn._id_incr += 1
        sys.stdout.write("\n")
        return roots
    #---- end duplicate_template_spns ----#
        

    def init_ops(self, no_mpe=False):
        """
        Init learning ops & MPE state.
        """
        self._init_ops_basics()
        if not no_mpe:
            print("Initializing MPE Ops...")
            mpe_state_gen = spn.MPEState(log=True, value_inference_type=spn.InferenceType.MPE)
            if self._template_mode == NodeTemplate.code():  ## NodeTemplate
                if not self._expanded:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._catg_inputs)
                else:
                    self._mpe_state = mpe_state_gen.get_state(self._root, self._semantic_inputs)

                    
    def infer_marginals(self, sess, query_nids, query, query_lh=None, normalize=False):
        """
        Computes marginal distribution of queried variables. Returns the marginal
        distribution of the variables being inferred, in the form of a dictionary:
        i -> list (variable index -> likelihoods).

        sess (tf.Session): a session.
        query_nids(list): list of node ids whose marginal distribution is inquired.

        *args
            query (dict): A dictionary mapping from the index i of Yi (node id) to its assignment.
                           ASSUME that the inquired nodes have already been assigned to '-1',
                           whose marginal distributions will be returned.
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id i to a numpy array of likelihoods
                               of size (m,) where m is the number of values Yi can take.
        Returns:
           a dictionary. Key: node id (corresponds to a node id in query_nids).
                         Value: log-space marginal probability distribution of this node.
        """
        marginals = {}
        samples = []
        samples_lh = []
        ivs_assignment = []
        for l in range(len(self._label_node_map)):
            nid = self._label_node_map[l]
            ivs_assignment.append(query[nid])
            
        if query_lh is not None:
            # expanded. So get likelihoods
            lh_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                lh_assignment.extend(query_lh[nid])
        
        for nid in query_nids:
            assert query[nid] == -1
            marginals[nid] = []
            for v in range(self._num_vals):
                sample = np.copy(ivs_assignment)
                sample[self._node_label_map[nid]] = v
                samples.append(sample)
                if query_lh is not None:
                    samples_lh.append(lh_assignment)  #lh_assignment is already flat

        if query_lh is None:
            likelihood_y_vals = sess.run(self._log_likelihood_root,
                                         feed_dict={self._catg_inputs: np.array(samples)})[0]
        else:
            likelihood_xy_vals = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: np.array(samples),
                                                                                self._likelihood_inputs: np.array(samples_lh)})

            # prob x
            likelihood_x_val = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: np.full((1, len(query)), -1),
                                                                              self._likelihood_inputs: np.array([lh_assignment], dtype=float32)})[0]

            # prob y|x = prob xy / prob x
            likelihood_y_vals = np.array([likelihood_xy_vals[i] - likelihood_x_val for i in range(len(likelihood_xy_vals))])

        for label in range(len(self._label_node_map)):
            likelihoods_yi = likelihood_y_vals[label*self._num_vals:(label+1)*self._num_vals]
            if normalize:
                likelihoods_yi = np.exp(likelihoods_yi -   # plus and minus the max is to prevent overflow
                                        (np.log(np.sum(np.exp(likelihoods_yi - np.max(likelihoods_yi)))) + np.max(likelihoods_yi)))
            marginals[self._label_node_map[label]] = likelihoods_yi.flatten()
        return marginals

        # marginals = {}
        # for nid in query_nids:
        #     orig = query[nid]
        #     marginals[nid] = []
        #     for val in range(self._num_vals):
        #         query[nid] = val
        #         marginals[nid].append(self.evaluate(sess, query, sample_lh=query_lh))
        #         query[nid] = orig
        # marginals = {nid:np.array(marginals[nid]) for nid in marginals}
        # return marginals

                
    def evaluate(self, sess, sample, sample_lh=None):
        """
        sess (tf.Session): a session.

        *args
            sample (dict): A dictionary mapping from the index i of Yi (node id) to its assignment.
          If expanded (additional parameters)
              query_lh (dict): A dictionary mapping from node id i to a numpy array of likelihoods
                               of size (m,) where m is the number of values Yi can take.
        """
        ivs_assignment = []
        for l in range(len(self._label_node_map)):
            nid = self._label_node_map[l]
            ivs_assignment.append(sample[nid])

        if not self._expanded:
            lh_val = sess.run(self._log_likelihood_root, feed_dict={self._catg_inputs: np.array([ivs_assignment], dtype=int)})[0]
        else:
            # expanded. So get likelihoods
            lh_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                lh_assignment.extend(list(sample_lh[nid]))
            lh_val = sess.run(self._log_likelihood_root, feed_dict={self._semantic_inputs: np.array([ivs_assignment], dtype=int),
                                                                    self._likelihood_inputs: np.array([lh_assignment])})[0]
        return lh_val


    
    def mpe_inference(self, sess, query, query_lh=None):
        """
        sess (tf.Session): a session.

        *args
          If NodeTemplate:
            query (dict): A dictionary mapping from node id to its category number
          If expanded (additional parameters)
            if NodeTemplate
              query_lh (dict): A dictionary mapping from node id to a tuple of NUM_CATEGORIES
                                number of float values, as likelihoods.
        Returns:
           If NodeTemplate:
              returns a category map (from id to category number)
        """
        ivs_assignment = []
        for l in range(len(self._label_node_map)):
            nid = self._label_node_map[l]
            ivs_assignment.append(query[nid])

        if not self._expanded:
            result = sess.run(self._mpe_state, feed_dict={self._catg_inputs: np.array([ivs_assignment], dtype=int)})[0]
        else:
            # expanded. So get likelihoods
            lh_assignment = []
            for l in range(len(self._label_node_map)):
                nid = self._label_node_map[l]
                lh_assignment.extend(list(query_lh[nid]))
            result = sess.run(self._mpe_state, feed_dict={self._semantic_inputs: np.array([ivs_assignment], dtype=int),
                                                          self._likelihood_inputs: np.array([lh_assignment])})[0]

        catg_map = {}
        for i in range(result.shape[1]):
            nid = self._label_node_map[i]
            catg_map[nid] = result[0][i]

        return catg_map
#------- END NodeTemplateinstancespn --------#
