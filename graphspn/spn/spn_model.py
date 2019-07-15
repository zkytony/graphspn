# SPN for all
# Python 3+ required.
#
# author: Kaiyu Zheng

from abc import ABC, abstractmethod
import yaml
import pickle

import tensorflow as tf
import libspn as spn
from libspn.graph.algorithms import traverse_graph


class SpnModel(ABC):

    """
    Class to be inherited by more concrete spn classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize
        """
        # args for libspn
        self._input_dist = kwargs.get('input_dist', spn.DenseSPNGeneratorLayerNodes.InputDist.RAW)
        self._num_decomps = kwargs.get('num_decomps', 1)
        self._num_subsets = kwargs.get('num_subsets', 2)
        self._num_mixtures = kwargs.get('num_mixtures', 2)
        self._num_input_mixtures = kwargs.get('num_input_mixtures', 2)
        self._weight_init_min = kwargs.get('weight_init_min', 10)
        self._weight_init_max = kwargs.get('weight_init_max', 11)
        
        self._value_inference_type = kwargs.get('value_inference_type', spn.InferenceType.MARGINAL)
        self._learning_algorithm = kwargs.get('learning_algorithm', spn.EMLearning)

        if self._learning_algorithm == spn.EMLearning:
            self._init_accum = kwargs.get('init_accum_val', 20)
            self._min_additive_smoothing = kwargs.get('min-additive_smoothing', 1)
            self._smoothing_decay = kwargs.get('smoothing_decay', 0.2)
            self._additive_smoothing = kwargs.get('additive_smoothing', 30)
            self._additive_smoothing_var = tf.Variable(self._additive_smoothing, dtype=spn.conf.dtype)

        elif self._learning_algorithm == spn.GDLearning:
            self._learning_type = kwargs.get('learning_type', spn.LearningType.DISCRIMINATIVE)
            self._learning_inference_type =  kwargs.get('learning_inference_type', spn.LearningInferenceType.SOFT)
            self._learning_rate = kwargs.get('learning_rate', 0.001)
            self._optimizer = kwargs.get('optimizer', tf.train.AdamOptimizer)
        
    @classmethod
    def params_list(cls, learning_algorithm):
        params = ["input_dist",
                  "num_decomps",
                  "num_subsets",
                  "num_mixtures",
                  "num_input_mixtures",
                  "weight_init_min",
                  "weight_init_max",
                  "value_inference_type",
                  "learning_algorithm"]

        if learning_algorithm == spn.EMLearning:
            params.extend([
                "min_additive_smoothing",
                "smoothing_decay",
                "additive_smoothing",
                "additive_smoothing_var"
            ])
        elif learning_algorithm == spn.GDLearning:
            params.extend([
                "learning_type",
                "learning_inference_type",
                "learning_rate",
                "optimizer"
            ])
        return params
        
    @property
    def weight_init_min(self):
        return self._weight_init_min


    @property
    def weight_init_max(self):
        return self._weight_init_max
    
        
    @abstractmethod
    def _init_struct(self, rnd=None, *args, **kwargs):
        """
        Initialize the structure for training. (private method)

        rnd (Random): instance of a random number generator used for
                      dense generator. 
        """
        pass

    

    @classmethod
    def serialize(cls, obj, file):
        """
        Deserialize this object

        obj (SpnModel): object to be serialized
        file: (from pickle doc) The `file` argument must have a write() method that accepts
              a single bytes argument. It can thus be an on-disk file opened for binary writing,
              an io.BytesIO instance, or any other custom object that meets this interface.
        """
        # side note: Python 3 has no key word 'file'.
        pickle.dump(self, obj, file)


    @classmethod
    def deserialize(cls, file):
        """
        Serialize this object

        file: (from pickle doc) The argument `file` must have two methods, a read() method that takes
        an integer argument, and a readline() method that requires no arguments. Both methods should
        return bytes. Thus file can be an on-disk file opened for binary reading, an io.BytesIO object,
        or any other custom object that meets this interface.
        """
        return pickle.load(self, obj, file)


    @abstractmethod
    def train(self, sess, *args, **kwargs):
        """
        Train the SPN.

        sess (tf.Session): Tensorflow session object
        """
        pass

    @property
    @abstractmethod
    def vn(self):
        """
        Returns a dictionary that contains the inputs VAR NODE NAMES
        for this spn model. Example entry: 'CATG_IVS_NAME' -> 'Catg_%s' % ...
        """
        pass

    @abstractmethod
    def save(self, path, sess, pretty=True):
        """
        Saves the SPN structure and parameters.

        path (str): save path.
        sess (tf.Session): session object, required to save parameters.
        """
        pass

    
    @abstractmethod
    def load(self, path, sess):
        """
        Loads the SPN structure and parameters.

        path(str): path to load spn.
        sess (tf.Session): session object, required to load parameters into current session.
        """
        pass

    
    @abstractmethod
    def evaluate(self, sess, *args, **kwargs):
        """
        Feeds inputs into the network and return the output of the network. Returns the
        output of the network

        sess (tf.Session): a session.
        """
        pass


    @abstractmethod
    def infer_marginals(self, sess, *args, **kwargs):
        """
        Performs marginal inference.
        """
        pass


    def print_params(self):
        print("===Parameters===")
        for param in SpnModel.params_list(self._learning_algorithm):
            val = getattr(self, "_" + param)
            print("%s: %s" % (param, str(val)))
        print("================")


    @classmethod
    def print_weights(cls, spn_root, sess):
        def fun(node):
            if hasattr(node, 'weights'):
                print("%s (%s)" % (node.name, node.__class__.__name__))
                w = sess.run(node.weights.node.get_value())
                print("%s [%s]" % (w, w.shape))

        traverse_graph(spn_root, fun, skip_params=False)


    @classmethod
    def print_structure(cls, spn_root, sess):
        def fun(node):
            if node.is_op:
                print("%s (%s)" % (node.name, node.__class__.__name__))
                for i in node.inputs:
                    if i.node is not None and (i.node.is_op or isinstance(i.node, spn.VarNode)):
                        print("    %s (%s)" % (i, i.node.__class__.__name__))
        traverse_graph(spn_root, fun, skip_params=True)


    @classmethod
    def copyweights(cls, sess, spn_trained, spn_target):
        """
        Copies parameters from spn_trained to spn_target. Assumes the two have exactly the
        same structure.
        
        ATTENTION: Currently, the only way to ensure identical structure in automatic generation
        is by providing a custom random instance to the dense generation function. After visualizing
        the produced spn graph, {the author} discovered that even though the random partitioning is
        identical, the resulting partitions are not always connected in the same order as inputs to
        to parent sum node. Therefore, we have to first match the sum nodes with equal size of weight nodes,
        then assign the weights according to the mapping.
             Also, there are multiple sum nodes with the same ize at the same level. How to map them? Right
        now, the nodes at the same level across two different spns do not have index-correspondence.

        spn_trained and spn_target are both root nodes
        """

        # def fun(node):
        #     if hasattr(node, 'weights'):
        #         print("Yo %s (%s)" % (node.name, node.__class__.__name__))
        #         w = sess.run(node.weights.node.get_value())
        #         print("%s [%s]" % (w, w.shape))


        # print("=====FROM=========")
        # traverse_graph(spn_trained, fun, skip_params=False)
        # print("======TO==========")
        # traverse_graph(spn_target, fun, skip_params=False)

                
        stack = [(spn_trained, spn_target)]
        while len(stack) > 0:

            spn_from, spn_to = stack.pop()

            if isinstance(spn_from, spn.Sum):
                assert isinstance(spn_to, spn.Sum)
                if spn_to.weights.node.num_weights != spn_from.weights.node.num_weights:
                    raise AssertionError("SPN structures not identical. Weights number differ. FROM: %s. TO: %s" % (spn_from.weights.node.num_weights, spn_to.weights.node.num_weights))
                sess.run(spn_to.weights.node.assign(spn_from.weights.node.get_value()))

                w1 = sess.run(spn_from.weights.node.get_value())
                w2 = sess.run(spn_to.weights.node.get_value())
                assert w1.all() == w2.all()

            if not isinstance(spn_from, spn.IVs) and not isinstance(spn_from, spn.ContVars)  and not isinstance(spn_from, spn.Concat):
                for i, _ in enumerate(spn_from.values):
                    if type(spn_from.values[i].node) != type(spn_to.values[i].node):
                        raise AssertionError("SPN structures not identical. FROM: %s. TO: %s" % (spn_from.values[i].node, spn_to.values[i].node))
                    stack.append((spn_from.values[i].node, spn_to.values[i].node))

    @classmethod
    def add_constant_to_weights(cls, sess, root, val):
        """
        Adds a constant value `val` to every weight of the `root` node. Does not affect the children
        of `root`. Will do nothing if `root` does not have weights.

        root (spn.Node): root node of an SPN
        val (float): constant value to add to each weight
        """
        if isinstance(root, spn.Sum):
            w = sess.run(root.weights.node.get_value())
            w = w + val
            sess.run(root.weights.node.assign(w))


    @classmethod
    def make_weights_same(cls, sess, spn_root):
        """
        Makes the weights of SPN rooted by `spn_xroot` node to have the same value.

        root (spn.Node): root node of an SPN
        val (float): constant value to add to each weight
        """
        def fun(node):
            if hasattr(node, 'weights'):
                w = sess.run(node.weights.node.get_value())
                w.fill(1)
                sess.run(node.weights.node.assign(w))

        traverse_graph(spn_root, fun, skip_params=False)

##############################
# WARNING: MY OWN METHOD!
##############################
# The following method is used to copy spn structures.
from collections import deque, defaultdict
def mod_compute_graph_up(root, val_fun, **kwargs):
    """Computes a certain value for the ``root`` node in the graph, assuming
    that for op nodes, the value depends on values produced by inputs of the op
    node. For this, it traverses the graph depth-first from the ``root`` node
    to the leaf nodes. (Modified libspn.compute_graph_up())

    Args:
        root (Node): The root of the SPN graph.
        val_fun (function): A function ``val_fun(input, *args)`` producing a
            certain value for the ``input``. If ``input`` has an op node, it will have
            additional arguments with values produced for the input nodes of
            this node.  The arguments will NOT be added if ``const_fun``
            returns ``True`` for the node. The arguments can be ``None`` if
            the input was empty.

        **kwargs: additional arguments for val_fun

    Returns:
        The value for the ``root`` node.


    NOTE: To actually use this function to copy an SPN structure, follow the following
    steps:
    1. MaKe sure your SPN has a SINGLE concat node that connects the inputs to the SPN.
    2. Remove the inputs from the concat node.
    3. Use the following function as val_func. Note that `conc` is a variable used to
       to store the conc node for the newly created SPN.

        conc = None
        def fun_up(inpt, *args):
            global conc
            node, indices = inpt
            if node.is_op:
                if isinstance(node, spn.Sum):
                    # [2:] is to skip the weights node and the explicit IVs node for this sum.
                    return spn.Sum(*args[2:], weights=args[0])
                elif isinstance(node, spn.Product):
                    return spn.Product(*args)
                elif isinstance(node, spn.Concat):
                    conc = spn.Concat()  # assume there is only one concat node.
                    return spn.Input(conc, indices)
            elif isinstance(node, spn.Weights):
                return node
            else:
                raise ValueError("We don't intend to deal with IVs here. Please remove them from the concat.")

    4. Call this function: new_root=mod_compute_graph_up(root, val_fun=fun_up)
    5. Connect the `conc` to the desired input IVs, using conc.set_inputs(*)
    """
    all_values = {}
    stack = deque()  # Stack of inputs to process
    stack.append((root, None))  # node and index

    while stack:
        next_input = stack[-1]
        # Was this node already processed?
        # This might happen if the node is referenced by several parents
        if next_input not in all_values:
            if next_input[0].is_op:
                # OpNode
                input_vals = []  # inputs to the node of 'next_input'
                all_input_vals = True
                # Gather input values for non-const val fun
                for inpt in next_input[0].inputs:
                    if inpt:  # Input is not empty
                        try:
                            # Check if input_node in all_vals
                            if inpt.indices is None:
                                input_vals.append(all_values[(inpt.node, None)])
                            else:
                                input_vals.append(all_values[(inpt.node, tuple(inpt.indices))])
                        except KeyError:
                            all_input_vals = False
                            if inpt.indices is None:
                                stack.append((inpt.node, None))
                            else:
                                stack.append((inpt.node, tuple(inpt.indices)))
                    else:
                        # This input was empty, use None as value
                        input_vals.append(None)
                # Got all inputs?
                if all_input_vals:
                    last_val = val_fun(next_input, *input_vals, **kwargs)
                    all_values[next_input] = last_val
                    stack.pop()
            else:
                # VarNode, ParamNode
                last_val = val_fun(next_input, **kwargs)
                all_values[next_input] = last_val
                stack.pop()
        else:
            stack.pop()

    return last_val
##############################
# END WARNING: MY OWN METHOD!
##############################
