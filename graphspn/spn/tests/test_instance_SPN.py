import random
import numpy as np
import os
import tensorflow as tf
from graphspn.spn.spn_model import SpnModel
from graphspn.graphs.graph_builder import build_graph
from graphspn.graphs.template import NodeTemplate, ThreeNodeTemplate, SingletonTemplate, PairTemplate
from graphspn.spn.template_spn import TemplateSpn, NodeTemplateSpn
from graphspn.spn.instance_spn import NodeTemplateInstanceSpn
from graphspn.experiments.scripts.topo_map_dataset import TopoMapDataset
from graphspn.util import ABS_DIR
from graphspn.experiments.scripts.topo_map import CategoryManager, ColdDatabaseManager
from test_template_spn import create_likelihoods_for_single_node, create_likelihoods_vector
import graphspn.graphs.graph as g
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DB_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-topomaps")
GROUNDTRUTH_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-groundtruth")


class SimpleSuperNode(g.SuperNode):
    
    def __init__(self, id, enset):
        """
        id (int) is for this node
        enset (EdgeNodeSet) underlying graph structure
        """
        super().__init__(id, enset)

    @classmethod
    def pick_id(cls, enset, existing_ids):
        """
        that always returns the same id if given the same enset, and different
        if otherwise.
        """
        if len(existing_ids) == 0:
            return 0
        return max(existing_ids) + 1

    @property
    def coords(self):
        """returns an (x,y) location on the plane, for visualization purposes.
        The coordinates have resolution """
        if hasattr(self, "_coords"):
            return self._coords
        else:
            if len(self.enset.nodes) == 0:
                return (0, 0)
            else:
                x_coord = sum(self.enset.nodes[nid].coords[0] for nid in self.enset.nodes) / len(self.enset.nodes)
                y_coord = sum(self.enset.nodes[nid].coords[1] for nid in self.enset.nodes) / len(self.enset.nodes)
            self._coords = (x_coord, y_coord)
            return self._coords

    @property
    def color(self):
        return ["green", "red", "blue", "brown", "purple", "cyan", "grey"][len(self.enset.nodes)]


class SimpleSuperEdge(g.SuperEdge):
    
    def __init__(self, id, supernode1, supernode2):
        super().__init__(id, supernode1, supernode2)

    @classmethod
    def pick_id(cls, supernode1, supernode2, existing_ids=set({}), edge=None):
        """
        that always returns the same id if given the same subgraph, and different
        if otherwise.
        """
        if len(existing_ids) == 0:
            return 0        
        return max(existing_ids) + 1

    @property
    def color(self):
        return ["green", "red", "blue", "brown", "purple", "cyan", "grey"][len(self.nodes[0].enset.nodes)]

class LabelNode(g.Node):
    def __init__(self, id, label, color="orange"):
        super().__init__(id, color)
        self._label = label
        
    @property
    def label(self):
        return self._label


def simple_interpret_node(nid, tokens):
    color = tokens[1]
    catg = int(tokens[0])
    return LabelNode(nid, catg, color)

def simple_interpret_edge(eid, node1, node2, tokens):
    return g.Edge(eid, node1, node2)

def setup_dataset(*dbs):
    dataset = TopoMapDataset(DB_ROOT)
    for db in dbs:
        dataset.load(db, skip_unknown=True,
                     skip_placeholders=True, single_component=False)
    return dataset

def setup_template_spns(dataset, num_vals,
                        batch_size=1000, num_epochs=5):
    templates = [ThreeNodeTemplate, SingletonTemplate, PairTemplate]
    tspns = [NodeTemplateSpn(template, num_vals=num_vals) for template in templates]
    sess = tf.Session()
    for i, template in enumerate(templates):
        tspn = tspns[i]
        tspn.generate_random_weights()
        tspn.init_weights_ops()
        tspn.init_learning_ops()
        tspn.initialize_weights(sess)
        sess.run(tf.global_variables_initializer())

        if template == SingletonTemplate:
            SpnModel.make_weights_same(sess, tspn.root)
        else:
            samples = np.array(dataset.create_template_dataset([template], db_names=["Stockholm"])[template.__name__]['Stockholm'],
                               dtype=np.int32)
            train_likelihoods, test_likelihoods \
                = tspn.train(sess, samples, shuffle=True, batch_size=batch_size,
                             likelihood_thres=0.05, num_epochs=num_epochs, dgsm_lh=None,
                             samples_test=None, dgsm_lh_test=None)
    return sess, tspns

def setup_instance_spn(sess, graph, tspns, expand=False):
    # Remove inputs to template spns
    for tspn in tspns:
        tspn._conc_inputs.set_inputs()
    
    ispn = NodeTemplateInstanceSpn(graph, sess, *[(tspn , tspn.template) for tspn in tspns],
                                   num_partitions=5,
                                   seq_id="haha",
                                   visualize_partitions_dirpath=None,
                                   db_name="Stockholm",
                                   divisions=8,
                                   super_node_class=SimpleSuperNode,
                                   super_edge_class=SimpleSuperEdge)
    assert ispn.root.is_valid()
    if expand:
        ispn.expand()
    ispn.init_ops(no_mpe=True)
    return ispn


def setup_test_case(graph, prob_incorrect=0.2):
    lh = {}

    query = {}
    for nid in graph.nodes:
        masked = False
        query[nid] = graph.nodes[nid].label
        if random.uniform(0, 1/prob_incorrect) < 1:
            masked = True
            query[nid] = -1
        lh[nid] = np.array(create_likelihoods_for_single_node(graph.nodes[nid].label,
                                                              (0.5, 0.7), (0.0004, 0.00065),
                                                              None, None,
                                                              masked=masked,
                                                              uniform_for_incorrect=True))
    return query, lh


def test_infer_marginals(sess, ispn, lh):
    query = {nid: -1 for nid in ispn.graph.nodes}
    query_nids = [nid for nid in ispn.graph.nodes]
    marginals = ispn.infer_marginals(sess, query_nids, query, query_lh=lh, normalize=True)
    print(marginals)
    most_likely = query
    for i in marginals:
        most_likely[i] = np.argmax(marginals[i])
    return most_likely

def main():
    CategoryManager.TYPE = "FULL"
    CategoryManager.init()        
    graph = build_graph("graph_example.graph", simple_interpret_node, simple_interpret_edge)

    # ax = plt.gca()
    # graph.visualize(ax, list(graph.nodes.keys()), dotsize=10)
    # plt.autoscale()
    # ax.set_aspect('equal', 'box')
    # plt.show()
    
    dataset = setup_dataset("Stockholm")
    sess, tspns = setup_template_spns(dataset, CategoryManager.NUM_CATEGORIES,
                                      num_epochs=5)
    ispn = setup_instance_spn(sess, graph, tspns, expand=True)
    query, lh = setup_test_case(graph, prob_incorrect=0.2)
    result = test_infer_marginals(sess, ispn, lh)
    print("Query: ")
    print(query)
    print("Result: ")
    print(result)

if __name__ == '__main__':
    main()
