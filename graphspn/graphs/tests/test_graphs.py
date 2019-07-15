from graphspn.graphs.graph_builder import build_graph
import random
import matplotlib.pyplot as plt
import numpy as np
import graphspn.graphs.graph as g
import graphspn.graphs.template as t
import graphspn.util as util


def simple_interpret_node(nid, tokens):
    color = tokens[0]
    return g.Node(nid, color)

def simple_interpret_edge(eid, node1, node2, tokens):
    return g.Edge(eid, node1, node2)



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


def setup_fixed():
    nodes = {}
    for i in range(8):
        node = g.Node(i)
        nodes[i] = node

    conns = {
        0: {1,2},
        1: {0,2},
        2: {0,1,3,7},
        3: {2,4,6},
        4: {3},
        5: {2},
        6: {3},
        7: {2}
    }
    edges = {}

    i = 0
    for nid1 in conns:
        for nid2 in conns[nid1]:
            edge = g.Edge(i, nodes[nid1], nodes[nid2])
            edges[i] = edge
            i += 1
    return g.Graph(edges)

def setup_random(num_nodes, num_edges, multi=False):
    nodes = {}
    for i in range(num_nodes):
        node = g.Node(i)
        nodes[i] = node

    conns = {}
    edges = {}
    for i in range(num_edges):
        nid1, nid2 = random.sample(np.arange(num_nodes).tolist(), 2)
        if not multi:
            if (nid1, nid2) in conns or (nid2, nid1) in conns:
                continue
        edge = g.Edge(i, nodes[nid1], nodes[nid2], data=util.compute_view_number(nodes[nid1], nodes[nid2]))
        conns[(nid1, nid2)] = i
        edges[i] = edge

    return g.Graph(edges)

def test_partition(graph):

    templates = [t.StarTemplate, t.ThreeNodeTemplate, t.SingletonTemplate]
    results, _ = graph.partition_by_templates(templates, super_node_class=SimpleSuperNode, super_edge_class=SimpleSuperEdge)
    ax = plt.gca()
    for template in templates:
        sg = results[template.__name__]
        sg.visualize(ax, list(g.nodes.keys()), dotsize=5*(2+template.size()))
        plt.autoscale()
        ax.set_aspect('equal', 'box')

def test_partition_edges(graph):
    templates = [t.ThreeRelTemplate, t.SingleRelTemplate, t.SingleTemplate, t.RelTemplate]
    results, _ = graph.partition_by_templates(templates, super_node_class=SimpleSuperNode, super_edge_class=SimpleSuperEdge)
    ax = plt.gca()
    for template in templates:
        sg = results[template.__name__]
        sg.visualize(ax, list(g.nodes.keys()), dotsize=5*(2+template.size()))
        plt.autoscale()
        ax.set_aspect('equal', 'box')    
    

if __name__ == "__main__":
    # g = setup_fixed()#(30, 40, multi=True)

    g = build_graph("graph_example.graph", simple_interpret_node, simple_interpret_edge)
    
    # g = setup_random(100, 2000, multi=True)    

    ax = plt.gca()
    g.visualize(ax, list(g.nodes.keys()))
    plt.autoscale()
    ax.set_aspect('equal', 'box')
    test_partition(g)
    # test_partition_edges(g)
    plt.show()        
    
