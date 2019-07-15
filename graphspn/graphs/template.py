# Templates.
#
# author: Kaiyu Zheng
from abc import ABC, abstractmethod
import random
from collections import deque
import itertools
import graphspn.graphs.graph as g

########################################
# Template
########################################
class Template(ABC):

    @classmethod
    @abstractmethod
    def size(cls):
        """
        A template has a defined size. (e.g. number of nodes/edges). 
        Useful for sorting template by complexity.
        """
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        """
        An integer that identifies the type of this template
        """
        pass

    @classmethod
    def templates_for(cls, symbol):
        if symbol.upper() == "THREE":
            return [ThreeNodeTemplate, PairTemplate, SingletonTemplate]
        elif symbol.upper() == "VIEW":
            return [ThreeRelTemplate, SingleRelTemplate, SingleTemplate, RelTemplate]
        elif symbol.upper() == "STAR":
            return [StarTemplate, ThreeNodeTemplate, PairTemplate, SingletonTemplate]
        else:
            raise Exception("Unrecognized symbol for templates: %s" % symbol)

    @classmethod
    def get_type(cls, template):
        if template == ThreeNodeTemplate:
            return "three"
        elif template == StarTemplate:
            return "star"
        elif template == ThreeRelTemplate:
            return "view"
        else:
            raise Exception("Unrecoginzed template to get type: %s" % template.__name__)

    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        """
        pivot_edge (Edge): the edge that was sampled from which the match search starts.
        pivot (Node): the node that was sampled from the pivot_edge from which the match search starts.
        """
        pass

########################################
# NodeTemplate
########################################
class NodeTemplate(Template):

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass

        
    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        pass


    @classmethod
    def size(cls):
        return cls.num_nodes()

    @classmethod
    def code(cls):
        return 0

    has_edge_info = False

    @classmethod
    def size_to_class(num_nodes):
        m = {
            SingletonTemplate.num_nodes: SingletonTemplate,
            PairTemplate.num_nodes: PairTemplate,
            ThreeNodeTemplate.num_nodes: ThreeNodeTemplate,
            StarTemplate.num_nodes: StarTemplate
        }
        return m[num_nodes]


class SingletonTemplate(NodeTemplate):
    """
    Single node
    """
    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        # Hardcode the match check
        if pivot.id not in excluded_nodes:
            return g.OrderedEdgeNodeSet([pivot], [])

class PairTemplate(NodeTemplate):
    """
    Simple pair
    """
    @classmethod
    def num_nodes(cls):
        return 2

    @classmethod
    def match(cls, graph, P, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):    
        """
        Returns an ordered edge node set that follows A-P or P-A.
        """
        # Hardcode the match check
        pivot_neighbors = graph.neighbors(P.id)

        for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
            if A_id not in excluded_nodes and A_id != P.id:
                edge = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                return g.OrderedEdgeNodeSet([P, graph.nodes[A_id]], [edge])
        return None


class StarTemplate(NodeTemplate):

    """
          A
          |
      B - P - C
          |
          D
    """
    @classmethod
    def num_nodes(cls):
        return 5
    
    @classmethod
    def match(cls, graph, X, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):        
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)

            #A
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    #B
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            #C
                            for C_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                if C_id not in excluded_nodes | set({A_id, B_id}):
                                    #D
                                    for D_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                        if D_id not in excluded_nodes | set({A_id, B_id, C_id}):
                                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]
                                            edge_PC = graph.edges[random.sample(graph.edges_between(P.id, C_id), 1)[0]]
                                            edge_PD = graph.edges[random.sample(graph.edges_between(P.id, D_id), 1)[0]]
                                            return g.OrderedEdgeNodeSet([graph.nodes[A_id], graph.nodes[B_id], P, graph.nodes[C_id], graph.nodes[D_id]],
                                                                        [edge_PA, edge_PB, edge_PC, edge_PD])
                                                                            
        subgraph = match_by_pivot(X, excluded_nodes=excluded_nodes)
        return subgraph    # subgraph could be None
    

    
class ThreeNodeTemplate(NodeTemplate):

    """
    Simple three node structure
    
    A--(P)--B

    P is the pivot. A and B are not connected
    """

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def match(cls, graph, X, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):        
        """
        Returns a list of node ids that are matched in this template. Order
        follows A-P-B, where P is the pivot node. X is the node where the
        matching starts, but not necessarily the pivot.
        """
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            # if not relax and A_id in graph.neighbors(B_id):
                            #     continue
                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]
                            return g.OrderedEdgeNodeSet([graph.nodes[A_id], P, graph.nodes[B_id]],
                                                         [edge_PA, edge_PB])


        subgraph = match_by_pivot(X, excluded_nodes=excluded_nodes)
        return subgraph


class EdgeRelTemplate(Template):
    
    @classmethod
    @abstractmethod
    def num_edges(cls):
        pass

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass
        
    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        pass


    @classmethod
    def size(cls):
        return cls.num_edges() + cls.num_nodes()

    @classmethod
    def code(cls):
        return 1

    has_edge_info = False

    @classmethod
    def size_to_class(num_nodes, num_edges):
        m = {
            # SingletonTemplate.num_nodes: SingletonTemplate,
            # PairTemplate.num_nodes: PairTemplate,
            # ThreeNodeTemplate.num_nodes: ThreeNodeTemplate,
            # StarTemplate.num_nodes: StarTemplate
        }
        return m[num_nodes]



class ThreeRelTemplate(EdgeRelTemplate):

    @classmethod
    def num_edges(cls):
        return 2

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        
        def match_by_pivot(P, excluded_nodes=set({}), excluded_edges=set({})):
            pivot_neighbors = graph.neighbors(P.id)
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            # if not relax and A_id in graph.neighbors(B_id):
                            #     continue
                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]
                            
                            if edge_PA.id not in excluded_edges and edge_PB.id not in excluded_edges:
                                return g.OrderedEdgeNodeSet([graph.nodes[A_id], P, graph.nodes[B_id]],
                                                            [edge_PA, edge_PB])                                
        return match_by_pivot(pivot, excluded_nodes=excluded_nodes, excluded_edges=excluded_edges)

class SingleRelTemplate(EdgeRelTemplate):

    @classmethod
    def num_edges(cls):
        return 1

    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot is not None and pivot.id not in excluded_nodes\
           and pivot_edge.id not in excluded_edges:
            return g.OrderedEdgeNodeSet([pivot], [pivot_edge])

class RelTemplate(EdgeRelTemplate):
    @classmethod
    def num_edges(cls):
        return 1
    
    @classmethod
    def num_nodes(cls):
        return 0

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot_edge.id not in excluded_edges:
            return g.OrderedEdgeNodeSet([], [pivot_edge])

class SingleTemplate(EdgeRelTemplate):
    @classmethod
    def num_edges(cls):
        return 0

    @classmethod
    def num_nodes(cls):
        return 1
    
    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot is not None and pivot.id not in excluded_nodes:
            return g.OrderedEdgeNodeSet([pivot], [])
