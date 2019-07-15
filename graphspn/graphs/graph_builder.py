# author: Kaiyu Zheng

import numpy as np
import graphspn.util as util
from graphspn.graphs.graph import Graph

def build_graph(graph_file_path, func_interpret_node, func_interpret_edge):
    """
    Reads a file that is a specification of a graph, then construct
    a graph object using the TopologicalMap class. The TopologicalMap
    class is for undirected graphs, where each node on the graph contains
    a fixed label and edges do not have labels. It is possible for
    nodes to have uncertain labels.
    
    This script reads a file of format ".ug" that indicates "undirected graph".
    The format is:


    <Node_Id> <attr1> <attr2> ...
    --
    <Edge_Id> <Node_Id_1> <Node_Id_2> <attr1> <attr2> ...
    --
    Undirected
    
    The first part specifies the nodes. It is not required that Node_Id
    starts from 0. Each node may have a list of attributes, which are
    interpreted using the given `func_interpret_node` function, that takes
    as input node id, [<attr1>, <attr2> ...], and returns an
    object of (sub)class of Node.
    
    The second part specifies the edges. The edge is undirected, and
    the node ids should be defined in the first part of the file.
    Each edge may have a list of attributes, which are
    interpreted using the given `func_interpret_edge` function, that takes
    as input the edge id, node1, node2 (objects) and [<attr1>, <attr2> ...], and
    returns an object of (sub)class of Node. For simplicity, you can omit <Edge_Id>
    by ":" which causes edge_id to be incremental from 0.

    Lastly, one can specify whether the graph is "directed" or "undirected".
    If unspecified, assume to be "undirected".

    There could be arbitrarily many empty lines, and can have comments
    by beginning the line with "#"
    
    This function can be used to parse a graph file and generate a Graph
    object, be used in GraphSPN experiments.
    """
    with open(graph_file_path) as f:
        lines = f.readlines()

    nodes = {}  # Map from node id to an actual node object
    edges = {}  # Map from edge id to an actual edge object
    use_log = None
    
    directed = False
        
    state = "nodes"
    for i, line in enumerate(lines):
        # Handle transition, if encountered
        try:
            line = line.rstrip()
            if len(line) == 0:
                continue # blank line
            if line.startswith("#"):
                continue # comment

            if line == "--":
                state = _next_state(state)
                continue # read next line
            # This line belongs to a state
            elif state == "nodes":
                tokens = line.split()  # split on whitespaces
                nid = int(tokens[0])  # split on whitespaces
                if nid in nodes:
                    raise ValueError("Node %d is already defined" % (nid))
                nodes[nid] = func_interpret_node(nid, tokens[1:])
            elif state == "edges":
                tokens = line.split()  # split on whitespaces
                if tokens[0] == ":":
                    eid = len(edges)
                else:
                    eid = int(tokens[0])
                nid1, nid2 = int(tokens[1]), int(tokens[2])
                if eid in edges:
                    raise ValueError("Edge %d is already defined" % (eid))
                if nid1 not in nodes:
                    raise ValueError("Node %d is undefined" % nid1)
                if nid2 not in nodes:
                    raise ValueError("Node %d is undefined" % nid2)

                edges[eid] = func_interpret_edge(eid, nodes[nid1], nodes[nid2], tokens[3:])
                
            elif state == "graph_type":
                directed = False if line.startswith("Undirected") else True
            else:
                raise ValueError("Unexpected state %s" % state)
        except Exception as e:
            print("Line %d caused an Error:" % i)
            print(e)
            raise e
        
    return Graph(edges, directed=directed) # We are done

#####################
# Utility functions #
#####################
def _next_state(state):
    if state == "nodes":
        return "edges"
    elif state == "edges":
        return "graph_type"  
    else:
        raise ValueError("Unexpected state %s" % state)
