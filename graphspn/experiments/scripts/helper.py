import graphspn.graphs.graph as g

## Classes used for graph partitioning
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
    
