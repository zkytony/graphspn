# Implementation of topological maps.
# Use python 3+
#
# author: Kaiyu Zheng

# Topological
import numpy as np
import random
import yaml
import os
import sys
import graphspn.util as util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
import copy
import math
import re
import itertools
from collections import deque

import graphspn.graphs.graph as g
from graphspn.util import ABS_DIR

class CategoryManager:

    # Invariant: categories have an associated number. The number
    #            goes from 0 to NUM_CATEGORIES - 1

    # SKIP_UNKNOWN is True if we want to SKIP unknown classes, which means all remaining
    # classes should be `known` and there is no 'UN' label.
    SKIP_UNKNOWN = True
    # TYPE of label mapping scheme. Offers SIMPLE and FULL.
    TYPE = None
    
    # Category mappings
    def load_mapping(name):
        MAPPING_DIR= os.path.join(ABS_DIR, "experiments/dataset/categories")
        with open(os.path.join(MAPPING_DIR, "%s.yaml" % name.lower())) as f:
            return yaml.safe_load(f)
            
    CAT_MAP_ALL = {
        'BINARY': load_mapping("binary"),
        'SIMPLE': load_mapping("simple"),
        'FULL': load_mapping("full"),
        'SIX': load_mapping("six_stockholm"),  # 1PO and 2PO combined
        'SEVEN': load_mapping("seven_stockholm"),
        'TEN': load_mapping("ten_stockholm")
    }

    CAT_MAP = None
    CAT_REV_MAP = None
    CAT_COLOR = None
    PLACEHOLDER_COLOR = '#b8b8b8'
    REGULAR_SWAPS = None
    
    @staticmethod
    def init():
        if CategoryManager.TYPE is not None:
            CategoryManager.CAT_MAP = CategoryManager.CAT_MAP_ALL[CategoryManager.TYPE]['FW']
            CategoryManager.CAT_REV_MAP = CategoryManager.CAT_MAP_ALL[CategoryManager.TYPE]['BW']
            CategoryManager.CAT_COLOR = CategoryManager.CAT_MAP_ALL[CategoryManager.TYPE]['CL']
            CategoryManager.PLACEHOLDER_COLOR = '#b8b8b8'
            if "regular_swaps" in CategoryManager.CAT_MAP_ALL[CategoryManager.TYPE]:
                CategoryManager.REGULAR_SWAPS = CategoryManager.CAT_MAP_ALL[CategoryManager.TYPE]['regular_swaps']

            if CategoryManager.SKIP_UNKNOWN:
                CategoryManager.NUM_CATEGORIES = len(CategoryManager.CAT_COLOR) - 2  # exclude '-1' and catg_num('UN')
            else:
                CategoryManager.NUM_CATEGORIES = len(CategoryManager.CAT_COLOR) - 1  # exclude '-1'.
    
    @staticmethod
    def category_map(category, rev=False, checking=False):
        """
        Return a number for a given category

        `rev` is True if want to return a category string for a given number
        `checking` is true if it is allowed to return 'UN' or its number, ignoring what is set
                   in SKIP_UNKNOWN. Namely, if SKIP_UNKNOWN is True, this function won't return
                   'UN' or its number, unless `checking` is True. This is useful for checking
                   if some node should be skipped when loading from the database, or when you
                   don't care if the category can be novince (i.e. unknown).
        """
        # If we 'skip unknown', and the passed in `category` is 'unknown' to our
        # category mapping, then the category map is missing something. Throw
        # an exception.
        if not checking and CategoryManager.SKIP_UNKNOWN:
            if not rev:  # str -> num
                if category not in CategoryManager.CAT_MAP or CategoryManager.CAT_MAP[category] == CategoryManager.CAT_MAP['UN']:
                    raise ValueError("Unknown category: %s" % category)
            else:  # num -> str
                if category not in CategoryManager.CAT_REV_MAP or CategoryManager.CAT_REV_MAP[category] == 'UN':
                    raise ValueError("Unknown category number: %d" % category)
        if not rev:
            if category in CategoryManager.CAT_MAP:
                return CategoryManager.CAT_MAP[category]
            else:
                if not checking:
                    assert CategoryManager.SKIP_UNKNOWN == False
                return CategoryManager.CAT_MAP['UN']
        else:
            if category in CategoryManager.CAT_REV_MAP:
                return CategoryManager.CAT_REV_MAP[category]
            else:
                if not checking:
                    assert CategoryManager.SKIP_UNKNOWN == False
                return 'UN'

    @staticmethod
    def canonical_category(catg, checking=False):
        """
        Returns the canonical category for category `catg`.

        Args:
          `catg` needs to be a string.

        Returns the string abbreviation of the canonical category.
        """
        # First, catg->num. Then num->cano_catg
        return CategoryManager.category_map(
            CategoryManager.category_map(catg, checking=checking),
            rev=True, checking=checking)
           

    @staticmethod
    def category_color(category):
        catg_num = CategoryManager.category_map(category, checking=True)
        if catg_num in CategoryManager.CAT_COLOR:
            return CategoryManager.CAT_COLOR[catg_num]
        else:  # unknown
            return CategoryManager.CAT_COLOR[CategoryManager.category_map('UN')]
        

    @staticmethod
    def known_categories():
        """Known categories should be the canonical categories besides 'unknown' and 'occluded'"""
        return [CategoryManager.category_map(k, rev=True)
                for k in range(CategoryManager.NUM_CATEGORIES)]

    @staticmethod
    def novel_categories():
        """Novel categories are all represented by the 'unknown' class"""
        return ['UN']

    @staticmethod
    def is_novel_swap(class1, class2):
        """Given a swap of class1 and class2 (strings), return True if this swap is resulting in a
        novel structure."""
        if CategoryManager.REGULAR_SWAPS is not None:
            return not ([class1, class2] in CategoryManager.REGULAR_SWAPS\
                        or [class2, class1] in CategoryManager.REGULAR_SWAPS)
        else:
            raise ValueError("Operation not possible. Did not specify regular class swaps.")


class ColdDatabaseManager:
    # Used to obtain cold database file paths

    def __init__(self, db_name, db_root, gt_root=None):
        self.db_root = db_root
        self.db_name = db_name
        s = re.search("(Stockholm|Freiburg|Saarbrucken)", db_name)
        if s is not None:
            # We know we may append floor number(s) after db_name. But we may
            # also use some custom db_name which could also have groundtruth
            # files.
            self.db_name = s.group()
        self.gt_root = gt_root

    def groundtruth_file(self, floor, filename):
        if self.gt_root is None:
            return os.path.join(self.db_root, self.db_name, 'groundtruth', floor, filename)
        else:
            return os.path.join(self.gt_root, self.db_name, 'groundtruth', floor, filename)


########################################
#  Node
########################################
class PlaceNode(g.Node):

    def __init__(self, id, placeholder, pose, anchor_pose, label, room):
        """
        Args:
        
        id (int): id for this node
        placeholder (bool): True if this node is a placeholder
        pose (tuple): a tuple (x, y) of the node's pose in the topo map.
        anchor_pose (tuple): a tuple (x, y) of the node's anchored grid cell's pose
                             in the topo map.
        label (str): category of this place (e.g. DW).
        """
        self.id = id
        self.placeholder = placeholder
        self.pose = pose
        self.anchor_pose = anchor_pose
        self.label = label
        self.room = room

    @property
    def label_num(self):
        return CategoryManager.category_map(self.label)

    def vscan(self, normalize=True):
        
        def normalize(polar_grid):
            vmap = {0: 0,
                    205: 1,
                    254: 2}
            for x in np.nditer(polar_grid, op_flags=['readwrite']):
                x[...] = vmap[int(x)]
            return polar_grid

        if self.polar_vscan is not None:
            return normalize(self.polar_vscan.grid.flatten())
        
    @property
    def coords(self):
        return self.anchor_pose

    @property
    def color(self):
        return CategoryManager.category_color(self.label)


class CompoundPlaceNode(g.SuperNode):
    """
    Does not inherit PlaceNode because there is no well-defined attributes such as vscan for
    a CompoundPlaceNode.
    """

    def __init__(self, id, enset):
        """
        nodes: a list of nodes included by this CompoundNode. Need to keep the order
        """
        super().__init__(id, enset)
        self.nodes = self.enset.nodes
        self._coords = (
            sum(self.enset.nodes[nid].coords[0] for nid in self.enset.nodes) / len(self.enset.nodes),
            sum(self.enset.nodes[nid].coords[1] for nid in self.enset.nodes) / len(self.enset.nodes)
        )
        self.anchor_pose = self._coords  # for the sake of having 'anchor_pose' field.
        self.label = 0

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
        return self._coords

    def contains(self, node):
        return node.id in self.nodes

    def __repr__(self):
        return "CompoundPlaceNode(%d){%s}" % (self.id, self.nodes)

    def to_catg_list(self):
        category_form = []
        for nid in self.nodes:
            n = self.nodes[nid]
            if isinstance(n, PlaceNode) or isinstance(n, CompoundLabeledPlaceNode):
                category_form.append(CategoryManager.category_map(n.label))
            elif isinstance(n, CompoundPlaceNode):
                category_form += n.to_catg_list()
        return category_form

    def to_vscans_list(self):
        vscan_form = []
        for nid in self.nodes:
            n = self.nodes[nid]
            if isinstance(n, PlaceNode):
                vscan_form.append(n.vscan())
            elif isinstance(n, CompoundPlaceNode):
                vscan_form += n.to_vscans_list()
        return vscan_form

    def to_place_id_list(self):
        pid_form = []
        for nid in self.nodes:
            n = self.nodes[nid]
            if isinstance(n, PlaceNode) or isinstance(n, CompoundLabeledPlaceNode):
                pid_form.append(n.id)
            elif isinstance(n, CompoundPlaceNode):
                pid_form += n.to_place_id_list()
        return pid_form
        
    def to_catg_ndarray(self):
        return np.array(self.to_list, dtype=int)

    def count(self):
        c = 0
        for n in self.nodes:
            c += n.count()
        return c

    
class CompoundLabeledPlaceNode(CompoundPlaceNode):
    """
    Same as CompoundPlaceNode except it has a `label` attribute. Used for graph segmentation.
    """
    def __init__(self, id, nodes, label):
        super().__init__(id, nodes)
        self.label = label



########################################
#  Edge
########################################
class TopoEdge(g.Edge):
    """
    Edge with view numbers
    """    
    
    def __init__(self, id, node1, node2, view_nums=None):
        """
        The id is expected to be unique in the graph edges.
        """
        # view numbers
        if view_nums is None:
            if node1 is not None and node2 is not None:
                self.view_nums = (util.compute_view_number(node1, node2),
                                  util.compute_view_number(node2, node1))
        else:
            assert len(view_nums) == 2
            self.view_nums = view_nums
        super().__init__(id, node1, node2, data=view_nums)

    @classmethod
    def pick_id(cls, supernode1, supernode2, existing_ids=set({}), edge=None):
        """
        that always returns the same id if given the same subgraph, and different
        if otherwise.
        """
        if len(existing_ids) == 0:
            return 0        
        return max(existing_ids) + 1        
            
    def __repr__(self):
        return "#%d[%d<%d>---%d<%d>]" % (self.id, self.nodes[0].id, self.view_nums[0],
                                         self.nodes[1].id, self.view_nums[1])


    def __eq__(self, other):
        """
        Two edges are equal if node1 and node2 have equal ids respectively (without order).
        """
        my_ids = set({self.nodes[0].id, self.nodes[1].id})
        other_ids = set({other.nodes[0].id, other.nodes[1].id})
        return my_ids == other_ids


    def __hash__(self):
        return hash((self.nodes[0].id, self.nodes[1].id))

    @classmethod
    def get_triplet_from_edge_pair(cls, topo_map, edge_pair, center_nid, catg=False):
        e1_nids = (edge_pair[0].nodes[0].id, edge_pair[0].nodes[1].id)
        e2_nids = (edge_pair[1].nodes[0].id, edge_pair[1].nodes[1].id)
        nid1 = e1_nids[1-e1_nids.index(center_nid)]
        nid2 = e2_nids[1-e2_nids.index(center_nid)]
        if catg:
            return (CategoryManager.category_map(topo_map.nodes[nid1].label),
                    CategoryManager.category_map(topo_map.nodes[center_nid].label),
                    CategoryManager.category_map(topo_map.nodes[nid2].label))
        else:
            return (nid1, center_nid, nid2)

        
class TopoSuperEdge(g.SuperEdge):
    
    def __init__(self, id, supernode1, supernode2):
        """
        id (int) is for this node
        enset (Graph) underlying graph structure
        """
        super().__init__(id, supernode1, supernode2)

    @classmethod
    def pick_id(cls, supernode1, supernode2, existing_ids=set({}), edge=None):
        """
        that always returns the same id if given the same enset, and different
        if otherwise.
        """
        if len(existing_ids) == 0:
            return 0        
        return max(existing_ids) + 1

    @property
    def color(self):
        return "black"


##############################################
#  TopologicalMap
##############################################
class TopologicalMap(g.Graph):

    """
    A TopologicalMap is a undirected graph. It can be constructed by a predefined set of nodes and a set of
    node connectivities. It supports partitioning by templates, and segmentation into less granularity of nodes.
    Functionality to modify the graph is not provided. Each node in a TopologicalMap is a PlaceNode, or a
    CompoundPlaceNode (see these classes below).
    """

    def __init__(self, edges, **kwargs):
        """
        Initializes a topological map from given nodes.

        @param nodes is a dictionary that maps from node id to actual node
        @param conns is a dictionary that maps from a node id to a set of tuples (neighbor_node_id, view_number)
        """
        super().__init__(edges, directed=False)        

        # Used when resetting the labels.
        self.__catg_backup = {nid:self.nodes[nid].label for nid in self.nodes}
    
        
    #--- Basic graph operations ---#
    def hard_count(self):
        """Count the number of place nodes in this topological map"""
        c = 0
        for nid in self.nodes:
            c += self.nodes[nid].count()
        return c

    def num_placeholders(self):
        """Returns the number of placeholders in this map"""
        return sum(1 for nid in self.nodes if self.nodes[nid].placeholder)

    def connected_edge_pairs(self):
        """
        Returns a dictionary from nid to set of all combinations of edge pairs.
        """
        node_edge_pairs = {}
        for nid in self.nodes: 
            neighbors = self.neighbors(nid)
            edges = set({})
            for nnid in neighbors:
                edges.add(self.edges_between(nid, nnid))
            pairs = set(itertools.combinations(edges, 2))  # order does not matter
            node_edge_pairs[nid] = pairs
        return node_edge_pairs


    #-- High level graph properties --#
    
    def segment(self, remove_doorway=True):
        """
        Segments this topological map into a new TopologicalMap object where each node is a CompoundLabeledPlaceNode that
        includes nodes of the same label (supposedly) in the same room. This is done by sampling nodes and BFS from 
        sampled nodes to "flood" the graph.

        If `remove_doorway` is true, before segmenting the graph, all doorway nodes are replaced by a node which has class
        most common in the neighbors of the doorway node.

        ASSUME the nodes in `self` are all PlaceNode objects.
        """
        copy_map = self.copy()
        if remove_doorway:
            for nid in copy_map.nodes:
                if copy_map.nodes[nid].label == 'DW':
                    votes = [0]*CategoryManager.NUM_CATEGORIES
                    for nnid in copy_map.neighbors(nid):
                        votes[copy_map.nodes[nnid].label_num] += 1
                    copy_map.nodes[nid].label = CategoryManager.category_map(votes.index(max(votes)), rev=True)
        
        to_cover = set(copy_map.nodes.keys())
        nodes_new = {}
        conns_new = {}
        nn_map = {} # map from topo map node's id to a node's id in the segmented graph.
        while len(to_cover) > 0:
            start_nid = random.sample(to_cover, 1)[0]
            """BFS from start_nid"""
            q = deque()
            q.append(start_nid)
            same_label_nodes = [start_nid]
            visited = set({start_nid})
            while len(q) > 0:
                nid = q.popleft()
                neighbors = copy_map.neighbors(nid)
                for neighbor_nid in neighbors:
                    if neighbor_nid not in visited:
                        visited.add(neighbor_nid)
                        if copy_map.nodes[neighbor_nid].label_num == copy_map.nodes[start_nid].label_num:
                            same_label_nodes.append(neighbor_nid)
                            q.append(neighbor_nid)
            compound_node = CompoundLabeledPlaceNode(util.pick_id(nodes_new.keys(),
                                                                  sum(n for n in same_label_nodes) % 211),
                                                     list(copy_map.nodes[n] for n in same_label_nodes), copy_map.nodes[start_nid].label)
            nodes_new[compound_node.id] = compound_node
            """Remove covered nodes"""
            to_cover -= set(same_label_nodes)
            
            """Form connections"""
            for nid in same_label_nodes:
                for neighbor_nid in copy_map.neighbors(nid):
                    if nn_map.get(neighbor_nid) is not None:
                        # The neighbor is indeed already mapped to a new node in the segmented graph.
                        new_node_neighbor = nodes_new[nn_map.get(neighbor_nid)]   # new node in the partitioned graph
                        if new_node_neighbor.id != compound_node.id: # Make sure we are not adding self edges in the segmented graph
                            util.sure_add(conns_new, new_node_neighbor.id, (compound_node.id, util.compute_view_number(compound_node, new_node_neighbor)))
                            util.sure_add(conns_new, compound_node.id, (new_node_neighbor.id, util.compute_view_number(new_node_neighbor, compound_node)))
                nn_map[nid] = compound_node.id

        return TopologicalMap(nodes_new, conns_new)
    
            
    #--- Masking the graph ---#

    # Functions for running experiments
    def occlude_placeholders(self):
        """
        Sets the labels of nodes that are placeholders to be -1.
        """
        catg_map = {
            nid:CategoryManager.category_map(self.nodes[nid].label)
            if not self.nodes[nid].placeholder else CategoryManager.category_map('OC')
            for nid in self.nodes
        }
        self.assign_categories(catg_map)


    def swap_classes(self, swapped_classes):
        """
        swapped_classes (tuple): a tuple of two category names (str), which will be swapped.
        """
        catg_map = self.current_category_map()
        c1, c2 = CategoryManager.category_map(swapped_classes[0]), \
                 CategoryManager.category_map(swapped_classes[1])
        for nid in catg_map:
            if catg_map[nid] == c1:
                catg_map[nid] = c2
            elif catg_map[nid] == c2:
                catg_map[nid] = c1
        self.assign_categories(catg_map)
        return catg_map
        

    def current_category_map(self):
        """
        Returns a dictionary from node id to numerical value of the category.
        """
        catg_map = {
            nid:CategoryManager.category_map(self.nodes[nid].label)
            for nid in self.nodes
        }
        return catg_map


    def mask_by_policy(self, policy, **kwargs):
        """
        Sets node label to -1 according to `policy`, which is a function that determines if
        the node should be masked. It takes two parameters: topo_map and node. The policy
        is applied to nodes in random order.

        Note that the policy executed later will receive the information of formerly occluded graph nodes.

        kwargs: arguments for the policy.
        """
        nids = random.sample(list(self.nodes.keys()), len(self.nodes))
        for nid in nids:
            if policy(self, self.nodes[nid], **kwargs):
                self.nodes[nid].label = CategoryManager.category_map(-1, rev=True)

    
    def assign_categories_by_grid(self, cat_grid, placegrid):
        """
        Change the category of this topological map according to a category grid (cat_grid).
        And placegrid is the corresponding PlaceGrid object for this category grid.
        """
        h, w = cat_grid.shape
        for y in range(h):
            for x in range(w):
                mapped_places = placegrid.places_at(y, x)
                for p in mapped_places:
                    self.nodes[p.id].label = util.CatgoryManager.category_map(cat_grid[y, x], rev=True)


    def assign_categories(self, categories_map):
        """
        categories_map is a dictionary from node id to the numerical value of the category.
        """
        for nid in categories_map:
            if nid in self.nodes:
                self.nodes[nid].label = CategoryManager.category_map(categories_map[nid], rev=True)


    def reset_categories(self):
        for nid in self.__catg_backup:
            self.nodes[nid].label = self.__catg_backup[nid]


    #--- Visualizations ---#
    def visualize(self, ax, canonical_map_yaml_path=None, included_nodes=None,
                  dotsize=13, linewidth=1.0,
                  img=None, consider_placeholders=False, show_nids=False):
        """Visualize the topological map `self`. Nodes are colored by labels, if possible.
        If `consider_placeholders` is True, then all placeholders will be colored grey.
        Note that the graph itself may or may not contain placeholders and `consider_placholders`
        is not used to specify that."""
        # Open the yaml file
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.safe_load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
        plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")

        h, w = img.shape
        util.zoom_rect((w/2, h/2), img, ax, h_zoom_level=3.0, v_zoom_level=2.0)
        
        # Plot the nodes
        for nid in self.nodes:
            if included_nodes is not None and nid not in included_nodes:
                continue

            nid_text = str(nid) if show_nids else None
                
            place = self.nodes[nid]
            node_color = CategoryManager.category_color(place.label) \
                         if not (consider_placeholders and place.placeholder) \
                            else CategoryManager.PLACEHOLDER_COLOR
            pose_x, pose_y = place.pose  # gmapping coordinates
            plot_x, plot_y = util.plot_dot_map(ax, pose_x, pose_y, map_spec, img,
                                               dotsize=dotsize, color=node_color, zorder=2,
                                               linewidth=linewidth, edgecolor='black', label_text=nid_text)

            # Plot the edges
            for neighbor_id in self._conns[nid]:
                if included_nodes is not None and neighbor_id not in included_nodes:
                    continue

                util.plot_line_map(ax, place.pose, self.nodes[neighbor_id].pose, map_spec, img,
                                   linewidth=1, color='black', zorder=1)


                
    def visualize_partition(self, ax, node_ids, canonical_map_yaml_path, ctype=1,
                            alpha=0.8, dotsize=6, linewidth=3, img=None):
        """
        node_ids is the list of tuples where each represents node ids on a template.

        if `img` is not None, assume the visualization of the topological map is already plotted.
        """

        # First, visualize the graph in a whole.
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.safe_load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
            plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")
            self.visualize(ax, canonical_map_yaml_path=canonical_map_yaml_path, img=img, dotsize=13)
        
        colors = set({})
        for tmpl_node_ids in node_ids:
            color = util.random_unique_color(colors, ctype=ctype)
            colors.add(color)
            # Plot dots
            for nid in tmpl_node_ids:
                rx, ry = self.nodes[nid].pose
                px, py = util.transform_coordinates(rx, ry, map_spec, img)
                very_center = util.plot_dot_map(ax, rx, ry, map_spec, img, dotsize=dotsize,
                                                color=color, zorder=4, linewidth=1.0, edgecolor='black')

            # Plot edges
            for nid in tmpl_node_ids:
                for mid in tmpl_node_ids:
                    # Plot an edge
                    if nid != mid:
                        if self.edges_between(nid, mid):
                            util.plot_line_map(ax, self.nodes[nid].pose, self.nodes[mid].pose, map_spec, img,
                                               linewidth=linewidth, color=color, zorder=3)

        return img
