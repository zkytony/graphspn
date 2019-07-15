# Manager for processing topological map dataset
#
# author: Kaiyu Zheng

import matplotlib
import matplotlib.pyplot as plt

from graphspn.experiments.scripts.topo_map import PlaceNode, TopologicalMap, TopoEdge, CompoundPlaceNode
from graphspn.graphs.template import Template
from graphspn.experiments.scripts.topo_map import CategoryManager
from graphspn.util import compute_view_number
import os, sys, re
import numpy as np
import csv
import math
import random
from collections import deque
import json
import copy


DEBUG = False

class TopoMapDataset:

    def __init__(self, db_root):
        """
        db_root: path to root db. Database should be in 'db_root/db_name' directory.
        """
        self.db_root = db_root
        self._topo_maps_data = {}  # {db_name: {seq_id: topo_map}}


    # TODO: This shouldn't really be here.  It's a more general function for the experiments.
    def get_dbname_info(self, db_name, train=True):
        """Given a db_name (str), e.g. Stockholm456, return a tuple ('Stockholm', '456', 7}.
        If `train` is False, then regard the floor information (i.e. 456 in this example) as
        testing floors and thus will return ('Stockholm', 7, '456')"""
        # Example: Stockholm456
        building = re.search("(stockholm|freiburg|saarbrucken)", db_name, re.IGNORECASE).group().capitalize()
        given_floors = db_name[len(building):]
        if building == "Stockholm":
            remaining_floors = {4, 5, 6, 7}
        elif building == "Freiburg":
            remaining_floors = {1, 2, 3}
        elif building == "Saarbrucken":
            remaining_floors = {1, 2, 3, 4}
        for f in given_floors:
            remaining_floors = remaining_floors - {int(f)}
        remaining_floors = "".join(map(str, (sorted(remaining_floors))))
        if train:
            return building, given_floors, remaining_floors
        else:
            return building, remaining_floors, given_floors

        
    def create_template_dataset(self, templates, num_partitions=10,
                                db_names=None, seq_ids=None, seqs_limit=-1):
        """
        Return a dataset of samples that can be used to train template SPNs. This
        dataset contains symmetrical data, i.e. for every pair of semantics, its
        reverse is also present in the dataset.
        If `db_names` and `seq_ids` are both None, load from all db_names. If both are not None,
        will treat `seq_ids` as None. `seq_ids` should be a list of "{db}-{seq_id}" strings.
        Return format:
           {K:V}, K is database name, V is -->
            --> NxM list, where N is the number of data samples, and M is the number of
           nodes in the templates provided. For example, with 3-node template, each
           data sample would be a list [a, b, c] where a, b, c are the category numbers
           for the nodes on the template.
        """
        samples = {tmpl.__name__:{} for tmpl in templates}
        total_seqs_count = 0
        if db_names is None and seq_ids is None:
            db_names = self._topo_maps_data.keys()

        topo_maps = {}  # map from "{db}-{seq_id}" to topo map
        if db_names is not None:
            for db_name in db_names:
                for tmpl in templates:
                    samples[tmpl.__name__][db_name] = []
                for seq_id in self._topo_maps_data[db_name]:
                    topo_maps[db_name+"-"+seq_id] = self._topo_maps_data[db_name][seq_id]

        else:  # seq_ids must not be None
            for db_seq_id in seq_ids:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                topo_maps[db_seq_id] = self._topo_maps_data[db_name][seq_id]
                if db_name not in samples:
                    for tmpl in templates:
                        samples[tmpl.__name__][db_name] = []                    
                
        for kk in range(num_partitions):
            for db_seq_id in topo_maps:
                db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
                supergraph = topo_maps[db_seq_id]

                for tmpl in templates:
                    supergraph = supergraph.partition(tmpl,
                                                      super_node_class=CompoundPlaceNode,
                                                      super_edge_class=TopoEdge)

                    for n in supergraph.nodes:
                        template_sample = supergraph.nodes[n].to_catg_list()
                        samples[tmpl.__name__][db_name].append(template_sample)
                        samples[tmpl.__name__][db_name].append(list(reversed(template_sample)))

                total_seqs_count += 1

                if seqs_limit > 0 and total_seqs_count >= seqs_limit:
                    return samples
        return samples                

    
    def get_topo_maps(self, db_name=None, seq_id=None, amount=1):
        """
        Returns a dictionary of seq_id to topo map.
        
        If `amount` is -1, get all with db_name.
        """
        if db_name is None:
            db_name = random.sample(self._topo_maps_data.keys(), 1)[0]
        if seq_id is None:
            if amount == -1:
                return self._topo_maps_data[db_name]
            topo_maps = {}
            for _ in range(amount):
                seq_id = random.sample(self._topo_maps_data[db_name].keys(), 1)[0]
                sample = self._topo_maps_data[db_name][seq_id]
                topo_maps[db_name+"-"+seq_id] = sample
            return topo_maps
        else:
            return {db_name+"-"+seq_id: self._topo_maps_data[db_name][seq_id]}



    def load(self, db_name, skip_unknown=False, skip_placeholders=False,
             limit=None, segment=False, single_component=True, room_remap={'7-2PO-1': '7-1PO-3'},
             skipped_seq_pattern={"floor6_base*"},
             skipped_rooms={'Stockholm':{}}):#{'4-1PO-1'}}):
        """
        loads data. The loaded data is stored in self.topo_maps_data, a dictionary
        where keys are database names and values are data.

        If 'skip_unknown' is set to True, then we skip nodes whose labels map to unknown category.

        If 'skip_placeholders' is set to True, then we skip nodes that are placeholders

        'limit' is the max number of topo maps to be loaded.

        If `segment` is True, the topological map will be segmented such that each node in the graph
           is a whole room, and there is no doorway node.

        If `single_component` is True, then will make sure the loaded graph is a single, connected graph. If the
           raw graph contains multiple components, will select the largest component as the topological map, and
           discard smaller ones. This is by default True.

        CSV format should be:
        "
        node['id'](int), node['placeholder'](bool), node_pose_x(float), node_pose_y(float), 
            match_pose[1](float), match_pose[2](float), node_anchor_pose[0](float), node_anchor_pose[1](float),
            match_label(string), match_vscan_time(timestamp), ... 8 views (see comments below) ...
        "
        """
        topo_maps = {}
        db_path = os.path.join(self.db_root, db_name)
        for seq_id in sorted(os.listdir(db_path)):

            # Previously the sequence skipping does not explicitly exist. GraphSPN relied on the fact
            # that there is no polar scans for floor6_base sequences so no DGSM results for them in order
            # to skip the floor6_base sequences. Now we can make this explicit through the skipped_seq_pattern.
            skip_seq = False
            for pattern in skipped_seq_pattern:
                if re.search(pattern, seq_id) is not None:
                    if DEBUG:
                        print("Skipping %s (matched pattern %s)" % (seq_id, pattern))
                    skip_seq = True
            if skip_seq:
                continue

            # check if over limit
            if limit is not None and len(topo_maps) >= limit:
                break

            node_room_mapping = {}
            with open(os.path.join(db_path, seq_id, "rooms.dat")) as f:
                rows = csv.reader(f, delimiter=' ')
                for row in rows:
                    nid = int(row[0])
                    room_id = row[1]
                    node_room_mapping[nid] = room_id

            # not over limit. Keep loading.
            with open(os.path.join(db_path, seq_id, "nodes.dat")) as f:
                nodes_data_raw = csv.reader(f, delimiter=' ')

                nodes = {}
                conn = {}
                skipped = set({})

                # We may want to skip nodes with unknown classes. This means we need to first
                # pick out those nodes that are skipped. So we iterate twice.
                for row in nodes_data_raw:
                    nid = int(row[0])
                    label = row[8]
                    placeholder = bool(int(row[1]))
                    if skip_unknown:
                        if CategoryManager.category_map(label, checking=True) == CategoryManager.category_map('UN', checking=True):
                            skipped.add(nid)
                    if skip_placeholders:
                        if placeholder:
                            skipped.add(nid)

                    # Also, skip nodes in rooms that we want to skip
                    building = self.get_dbname_info(db_name)[0]  # we don't care whether db_name is test or train here.
                    if building in skipped_rooms and node_room_mapping[nid] in skipped_rooms[building]:
                        if DEBUG:
                            print("Skipping node %d in room %s" % (nid, node_room_mapping[nid]))
                        skipped.add(nid)

                f.seek(0)

                for row in nodes_data_raw:
                    node = {
                        'id': int(row[0]),
                        'placeholder': bool(int(row[1])),
                        'pose': tuple(map(float, row[2:4])),
                        'anchor_pose': tuple(map(float, row[6:8])),
                        'label': row[8],
                        'vscan_timestamp': float(row[9])
                    }

                    # Room mapping
                    room_id = node_room_mapping[node['id']]
                    if room_id in room_remap:
                        room_using = room_remap[room_id]
                        node['label'] = room_using.split("-")[1]  # since we change the room, label should also change
                        # print("Remap node %d from room %s to room %s" % (node['id'], room_id, room_using))
                    node['room'] = room_id
                    
                    # Skip it?
                    if skip_unknown and node['id'] in skipped:
                        continue

                    # Add connections
                    edges = set({})
                    i = 10
                    while i < len(row):
                        neighbor_id = int(row[i])
                        # Check if we should skip this
                        if skip_unknown and neighbor_id in skipped:
                            i += 3 # still need to advance the index
                            continue
                        
                        affordance = float(row[i+1])
                        view_number = float(row[i+2])
                        if neighbor_id != -1:
                            edges.add((neighbor_id, view_number))
                        i += 3

                    # If there is no edge for this node, we just skip it.
                    if len(edges) == 0:
                        continue

                    # No, we don't skip this.
                    pnode = PlaceNode(
                        node['id'], node['placeholder'], node['pose'], node['anchor_pose'], node['label'], node['room']
                    )
                    nodes[pnode.id] = pnode
                    conn[pnode.id] = edges
                # Build `edges` from nodes and conn.
                edges_prep = {}
                for nid in conn:
                    for nnid, view_number in conn[nid]:
                        if (nid, nnid) in edges or (nnid, nid) in edges:
                            continue
                        eid = len(edges_prep)
                        edge = TopoEdge(eid, nodes[nid], nodes[nnid])
                        edges_prep[(nid,nnid)] = edge
                edges = {edges_prep[pair].id:edges_prep[pair]
                         for pair in edges_prep}
                topo_map = TopologicalMap(edges)
                if segment:
                    topo_map = topo_map.segment(remove_doorway=True)
                # There may be multiple connected components in the topological map (due to skipping nodes).
                # If single_component is True, we only keep the largest component.
                if single_component:
                    components = topo_map.connected_components()
                    if len(components) > 1:
                        if DEBUG:
                            print("-- %s is broken into %d components" % (seq_id, len(components)))
                    topo_maps[seq_id] = max(components, key=lambda c:len(c.nodes))
                else:
                    topo_maps[seq_id] = topo_map

        # Save the loaded topo maps
        self._topo_maps_data[db_name] = topo_maps


    def split(self, mix_ratio, *db_names):
        """
        Mix up the sequences in dbs (from `db_names`), and split them up into two groups of sequences.
        The first group takes `mix_ratio` percentage, used for training. The second group takes 1 - `mix_ratio`
        percentage, used for testing.

        Returns two lists of sequoence ids, each for one group. Note that to distinguish Freiburg sequences and
        Saarbrucken sequences, each sequence id is prepended a string for its database (format "{db}-{seq_id}")
        """
        all_seqs = []
        for db_name in db_names:
            seqs = [db_name+"-"+seq_id for seq_id in self.get_topo_maps(db_name=db_name, amount=-1).keys()]
            all_seqs.extend(seqs)
        # Shuffle and split
        random.shuffle(all_seqs)
        split_indx = round(mix_ratio*len(all_seqs))
        return all_seqs[:split_indx], all_seqs[split_indx:]
        

    @property
    def topo_maps(self):
        return self._topo_maps_data

    def get(self, db_name, seq_id):
        return self._topo_maps_data[db_name][seq_id]
    

    def generate_visualization(self, coldmgr, db_names=None):
        """
        Generates visualizations of topological maps and save them
        to {db_root}/{db_name}/{seq_id}/{topo_map.png}

        coldmgr (ColdDatabaseManager): Cold database manager instance.
        """
        matplotlib.use('Agg')
        if db_names is None:
            db_names = self._topo_maps_data.keys()
        for db_name in db_names:
            coldmgr.db_name = db_name
            for seq_id in self._topo_maps_data[db_name]:
                topo_map = self._topo_maps_data[db_name][seq_id]
                topo_map.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'))
                plt.savefig(os.path.join(self.db_root, db_name, seq_id, "topo_map.png"))
                plt.clf()
                
                sys.stdout.write(".")
                sys.stdout.flush()
        sys.stdout.write('\n')

                
