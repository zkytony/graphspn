import random
import numpy as np
import os
import tensorflow as tf
import json
import sys
import time
import glob
from multiprocessing import Process
from pprint import pprint
import heapq

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import rcParams

from graphspn.spn.spn_model import SpnModel
from graphspn.graphs.template import NodeTemplate, StarTemplate, ThreeNodeTemplate, SingletonTemplate, PairTemplate
from graphspn.spn.template_spn import TemplateSpn, NodeTemplateSpn
from graphspn.spn.instance_spn import NodeTemplateInstanceSpn
from graphspn.experiments.scripts.topo_map_dataset import TopoMapDataset
from graphspn.util import ABS_DIR, json_safe
from graphspn.experiments.scripts.topo_map import CategoryManager, ColdDatabaseManager
from graphspn.experiments.scripts.topo_map import PlaceNode, TopologicalMap, TopoEdge, CompoundPlaceNode

## Hyper parameters
#-- template SPN structure
NUM_DECOMPS=1
NUM_SUBSETS=3
NUM_MIXTURES=5
NUM_INPUT_MIXTURES=5
#-- template SPN training
LIKELIHOOD_THRES=0.2
NUM_PARTITIONS_PER_TRAINING_GRAPH=10
BATCH_SIZE=2000
NUM_EPOCHS=10
#-- instantiation (instance SPN)
NUM_PARTITIONS=5
EXTRA_PARTITION_MULTIPLYER=3

## Other parameters
#-- overall experiment
NUM_GRAPHS=1
NUM_ROUNDS=3
RATE_OCCLUDED=0.2  # percentage of incorrectly classified nodes
NOISE_LEVEL=0
#-- template SPN training
SAVE=True
LOAD_IF_EXISTS=True
TEMPLATES = [StarTemplate, ThreeNodeTemplate, PairTemplate, SingletonTemplate]
#-- paths
DB_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-topomaps")
GROUNDTRUTH_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-groundtruth")



## Disambiguate noisy local evidence
def test_inference_noisy(sess,
                         template_spns,
                         dataset,
                         noise_level=0,
                         ngraphs=5, nrounds=3,
                         num_partitions=5,
                         extra_partition_multiplyer=3,
                         testdb="Freiburg", rate_occluded=0.2):
    # synthetic experiment (AAAI 2018).
    hic, loc, hiic, loic = interpret_noise_level(noise_level)
    pprint(get_noisification_level(hic, loc, hiic, loic, uniform_for_incorrect=noise_level == 0))

    topo_maps = dataset.get_topo_maps(db_name=testdb, amount=ngraphs)
    for db_seq_id in topo_maps:
        _test_single_graph(sess, db_seq_id, topo_maps[db_seq_id], template_spns, nrounds,
                           hic, loc, hiic, loic,
                           uniform_for_incorrect=noise_level == 0,
                           num_partitions=num_partitions,
                           extra_partition_multiplyer=extra_partition_multiplyer,
                           consider_placeholders=False,
                           save_results=True, rate_occluded=rate_occluded)
    

def _test_single_graph(sess, db_seq_id, topo_map, template_spns, nrounds,
                       high_likelihood_correct, low_likelihood_correct,
                       high_likelihood_incorrect, low_likelihood_incorrect,
                       uniform_for_incorrect=False,
                       num_partitions=5,
                       extra_partition_multiplyer=3, 
                       consider_placeholders=False,
                       save_results=False, rate_occluded=0.2):
    
    results = []

    db_name, seq_id = db_seq_id.split("-")[0], db_seq_id.split("-")[1]
    print("--" + seq_id + "---------------------------------------")

    true_catg_map = topo_map.current_category_map()


    instance_spn = NodeTemplateInstanceSpn(topo_map, sess,
                                           *[(tspn , tspn.template) for tspn in template_spns],
                                           num_partitions=num_partitions,
                                           graph_name=seq_id,
                                           super_node_class=CompoundPlaceNode,
                                           super_edge_class=TopoEdge,
                                           extra_partition_multiplyer=extra_partition_multiplyer)
    # Visualize partitions
    viz_partitions_dirpath = os.path.join("results", "aaai18", "noisy", "partitions-%s" % db_seq_id)
    os.makedirs(viz_partitions_dirpath, exist_ok=True)
    visualize_partitions(viz_partitions_dirpath, instance_spn, db_name, seq_id)

    # expand, init ops
    instance_spn.expand(use_cont_vars=True)  # use_cont_vars converts non-log scale continuous-valued inputs into log scale.
    start = time.time()
    instance_spn.init_ops(no_mpe=True)  # This step takes a few minutes.
    print("  Ops intialized after %.3fs" % (time.time() - start))


    for rr in range(nrounds):
        # After masking, some nodes in topo_map have category '-1' (indicating "occluded", or "masked").
        topo_map.mask_by_policy(random_fixed_policy, rate_occluded=rate_occluded)
        query_catg_map = topo_map.current_category_map()  # contains -1

        # Create likelihoods.
        query_lh = create_instance_spn_likelihoods(topo_map, true_catg_map,
                                                   high_likelihood_correct, low_likelihood_correct,
                                                   high_likelihood_incorrect, low_likelihood_incorrect,
                                                   uniform_for_incorrect=uniform_for_incorrect,
                                                   consider_placeholders=consider_placeholders)

        # Query should be all -1, because the network doesn't know which ones are incorrect.
        # And we query for all nodes.
        query_nids = list(topo_map.nodes.keys())
        query = {k:-1 for k in query_nids}#TbmExperiment.create_category_map_from_likelihoods(0, query_lh) #

        # Inference. Get marginals.
        print("Inferring most likely categories (Graph has %d nodes and %d edges)"
              % (len(topo_map.nodes), len(topo_map.edges)))
        start = time.time()
        marginals = instance_spn.infer_marginals(sess, query_nids, query, query_lh=query_lh)
        print("  Inference completed after %.3fs" % (time.time() - start))
        
        result_catg_map = {
            nid: np.argmax(marginals[nid]) for nid in query
        }

        print("One upward pass")
        start = time.time()
        likelihood = instance_spn.evaluate(sess, result_catg_map, query_lh)
        print("  Upward pass completed after %.3fs" % (time.time() - start))
        
        topo_map.reset_categories()

        # Calculate accuracy
        total_correct, total_cases = 0, 0
        record = {
            'results':{
                CategoryManager.category_map(k, rev=True):[0,0,0]  # correct, total, accuracy
                for k in range(CategoryManager.NUM_CATEGORIES)
            },
            'instance':{}
        }

        for nid in true_catg_map:
            true_catg = CategoryManager.category_map(true_catg_map[nid], rev=True) # (str)
            infrd_catg = CategoryManager.category_map(result_catg_map[nid], rev=True)

            if true_catg == infrd_catg:
                record['results'][true_catg][0] += 1  # record 
                total_correct += 1
            record['results'][true_catg][1] += 1
            record['results'][true_catg][2] = record['results'][true_catg][0] / record['results'][true_catg][1]
            total_cases += 1 

        record['instance']['true'] = true_catg_map
        record['instance']['query'] = query_catg_map
        record['instance']['result'] = result_catg_map
        record['instance']['likelihoods'] = query_lh
        record['instance']['policy'] = "random_fixed_policy"
        record['results']['_overall_'] = total_correct / max(total_cases,1)
        record['results']['_total_correct_'] = total_correct
        record['results']['_total_inferred_'] = total_cases

        if save_results:
            save_path = os.path.join("results", "aaai18", "noisy", db_seq_id + "-round_" + str(rr))
            os.makedirs(save_path, exist_ok=True)

            with open(os.path.join(save_path, "report.log"), "w") as f:
                json.dump(json_safe(record['results']), f, indent=4, sort_keys=True)
            with open(os.path.join(save_path, "test_instance.log"), "w") as f:
                json.dump(json_safe(record['instance']), f, indent=4, sort_keys=True)

            # save visualizations
            save_vis(topo_map, record['instance']['true'], db_name, seq_id, save_path, 'groundtruth', False)
            save_vis(topo_map, record['instance']['query'], db_name, seq_id, save_path, 'query', False)
            save_vis(topo_map, record['instance']['result'], db_name, seq_id, save_path, 'result', False)

        results.append(record)
    

## Creating random noisy evidence
def create_likelihoods_vector(sample, model,
                              high_likelihood_correct, low_likelihood_correct,
                              high_likelihood_incorrect, low_likelihood_incorrect, 
                              masked_sample=None):
    """
    Creates a numpy.array of shape (n, m) where n is the number of semantics variables (per template).
    m is the number of semantic categories. 

    sample (list): contains semantic variable values, in order.
    model (TemplateModel): the spn model that `sample` suits for.
    high_likelihood (tuple): the min & max likelihood of the semantics variable's true class.
    low_likelihood (tuple): the min & max likelihood of the masked semantics variable's true class.
                and the other (untrue) classes.

    masked_sample (list): Masked sample that contains -1. For masked values,
                  we reverse the pattern of assigning likelihoods to classes.
    """
    likelihoods = np.zeros((model.num_nodes, CategoryManager.NUM_CATEGORIES))

    for i in range(len(sample)):
        likelihoods[i] = create_likelihoods_for_single_node(sample[i],
                                                            high_likelihood_correct, low_likelihood_correct,
                                                            high_likelihood_incorrect, low_likelihood_incorrect, 
                                                            masked=masked_sample is not None and masked_sample[i] == -1)

    return likelihoods


def create_likelihoods_for_single_node(catg,
                                       high_likelihood_correct, low_likelihood_correct,
                                       high_likelihood_incorrect, low_likelihood_incorrect, 
                                       masked=False, uniform_for_incorrect=False,
                                       consider_placeholders=False, is_placeholder=False):
    """
    Creates a numpy.array of shape (m,) where m is the number of semantic categories. Each element
    is a float number indicating the likelihood of that category.

    catg (int): true category number (semantic value)
    high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
    low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                and the other (untrue) classes.
    masked (bool): True if this semantic value will be masked to -1.
    uniform_for_incorrect (bool): True if use uniform distribution for likelihoods in incorrectly classified nodes.
    consider_placeholders (bool): If True, will set the likelihoods for a placeholder node to be uniform, regardless
                                  of the high/low likelihoods setting.
    is_placeholder (bool): If True, the current node is a placeholder.
    """
    likelihoods = np.zeros((CategoryManager.NUM_CATEGORIES,))
    if not masked:
        for k in range(CategoryManager.NUM_CATEGORIES):
            if catg == k:
                likelihoods[k] = random.uniform(high_likelihood_correct[0], high_likelihood_correct[1])
            else:
                likelihoods[k] = random.uniform(low_likelihood_correct[0], low_likelihood_correct[1])
    else:
        highest = float('-inf')
        # Randomly select a non-groundtruth class to be the correct class
        cc = random.randint(0, CategoryManager.NUM_CATEGORIES)
        while cc == catg:
            cc = random.randint(0, CategoryManager.NUM_CATEGORIES)
        for k in range(CategoryManager.NUM_CATEGORIES):
            if uniform_for_incorrect or (consider_placeholders and is_placeholder):
                likelihoods[k] = 1.0
            else:
                if cc == k:
                    likelihoods[k] = random.uniform(high_likelihood_incorrect[0], high_likelihood_incorrect[1])
                else:
                    likelihoods[k] = random.uniform(low_likelihood_incorrect[0], low_likelihood_incorrect[1])
    # normalize so that likelihoods sum up to 1.
    likelihoods = likelihoods / np.sum(likelihoods)
    return likelihoods


def create_instance_spn_likelihoods(topo_map, true_catg_map,
                                    high_likelihood_correct, low_likelihood_correct,
                                    high_likelihood_incorrect, low_likelihood_incorrect, 
                                    uniform_for_incorrect=False, consider_placeholders=False):
    """
    Create a dictionary with format as described in mpe_inference():sample_lh.

    template_mode (0): 0 if the experiment is for NodeTemplate. 1 for EdgeTemplate.
    topo_map (TopologicalMap) The topo map
    true_catg_map (dict): a dictionary mapping from node id to groundtruth category value.
    high_likelihood_(in)correct (tuple): the min & max likelihood of the semantics variable's true class.
    low_likelihood_(in)correct (tuple): the min & max likelihood of the masked semantics variable's true class.
                and the other (untrue) classes.
    consider_placeholders (bool): If True, will set the likelihoods for a placeholder node to be uniform, regardless
                                  of the high/low likelihoods setting.
    """
    lh = {}
    for nid in topo_map.nodes:
        lh[nid] = tuple(create_likelihoods_for_single_node(true_catg_map[nid],
                                                           high_likelihood_correct, low_likelihood_correct,
                                                           high_likelihood_incorrect, low_likelihood_incorrect, 
                                                           masked=CategoryManager.category_map(topo_map.nodes[nid].label) == -1,
                                                           uniform_for_incorrect=uniform_for_incorrect,
                                                           consider_placeholders=consider_placeholders, is_placeholder=topo_map.nodes[nid].placeholder))
    return lh


## Setup dataset
def setup_dataset(dbs, skip_placeholders=False):
    dataset = TopoMapDataset(DB_ROOT)
    for db in dbs:
        dataset.load(db, skip_unknown=True,
                     skip_placeholders=skip_placeholders,
                     single_component=True)
    return dataset

## Setup template SPN
def setup_template_spns(sess, dataset, num_vals, traindbs,
                        num_decomps=1, num_subsets=1,
                        num_mixtures=2, num_input_mixtures=2,
                        batch_size=1000, num_epochs=5,
                        num_partitions_per_training_graph=10,
                        likelihood_thres=0.2,
                        save=True, load_if_exists=True):
    templates = TEMPLATES
    tspns = [NodeTemplateSpn(template, num_vals=num_vals,
                             num_decomps=num_decomps,
                             num_subsets=num_subsets,
                             num_mixtures=num_mixtures,
                             num_input_mixtures=num_input_mixtures) for template in templates]
    for i, template in enumerate(templates):
        tspn = tspns[i]
        print("____" + template.__name__ + "____")
        tspn.print_params()
        tspn_path = os.path.join("models", template.__name__, "-".join(traindbs) + ".spn")
        if load_if_exists and os.path.exists(tspn_path):
            print("Loading template SPN for %s" % template.__name__)
            tspn.init_weights_ops()
            tspn.initialize_weights(sess)
            tspn.load(tspn_path, sess)
            tspn.init_learning_ops()
            sess.run(tf.global_variables_initializer())
        else:
            tspn.generate_random_weights()
            tspn.init_weights_ops()
            tspn.init_learning_ops()
            tspn.initialize_weights(sess)
            sess.run(tf.global_variables_initializer())

            # singleton SPN should model a uniform distribution due to its lack of structure.

            # Prepare training samples
            samples_by_db = dataset.create_template_dataset([template],
                                                            num_partitions=num_partitions_per_training_graph,
                                                            db_names=traindbs)
            samples = []
            for db in samples_by_db[template.__name__]:
                samples.extend(samples_by_db[template.__name__][db])
            samples = np.array(samples, dtype=np.int32)

            train_likelihoods, test_likelihoods \
                = tspn.train(sess, samples, shuffle=True, batch_size=batch_size,
                             likelihood_thres=likelihood_thres,
                             num_epochs=num_epochs, dgsm_lh=None,
                             samples_test=None, dgsm_lh_test=None)
            if save:
                sys.stdout.write("Saving trained %s-SPN saved to path: %s ...\n" % (template.__name__, tspn_path))
                os.makedirs(os.path.dirname(tspn_path), exist_ok=True)
                tspn.save(tspn_path, sess)
                
        if template == SingletonTemplate:
            print("Relaxing prior...")
            SpnModel.make_weights_same(sess, tspn.root)
                    
    return tspns


## Masking policies
def random_policy(topo_map, node, rand_rate=0.2):
    return random.uniform(0, 1.0) <= rand_rate

def random_fixed_policy(topo_map, node, rate_occluded=0.2):
    """
    Randomly occlude a fixed percentage of nodes from topo_map. Default
    percentage is 0.2.
    """
    cur_num = sum(1 for nid in topo_map.nodes if CategoryManager.category_map(topo_map.nodes[nid].label) == -1)
    if cur_num > len(topo_map.nodes) * rate_occluded:
        return False
    return True

def random_fixed_plus_placeholders(topo_map, node, rate_occluded=0.2):
    """
    Randomly occlude a fixed percentage of nodes from topo_map. Default
    percentage is 0.2. ALSO occlude placeholder nodes, but does not consider placeholders when
    counting the number of masked nodes; The number of masked nodes does not depend on placeholder nodes.
    """
    if node.placeholder:
        return True
    cur_num = sum(1 for nid in topo_map.nodes if CategoryManager.category_map(topo_map.nodes[nid].label) == -1)
    num_ph = topo_map.num_placeholders()
    if cur_num - num_ph > (len(topo_map.nodes) - num_ph) * rate_occluded:
        return False
    return True

## Utilities
def save_vis(topo_map, category_map, db_name, seq_id, save_path, name, consider_placeholders):
    """
    If `consider_placeholders` is True, then all placeholders will be colored grey.
    Note that the graph itself may or may not contain placeholders and `consider_placholders`
    is not used to specify that.
    """
    ColdMgr = ColdDatabaseManager(db_name, None, gt_root=GROUNDTRUTH_ROOT)
    topo_map.assign_categories(category_map)
    rcParams['figure.figsize'] = 22, 14
    topo_map.visualize(plt.gca(), ColdMgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'),
                       consider_placeholders=consider_placeholders)
    plt.savefig(os.path.join(save_path, '%s_%s_%s.png' % (db_name, seq_id, name)))
    plt.clf()
    print("Saved %s visualization for %s:%s." % (name, db_name, seq_id))


def get_noisification_level(high_likelihood_correct, low_likelihood_correct,
                            high_likelihood_incorrect, low_likelihood_incorrect,
                            uniform_for_incorrect=False):
    """
    This function computes the average highest likelihood - next likelihood and std.
    These numbers reflect the extremity of noisification.
    - For nodes made correct (D_80):
    A closer value indicate more noisification
    and less confidence in local classification results.
    - For nodes made incorrect (D_20):
    we focus on the difference between the true class's likelihood and the highest likelihood.
    If the difference is large, then nosification is large.
    """
    dataset = TopoMapDataset(DB_ROOT)
    print("Loading data...")
    dataset.load("Stockholm", skip_unknown=True)
    topo_maps = dataset.get_topo_maps(db_name="Stockholm", amount=-1)
    stat_incrct = []
    stat_crct = []
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        groundtruth = topo_map.current_category_map()
        topo_map.mask_by_policy(random_policy)
        masked = topo_map.current_category_map()
        likelihoods = create_instance_spn_likelihoods(topo_map, groundtruth,
                                                      high_likelihood_correct, low_likelihood_correct,
                                                      high_likelihood_incorrect, low_likelihood_incorrect,
                                                      uniform_for_incorrect=uniform_for_incorrect)
        for nid in likelihoods:
            if masked[nid] == -1:
                # Made incorrect
                truecl_lh = likelihoods[nid][groundtruth[nid]]
                largest_lh = max(likelihoods[nid])
                stat_incrct.append(largest_lh - truecl_lh)
            else:
                # Not made incorrect
                largest2 = heapq.nlargest(2, likelihoods[nid])
                stat_crct.append(largest2[0] - largest2[1])

    result = {
        '_D_20_': {
            'avg': np.mean(stat_incrct),
            'std': np.std(stat_incrct),
        },
        '_D_80_': {
            'avg': np.mean(stat_crct),
            'std': np.std(stat_crct),
        }
    }
    return result    


def interpret_noise_level(noise_level):
    if noise_level == 0:
        hic = (0.5, 0.7)
        loc = (0.0004, 0.00065)
        hiic = None
        loic = None
        # incorrect node has uniform distribution
    elif noise_level == 1:
        hic = (0.5, 0.7)
        loc = (0.002, 0.008)
        hiic = (0.5, 0.7)
        loic = (0.22, 0.40)
    elif noise_level == 2:
        hic = (0.5, 0.7)
        loc = (0.014, 0.029)
        hiic = (0.5, 0.7)
        loic = (0.22, 0.35)
    elif noise_level == 3:
        hic = (0.5, 0.7)
        loc = (0.04, 0.09001)
        hiic = (0.5, 0.7)
        loic = (0.22, 0.315)
    elif noise_level == 4:
        hic = (0.5, 0.7)
        loc = (0.07, 0.13)
        hiic = (0.5, 0.7)
        loic = (0.17, 0.22)
    elif noise_level == 5:
        hic = (0.5, 0.7)
        loc = (0.17, 0.234)
        hiic = (0.5, 0.7)
        loic = (0.13, 0.155)
    return hic, loc, hiic, loic


def visualize_partitions(viz_dirpath, instance_spn, db_name, seq_id):
    print("Visualizing partitions")
    rcParams['figure.figsize'] = 22, 14
    coldmgr = ColdDatabaseManager(db_name, None, GROUNDTRUTH_ROOT)
    for k, partition in enumerate(instance_spn.partitions):
        ctype = 2
        for template in instance_spn.templates:
            supergraph = partition[template]
            node_ids = []
            for snid in supergraph.nodes:
                node_ids.append(supergraph.nodes[snid].to_place_id_list())
            img = instance_spn.graph.visualize_partition(plt.gca(), node_ids,
                                                         coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), ctype=ctype)
            ctype += 1
        plt.savefig(os.path.join(viz_dirpath, "partition-%d.png" % (k+1)))
        plt.clf()
        print("Visualized partition %d" % (k+1))


def collect_results(testdb):
    results_path = os.path.join("results", "aaai18", "noisy")
    summary = {}
    accuracies_by_catg = []  # accuracy per category
    accuracies = []  # accuracy per inference trial
    total_correct = 0
    total_inferred = 0
    for dirname in glob.glob(os.path.join(results_path, "%s-*" % testdb)):
        with open(os.path.join(dirname, "report.log")) as f:
            report = json.load(f)
            accuracies.append(float(report['_overall_']))

            for catg in report:
                if catg.startswith("_"):
                    continue
                if catg not in summary:
                    summary[catg] = [0,0,0]
                summary[catg][0] += int(report[catg][0])
                summary[catg][1] += int(report[catg][1])
                summary[catg][2] = summary[catg][0] / max(1, summary[catg][1])
                total_correct += int(report[catg][0])
                total_inferred += int(report[catg][1])

    for catg in summary:
        accuracies_by_catg.append(summary[catg][2])
    summary["_total_correct_"] = total_correct
    summary["_total_inferred_"] = total_inferred
    summary["_overall_"] = float(np.mean(accuracies))
    summary["_stdev_"] = float(np.std(accuracies))
    summary["_overall_by_class"] = float(np.mean(accuracies_by_catg))
    summary["_stdev_by_class"] = float(np.std(accuracies_by_catg))
    pprint(summary)
        
def main():
    if len(sys.argv) > 1:
        testdb = sys.argv[1]
    else:
        testdb = "Stockholm"
    CategoryManager.TYPE = "FULL"
    CategoryManager.init()        
    
    dbs = {"Stockholm", "Saarbrucken", "Freiburg"}
    traindbs = sorted(list(dbs - {testdb}))
    
    print("Loading dataset")
    dataset = setup_dataset(dbs)

    sess = tf.Session()

    print("Setting up template spns")
    tspns = setup_template_spns(sess, dataset, CategoryManager.NUM_CATEGORIES, traindbs,
                                likelihood_thres=LIKELIHOOD_THRES,
                                num_decomps=NUM_DECOMPS, num_subsets=NUM_SUBSETS,
                                num_mixtures=NUM_MIXTURES, num_input_mixtures=NUM_INPUT_MIXTURES,
                                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                                num_partitions_per_training_graph=NUM_PARTITIONS_PER_TRAINING_GRAPH,
                                save=SAVE, load_if_exists=LOAD_IF_EXISTS)

    threetspn = tspns[1]
    cases = []
    print("evaluating template spn...")
    for i in range(CategoryManager.NUM_CATEGORIES):
        for j in range(CategoryManager.NUM_CATEGORIES):
            for k in range(CategoryManager.NUM_CATEGORIES):
                sample = np.array([i, j, k])
                likelihood_val = threetspn.evaluate(sess, sample)[0][0]
                cases.append(tuple(CategoryManager.category_map(m, rev=True)
                                   for m in [i,j,k]) + ("%.5f" % (likelihood_val),))
    with open("three_tspn_training.csv", "w") as f:
        for c in cases:
            f.write(", ".join(c) + "\n")
                                

    try:
        print("Experiment: Disambiguating noisy information")
        test_inference_noisy(sess, tspns, dataset,
                             noise_level=NOISE_LEVEL,
                             num_partitions=NUM_PARTITIONS,
                             extra_partition_multiplyer=EXTRA_PARTITION_MULTIPLYER,
                             ngraphs=NUM_GRAPHS, nrounds=NUM_ROUNDS,
                             testdb=testdb, rate_occluded=RATE_OCCLUDED)
    finally:
        collect_results(testdb)
    sess.close()


if __name__ == '__main__':
    main()
    
