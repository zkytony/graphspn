import random
import os
import tensorflow as tf
from graphspn.graphs.template import NodeTemplate, ThreeNodeTemplate, ThreeRelTemplate
from graphspn.spn.template_spn import TemplateSpn, NodeTemplateSpn
from graphspn.experiments.scripts.topo_map_dataset import TopoMapDataset
from graphspn.experiments.scripts.topo_map import CategoryManager, ColdDatabaseManager
from graphspn.util import ABS_DIR
import numpy as np

DB_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-topomaps")
GROUNDTRUTH_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-groundtruth")


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



def setup_threenode_tspn(dataset, num_vals, expand=False):
    tspn = NodeTemplateSpn(ThreeNodeTemplate, num_vals=num_vals)
    sess = tf.Session()
    
    tspn.generate_random_weights()
    tspn.init_weights_ops()
    tspn.init_learning_ops()
    tspn.initialize_weights(sess)
    sess.run(tf.global_variables_initializer())

    # train
    samples = np.array(dataset.create_template_dataset(ThreeNodeTemplate)['Stockholm'],
                       dtype=np.int32)
    train_likelihoods, test_likelihoods \
        = tspn.train(sess, samples, shuffle=True, batch_size=1000,
                     likelihood_thres=0.05, num_epochs=5, dgsm_lh=None,
                     samples_test=None, dgsm_lh_test=None)
    if expand:
        tspn.expand()
    tspn.init_learning_ops()        
    return sess, tspn
    

def setup_dataset(db,):
    dataset = TopoMapDataset(DB_ROOT)
    dataset.load(db, skip_unknown=True,
                 skip_placeholders=True, single_component=False)
    return dataset

def test_marginal_inference(sess, tspn):
    query = [2, -1, 2]
    marginals = tspn.marginal_inference(sess, np.array(query), normalize=True)
    print(marginals)
    most_likely = query
    for i in marginals:
        most_likely[i] = np.argmax(marginals[i])
    print(most_likely)

def test_marginal_inference_expand(sess, tspn):
    groundtruth = [2, 2, 2]
    query = [2, -1, 2]
    query_lh = create_likelihoods_vector(groundtruth,
                                         tspn,
                                         (0.995, 0.999), (0.001, 0.005),
                                         (0.995, 0.999), (0.001, 0.005),
                                         masked_sample=query)
    marginals = tspn.marginal_inference(sess, np.array(query), query_lh, normalize=True)
    print(marginals)
    most_likely = query
    for i in marginals:
        most_likely[i] = np.argmax(marginals[i])
    print(most_likely)    
    

def main():
    CategoryManager.TYPE = "FULL"
    CategoryManager.init()    
    dataset = setup_dataset("Stockholm")
    expand = True
    sess, tspn = setup_threenode_tspn(dataset, num_vals=CategoryManager.NUM_CATEGORIES, expand=expand)
    if not expand:
        test_marginal_inference(sess, tspn)
    else:
        test_marginal_inference_expand(sess, tspn)


if __name__ == "__main__":
    main()
