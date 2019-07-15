# Manages all paths

import os
from deepsm.experiments.common import DGSM_DB_ROOT, DGSM_RESULTS_ROOT

"""
Dataset path:
dataset_root/
    experiment_data/
        Nclasses/
            same_building/
                Stockholm/
                    real_data    [[{building}_{room_id} ... ], ... [{building#}_{seq_id} ... ] ...]
                    456-7/
                        set_defs
                    ...
                ... (freiburg, saarbrucken)
            across_buildings/
                real_data
                Stockholm_Freiburg-Saarbrucken     (stockholm for testing, freiburg saarbrucken for training)
                    set_defs
                ... (Freiburg_, Saarbrucken_)

Result path
results_root/
    Nclasses/
        same_building/
            Stockholm/
                456-7/
                    1PO/
                    CR/
                    ...
                    graphs/
                        trial#/
                            {graph_id}_likelihoods.json   (graph_id == {building#}_{seq_id})
                ...
            ...
        across_buildings/
            Stockholm_Freiburg-Saarbrucken     (stockholm for testing, freiburg saarbrucken for training)
                1PO/
                CR/
                ...
                graphs/
                    {graph_id}_likelihoods.json   (graph_id == {building#}_{seq_id})
            ...
"""


def path_to_polar_scans(building, dim="56x21", seq_id=None):
    if seq_id is not None:
        return os.path.join(DGSM_DB_ROOT,
                            "polar_scans", "polar_scans_%s" % building.lower(),
                            dim, "%s.pkl" % seq_id)
    else:
        return os.path.join(DGSM_DB_ROOT,
                            "polar_scans", "polar_scans_%s" % building.lower(), dim)


## same building ##
def path_to_dgsm_dataset_same_building(num_categories,
                                       building_name):
    return os.path.join(DGSM_DB_ROOT,
                        "experiment_data",
                        "%dclasses" % num_categories,
                        "same_building",
                        building_name)

def path_to_dgsm_set_defs_same_building(dgsm_dataset_same_building_path,
                                        train_floors,
                                        test_floor):
    """
    Args:
       train_floors (str): For example, 456 means floor4, floor5, floor6 (or seq#)
       test_floor (str): For example 7 means floor7
    """
    return os.path.join(dgsm_dataset_same_building_path,
                        "%s-%s" % (train_floors, test_floor))


def path_to_dgsm_result_same_building(num_categories,
                                      building_name,
                                      submodel_class,
                                      trial_number,
                                      train_floors,
                                      test_floor):
    return os.path.join(DGSM_RESULTS_ROOT,
                        "%dclasses" % num_categories,
                        "same_building",
                        building_name,
                        "%s-%s" % (train_floors, test_floor),
                        submodel_class,
                        "trial%d" % trial_number)

## across buildings ##
def path_to_dgsm_dataset_across_buildings(num_categories):
    return os.path.join(DGSM_DB_ROOT,
                        "experiment_data",
                        "%dclasses" % num_categories,
                        "across_buildings")


def path_to_dgsm_set_defs_across_buildings(dgsm_dataset_across_buildings_path,
                                           train_buildings,
                                           test_building):
    """
    Args:
       train_buildings (list): list of buildings for training (e.g. ['Stockholm','Freiburg'])
       test_building (str): building for testing (e.g. 'Saarbrucken')
    """
    case_dir = "%s_%s" % (test_building, "-".join(train_buildings))
    return os.path.join(dgsm_dataset_across_buildings_path,
                        case_dir)

def path_to_dgsm_result_across_buildings(num_categories,
                                         submodel_class,
                                         train_buildings,
                                         test_building):
    """
    Args:
       train_buildings (list): list of buildings for training (e.g. ['Stockholm','Freiburg'])
       test_building (str): building for testing (e.g. 'Saarbrucken')
    """
    case_dir = "%s_%s" % (test_building, "-".join(train_buildings))
    return os.path.join(DGSM_RESULTS_ROOT,
                        "%dclasses" % num_categories,
                        "across_buildings",
                        case_dir,
                        submodel_class)
