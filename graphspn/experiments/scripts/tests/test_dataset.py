import os
import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
from graphspn.experiments.scripts.topo_map_dataset import TopoMapDataset
from graphspn.experiments.scripts.topo_map import CategoryManager, ColdDatabaseManager
from graphspn.graphs.template import ThreeNodeTemplate, ThreeRelTemplate
from graphspn.util import ABS_DIR
import matplotlib.pyplot as plt


DB_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-topomaps")
GROUNDTRUTH_ROOT = os.path.join(ABS_DIR, "experiments", "dataset", "cold-groundtruth")

def test_topo_map_visualization(dataset, coldmgr, seq_id=None):
    rcParams['figure.figsize'] = 44, 30
    topo_maps = dataset.get_topo_maps(db_name="Stockholm",
                                      amount=1, seq_id=seq_id)
    for seq_id in topo_maps:
        topo_map = topo_maps[seq_id]
        print(len(topo_map.nodes))
        topo_map.visualize(plt.gca(), coldmgr.groundtruth_file(seq_id.split("_")[0], 'map.yaml'), show_nids=True)
        plt.savefig('%s.png' % seq_id)
        plt.clf()
        plt.close()
        # print("Saved %s.png" % seq_id)

def test_loading_training_data(dataset):
    samples = dataset.create_template_dataset(ThreeNodeTemplate)
    print(len(samples['Stockholm']))
    print("HEY")

def main():
    CategoryManager.TYPE = "FULL"
    CategoryManager.init()
    
    coldmgr = ColdDatabaseManager("Stockholm", GROUNDTRUTH_ROOT)
    dataset = TopoMapDataset(DB_ROOT)
    dataset.load("Stockholm", skip_unknown=True,
                 skip_placeholders=True, single_component=False)
    test_topo_map_visualization(dataset, coldmgr, seq_id="floor6_cloudy_a1")
    test_loading_training_data(dataset)

if __name__ == "__main__":
    main()
