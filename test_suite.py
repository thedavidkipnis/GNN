import unittest
import main

class TestDataPermanence(unittest.TestCase):

    def test_baseline_deltas(self):
        graph_1 = main.run(gen_node_deltas=True, update_task_baseline_deltas=True)
        graph_2 = main.run(gen_node_deltas=False, update_task_baseline_deltas=False)

        for node in graph_1:
            self.assertEqual(graph_1._node[node]['baseline_delta'], graph_2._node[node]['baseline_delta'])


    def test_edge_equality(self):
        graph_1 = main.run(gen_node_deltas=True, update_task_baseline_deltas=True)
        graph_2 = main.run(gen_node_deltas=False, update_task_baseline_deltas=False)

        self.assertEqual(graph_1.edges(),graph_2.edges())

if __name__ == '__main__':
    unittest.main()