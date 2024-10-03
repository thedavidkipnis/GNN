'''
Stores all functionality for DAG, DirGNN, and associated processes
'''

import random
import rng_funcs as rng
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def gen_DAG(num_top_layers: int, teams, employees, task_baseline_time, update_task_baseline_deltas):

    # Step 1: start at first node
    # Step 2: using num nodes to top. layer relationship func, determine how many nodes to generate for next "layer"
    # Step 3: generate next layer's nodes
    # Step 3: generate edges between current layer's nodes and next layer's nodes

    # setup
#region
    DAG = nx.DiGraph()

    cur_layer = []
    next_layer = []

    layer_counter = 0
    node_id_counter = 0

    task_delta_dict = {}

    team_idx_counter = 0 # TODO: refactor so that team IDs get assigned, not just counters
#endregion

    for i in range(num_top_layers):

        # emptying next layer
        next_layer.clear()

        # computing number of nodes to generate for following layer based on relationship function
        num_nodes_to_generate_for_cur_layer = rng.node_count_generation_by_top_layer_small(layer_counter)
        layer_counter += 1

        # generating nodes and creating next topological layer
        for node in range(num_nodes_to_generate_for_cur_layer):

            '''
            For each node:
                baseline_delta: the pre-simulation, predicted time for the task's completion (stays static during each simulation)
                local_delta: total time to complete task
                global_delta: time delta from start of project until end of current task
                team: which team can execute this task
                nc_porc: non-conformance probability | chance of error
            '''
            # generate task base time, assign employee, adjust according to experience
            baseline_delta = random.choice(task_baseline_time)

            employee = random.sample(teams[team_idx_counter].employees, 1)[0]

            exp_coefficient, nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[employee].exp_years)
            local_delta = round(exp_coefficient * baseline_delta, 5)

            nc_occured = False
            if random.random() < nc_prob:
                nc_occured = True
                local_delta = rng.adjust_local_delta_based_on_nc(local_delta)

            temp_node = (node_id_counter, {'baseline_delta': baseline_delta,
                                           'local_delta': local_delta,
                                            'global_delta': 0, 
                                            'team': team_idx_counter, 
                                            'emp_ID': employee, 
                                            'exp_coefficient' : round(exp_coefficient, 5), 
                                            'nc_prob': round(nc_prob, 5), 
                                            'nc_occured': nc_occured})
            
            if update_task_baseline_deltas:
                task_delta_dict[node_id_counter] = baseline_delta

            node_id_counter += 1
            next_layer.append(temp_node)
            team_idx_counter = (team_idx_counter + 1) % len(teams)


        DAG.add_nodes_from(next_layer)

        if len(cur_layer) == 0: # edge case for the first node (cur_layer doesn't exist yet)
            cur_layer = next_layer.copy()
            continue

        # creating edges between nodes
        if len(cur_layer) == 1: # case: cur_layer size is 1
            for node in next_layer:
                DAG.add_edge(cur_layer[0][0], node[0])

        elif len(cur_layer) > len(next_layer): #case: cur_layer size is larger than next_layer
            chance_of_connectivity = len(next_layer) / len(cur_layer)

            if len(next_layer) <= 4: # situational case for making enough connections in cases where size difference between layers is too great
                chance_of_connectivity = 0.5

            connection_found = False
            for node in cur_layer:
                for next_node in next_layer:
                    if (random.random() < chance_of_connectivity):
                        connection_found = True
                        DAG.add_edge(node[0], next_node[0])

            # need to add edge case check for when NONE of the cur_layer nodes connect to a next_layer of size 1
            # if no chance connection was made
            if not connection_found:
                connection_idx_to_add = random.randint(0, len(cur_layer)-1)
                DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])

        else: # case: cur_layer size smaller than or equal to next_layer size
            chance_of_connectivity = 0.5
            for next_node in next_layer:
                connection_found = False
                for node in cur_layer:
                    if(random.random() > chance_of_connectivity):
                        DAG.add_edge(node[0], next_node[0])
                        connection_found = True
                if not connection_found:
                    connection_idx_to_add = random.randint(0, len(cur_layer)-1)
                    DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])

        # assigning next layer to be current layer for next iteration
        cur_layer = next_layer.copy()

    if update_task_baseline_deltas:
        np.save('task_baseline_deltas.npy', task_delta_dict)

    return DAG


def gen_DAG_from_file(filename):
    try:
        d = np.load(filename, allow_pickle='TRUE').item()
        # TODO: implement
    except:
        print(f'Failed to open {filename}')
        return



'''
Main point for simulation - filling out the global deltas based on predecessor attributes per node
'''
def simulation_global_delta_process_DAG(DAG):

    node_bucket = set()

    # inserting first node's ID into bucket
    node_bucket.add(0)

    while len(node_bucket) > 0:

        # list of successors to be added for next iteration of processing (neighbors)
        to_be_added = set()

        # looking at every current node; this is where resource contention reolution will be happening
        for node in node_bucket:
            print(node, DAG[node])

            for successor in DAG[node]:
                to_be_added.add(successor)

        print(to_be_added)

        node_bucket.clear()
        for i in to_be_added:
            node_bucket.add(i)

        print(node_bucket)
        print('======')


'''
Prints DAG to console
'''
def print_DAG(DAG):
    for node in DAG:
        print(node, DAG._node[node])


'''
Displays DAG using pyplot
'''
def display_DAG(DAG):
    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        for node in nodes:
            DAG.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(DAG, subset_key="layer")

    fig, ax = plt.subplots(figsize=(10,10))
    nx.draw_networkx(DAG, pos=pos, ax=ax, node_size=10, node_color = 'white' , edge_color = 'white', font_color = 'black', with_labels=False)
    ax.set_facecolor('#1AA7EC')
    fig.tight_layout()
    plt.show()


'''
Method for generating edges between layers that are not right next to each other, TODO: complete
'''
def post_process_edge_generation(DAG):

    layers = {}

    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        layers[layer] = []
        for node in nodes:
            DAG.nodes[node]["layer"] = layer
            layers[layer].append(node)