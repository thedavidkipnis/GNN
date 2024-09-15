import networkx as nx
import rng_funcs as rng
import random
import matplotlib.pyplot as plt
import pandas as pd

# to install stuff: py -m pip install scipy

# List representing the attandance probability of given employee 
# Each index represents a day of the week
# 5 indexes - 0 is Monday, 1 is Tuesday, etc.
emp_attendance_probability = [0.65, 0.73, 0.86, 0.89, .8]

'''
During task (node) generation, a time will be pulled at random from this list and assigned to a task
'''
task_baseline_time = []
for i in range(1,29):
    task_baseline_time.append(i*15)


TEAM_COUNT = 4 # used to be 20
EMPLOYEE_COUNT = 10 # used to be 200

# keeping class structure present for future if custome func. is needed
class Node:
    
    def __init__(self, ID, local_delta, pred_completion_delta, children, team, nc_prob):
        self.ID = ID
        self.local_delta = local_delta
        self.pred_completion_delta = pred_completion_delta
        self.children = children

        self.team = team
        self.nc_prob = nc_prob


class Employee:

    def __init__(self, ID, is_busy, team, exp_years, resource_type):
        self.ID = ID
        self.is_busy = is_busy
        self.team = team
        self.exp_years = exp_years
        self.resource_type = resource_type


class Team:

    def __init__(self, ID, employees, shop_type):
        self.ID = ID
        self.employees = employees
        self.shop_type = shop_type
        # self.shift_start = shift_start
        # self.shift_end = shift_end


def gen_employees(num_employees):

    # experience years 0-25yrs, skewed towards 

    employees = {}

    for i in range(num_employees):

        exp = round(rng.emp_exp_years_skewed_dist_gen(), 2)

        temp = Employee(i, False, None, exp, 'B')
        if i > num_employees / 2:
            temp.resource_type = 'A'

        employees[i] = temp

    return employees


# generates the teams based on previously created employee objects
def gen_teams(num_teams, employees):

    teams = {}
    
    # generating individual teams
    for i in range(num_teams):
        temp_team = Team(i,set(),1)
        if i % 2 == 0:
            temp_team.shop_type = 2

        teams[i] = temp_team

    # assigning employees to teams
    team_counter = 0
    for emp in employees:
        employees[emp].team = team_counter
        teams[team_counter].employees.add(emp)
        team_counter = (team_counter + 1) % num_teams

    return teams


def gen_DAG(num_top_layers: int, teams, employees):

    # Step 1: start at first node
    # Step 2: using num nodes to top. layer relationship func, determine how many nodes to generate for next "layer"
    # Step 3: generate next layer's nodes
    # Step 3: generate edges between current layer's nodes and next layer's nodes

    DAG = nx.DiGraph()

    cur_layer = []
    next_layer = []

    # adding starting node to first layer and graph
    exp_coefficient, starting_nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[0].exp_years)
    local_delta = round(exp_coefficient * 15, 5)

    starting_nc_occured = False
    if random.random() < starting_nc_prob:
        starting_nc_occured = True

    starter_node = (0, {'baseline_delta': 15, 'local_delta': local_delta, 'global_delta': local_delta, 'team': 0, 'emp_ID': 0, 'exp_coefficient': exp_coefficient, 'nc_prob': round(starting_nc_prob, 5), 'nc_occured': starting_nc_occured})
    cur_layer.append(starter_node)

    DAG.add_node(0)
    DAG._node[0] = starter_node[1]

    layer_counter = 1
    node_id_counter = 1

    team_idx_counter = 1 # TODO: refactor so that team IDs get assigned, not just counters

    # task_times_dict = {}
    # task_times_dict[0] = 15

    for i in range(num_top_layers-1):

        # emptying next layer
        next_layer.clear()

        # computing number of nodes to generate for following layer based on relationship function
        num_nodes_to_generate = rng.node_count_generation_by_top_layer_small(layer_counter)
        layer_counter += 1

        # generating nodes and creating next topological layer
        for node in range(num_nodes_to_generate):

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

            # task_times_dict[node_id_counter] = baseline_delta


            employee = random.sample(teams[team_idx_counter].employees, 1)[0]

            exp_coefficient, nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[employee].exp_years)
            local_delta = round(exp_coefficient * baseline_delta, 5)

            nc_occured = False
            if random.random() < nc_prob:
                nc_occured = True
                print(">>>>>>>>>>> og delta:", local_delta)
                local_delta = rng.adjust_local_delta_based_on_nc(local_delta)

            temp_node = (node_id_counter, {'baseline_delta': baseline_delta,
                                           'local_delta': local_delta,
                                            'global_delta': 0, 
                                            'team': team_idx_counter, 
                                            'emp_ID': employee, 
                                            'exp_coefficient' : round(exp_coefficient, 5), 
                                            'nc_prob': round(nc_prob, 5), 
                                            'nc_occured': nc_occured})
            node_id_counter += 1
            next_layer.append(temp_node)


            team_idx_counter = (team_idx_counter + 1) % TEAM_COUNT


        DAG.add_nodes_from(next_layer)

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

        # df = pd.DataFrame(task_times_dict, index=[0])
        # df.to_csv('task_baseline_deltas.csv')

    return DAG

'''
Method for generating edges between layers that are not right next to each other
'''
def post_process_edge_generation(DAG):

    layers = {}

    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        layers[layer] = []
        for node in nodes:
            DAG.nodes[node]["layer"] = layer
            layers[layer].append(node)


'''
Main point for simulation - filling out the global deltas based on predecessor attributes per node
'''
def simulation_global_delta_process_DAG(DAG):

    for node in DAG:
        print("Looking at node " + str(node))

        # populate global delta
        max_pred_global_delta = 0
        for i in DAG.predecessors(node):
            if DAG._node[i]["global_delta"] > max_pred_global_delta:
                max_pred_global_delta = DAG._node[i]["global_delta"]

        DAG._node[node]["global_delta"] = max_pred_global_delta + DAG._node[node]["local_delta"]

        print('Attributes:', DAG._node[node])

        print('Successors:', DAG[node])

        

        print("=======")


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


def run():
    
    # n1 = (1, {'local_delta': 1, 'pred_completion_delta': 0, 'team': 1, 'nc_prob': 0.0})
    # n2 = (2, {'local_delta': 3, 'pred_completion_delta': 0, 'team': 2, 'nc_prob': 0.0})
    # n3 = (3, {'local_delta': 55, 'pred_completion_delta': 0, 'team': 3, 'nc_prob': 0.0})
    # n4 = (4, {'local_delta': 2, 'pred_completion_delta': 0, 'team': 4, 'nc_prob': 0.0})
    # n5 = (5, {'local_delta': 13, 'pred_completion_delta': 0, 'team': 5, 'nc_prob': 0.0})

    # sample_nodes = [n1, n2, n3, n4, n5]
    # sample_edges = [(1,2),(1,3),(2,4),(3,4),(2,5)]

    # DAG = nx.DiGraph()
    # DAG.add_nodes_from(sample_nodes)
    # DAG.add_edges_from(sample_edges)    

    # Generating all employees
    EMPLOYEES = gen_employees(EMPLOYEE_COUNT)
    
    # Generating teams and dividing employees into teams
    TEAMS = gen_teams(TEAM_COUNT,EMPLOYEES)

    # print(len(EMPLOYEES))
    # for i in EMPLOYEES:
    #     print(i.ID)

    DAG = gen_DAG(5, TEAMS, EMPLOYEES)
    simulation_global_delta_process_DAG(DAG)
    
    # for node in DAG:
    #     print(DAG._node[node])


    display_DAG(DAG)

    
if __name__ == "__main__":
    run()

