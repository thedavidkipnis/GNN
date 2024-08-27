import networkx as nx
import rng_funcs as rng
import random
import matplotlib.pyplot as plt

# to install stuff: py -m pip install scipy

# List representing the attandance probability of given employee 
# Each index represents a day of the week
# 5 indexes - 0 is Monday, 1 is Tuesday, etc.
emp_attendance_probability = [0.65, 0.73, 0.86, 0.89, .8]


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

    employees = []

    for i in range(num_employees):

        exp = round(rng.emp_exp_years_skewed_dist_gen(), 2)

        temp = Employee(i, False, None, exp, 'B')
        if i > num_employees / 2:
            temp.resource_type = 'A'

        employees.append(temp)

    return employees


# generates the teams based on previously created employee objects
def gen_teams(num_teams, employees):

    teams = []
    
    # generating individual teams
    for i in range(num_teams):
        temp_team = Team(i,[],1)
        if i % 2 == 0:
            temp_team.shop_type = 2

        teams.append(temp_team)

    # assigning employees to teams
    team_counter = 0
    for emp in employees:
        emp.team = team_counter
        teams[team_counter].employees.append(emp)
        team_counter = (team_counter + 1) % num_teams

    return teams


def gen_nodes(num_nodes):

    pass


def gen_DAG(num_top_layers):

    # Step 1: create static set of topological layers
    # Step 2: populate topological layers based on function of relationship between num nodes (y) and layer (x)
    # Step 3: connect nodes with random probability between layers

    # ALT PLAN
    # Step 1: start at first node
    # Step 2: using num nodes to top. layer relationship func, determine how many nodes to generate for next "layer"
    # Step 3: generate next layer's nodes
    # Step 3: generate edges between current layer's nodes and next layer's nodes

    DAG = nx.DiGraph()

    cur_layer = []
    next_layer = []

    # adding starting node to first layer and graph
    starter_node = (0, {'local_delta': 1, 'pred_completion_delta': 0, 'team': 1, 'nc_prob': 0.0})
    cur_layer.append(starter_node)

    DAG.add_node(0)
    DAG._node[0] = starter_node[1]

    layer_counter = 1
    node_id_counter = 1

    for i in range(num_top_layers-1):
        # emptying next layer
        next_layer.clear()

        # computing number of nodes to generate for following layer based on relationship function
        num_nodes_to_generate = rng.node_count_generation_by_top_layer(layer_counter)
        layer_counter += 1

        # generating nodes and creating next topological layer
        for node in range(num_nodes_to_generate):
            temp_node = (node_id_counter, {'local_delta': 1, 'pred_completion_delta': 0, 'team': 1, 'nc_prob': 0.0})
            node_id_counter += 1
            next_layer.append(temp_node)


        DAG.add_nodes_from(next_layer)

        # creating edges between nodes
        if len(cur_layer) == 1: # case: cur_layer size is 1
            for node in next_layer:
                DAG.add_edge(cur_layer[0][0], node[0])

        elif len(cur_layer) > len(next_layer): #case: cur_layer size is larger than next_layer
            chance_of_connectivity = len(next_layer) / len(cur_layer)
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
    
    return DAG


def display_DAG(DAG):
    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        for node in nodes:
            DAG.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(DAG, subset_key="layer")

    fig, ax = plt.subplots(figsize=(10,5))
    nx.draw_networkx(DAG, pos=pos, ax=ax, node_size=1000, node_color = 'white' , edge_color = 'white', font_color = 'black' )
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
    employees = gen_employees(50)
    
    # Generating teams and dividing employees into teams
    teams = gen_teams(10,employees)


    DAG = gen_DAG(7)
    #print(DAG.nodes())
    #print(DAG.edges())
    
    display_DAG(DAG)
    
if __name__ == "__main__":
    run()

