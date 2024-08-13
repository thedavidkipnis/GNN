import random
import networkx as nx


class Node:
    
    def __init__(self, ID, local_delta, pred_completion_delta, children, team, nc_prob):
        self.ID = ID
        self.local_delta = local_delta
        self.pred_completion_delta = pred_completion_delta
        self.children = children

        self.team = team
        self.nc_prob = nc_prob


class Employee:

    def __init__(self, ID, team, exp_years, attendance_probability):
        self.ID = ID
        self.team = team
        self.exp_years = exp_years
        self.attendance_probability = attendance_probability


class Team:

    def __init__(self, ID, employees):
        self.id = ID
        self.employees = employees
        # self.shift_start = shift_start
        # self.shift_end = shift_end


def gen_employees(num_employees):

    employees = []

    for i in range(num_employees):
        temp = Employee(i, None, 0, 0) # TODO: change 3rd and 4th parameters with rand probs based on functions Jonah created
        employees.append(temp)

    return employees


# generates the teams based on previously created employee objects
def gen_teams(num_teams, employees):

    pass


def gen_nodes(num_nodes):

    pass


def run():

    # nodes need to be added as tuples: (node_id, {attribute dict.})

    n1 = (1, {'local_delta': 0, 'pred_completion_delta': 0, 'team': 1, 'nc_prob': 0.0})
    n2 = (2, {'local_delta': 0, 'pred_completion_delta': 0, 'team': 2, 'nc_prob': 0.0})
    n3 = (3, {'local_delta': 0, 'pred_completion_delta': 0, 'team': 3, 'nc_prob': 0.0})
    n4 = (4, {'local_delta': 0, 'pred_completion_delta': 0, 'team': 4, 'nc_prob': 0.0})
    n5 = (5, {'local_delta': 0, 'pred_completion_delta': 0, 'team': 5, 'nc_prob': 0.0})

    sample_nodes = [n1, n2, n3, n4, n5]
    sample_edges = [(1,2),(1,3),(2,4),(3,4),(2,5)]

    DAG = nx.DiGraph()
    DAG.add_nodes_from(sample_nodes)
    DAG.add_edges_from(sample_edges)

    print(DAG._node[1]['local_delta'])

if __name__ == "__main__":
    run()

