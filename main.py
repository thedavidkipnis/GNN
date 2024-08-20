import networkx as nx
import rng_funcs as rng

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


    for t in teams:
        print(t.ID)
        for emp in t.employees:
            print(emp.ID, emp.resource_type, emp.team, '')
        print('=====')

    

    
if __name__ == "__main__":
    run()

