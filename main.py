import random


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

    teams = []

    for i in range(num_teams):
        temp = Team(i, employees)
        teams.append(temp)

    return teams


def run():

    employees = gen_employees(26)
    teams = gen_teams(2, employees)


    DAG = [] # collection of nodes/tasks

    for node_counter in range(100):
        temp = Node(node_counter, 0, 0, [], 0, 0)
        DAG.append(temp)

    for node in DAG:
        print(node.ID)


if __name__ == "__main__":
    run()

