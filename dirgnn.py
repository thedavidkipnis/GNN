'''
Stores all functionality for DAG, DirGNN, and associated processes
'''

import random
import rng_funcs as rng
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


'''
gen_DAG helper function for generating and adding nodes to the current layer, then adding them to the DAG
'''
def gen_layer_nodes(teams, employees, task_baseline_time, num_nodes_to_generate_for_cur_layer, node_id_counter, team_idx_counter):
    next_layer = []

    for node in range(num_nodes_to_generate_for_cur_layer):

            '''
            For each node:
                baseline_delta: the pre-simulation, predicted time for the task's completion (stays static during each simulation)
                local_delta: total time to complete task
                global_delta: time delta from start of project until end of current task
                team: which team can execute this task
                exp_coefficient: how much the deltas will be getting multiplied by
                nc_prob: non-conformance probability | chance of error
                nc_occured: whether or not a non-conformance occured during runthrough
            '''
            # generate task base time, assign employee, adjust according to experience
            baseline_delta = random.choice(task_baseline_time)

            employee = random.sample(sorted(teams[team_idx_counter].employees), 1)[0]

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

            node_id_counter += 1
            next_layer.append(temp_node)
            team_idx_counter = (team_idx_counter + 1) % len(teams)

    return [next_layer, node_id_counter, team_idx_counter]


'''
gen_DAG helper function for generating and adding edges between layers in the DAG during creation
'''
def gen_layer_edges(DAG, cur_layer, next_layer):
    
    edges_to_save = []

    # creating edges between nodes
    if len(cur_layer) == 1: # case: cur_layer size is 1
        for node in next_layer:
            DAG.add_edge(cur_layer[0][0], node[0])
            edges_to_save.append((cur_layer[0][0], node[0]))

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
                    edges_to_save.append((node[0], next_node[0]))

        # if no chance connection was made
        if not connection_found:
            connection_idx_to_add = random.randint(0, len(cur_layer)-1)
            DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])
            edges_to_save.append((cur_layer[connection_idx_to_add][0], next_node[0]))

    else: # case: cur_layer size smaller than or equal to next_layer size
        chance_of_connectivity = 0.5
        for next_node in next_layer:
            connection_found = False
            for node in cur_layer:
                if(random.random() > chance_of_connectivity):
                    DAG.add_edge(node[0], next_node[0])
                    edges_to_save.append((node[0], next_node[0]))
                    connection_found = True

            if not connection_found:
                connection_idx_to_add = random.randint(0, len(cur_layer)-1)
                DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])
                edges_to_save.append((cur_layer[connection_idx_to_add][0], next_node[0]))

    return edges_to_save


'''
Function for generating DAG nodes and edges
'''
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

    task_deltas = {}
    task_edges = []

    team_idx_counter = 0 # TODO: refactor so that team IDs get assigned, not just counters
#endregion

    for i in range(num_top_layers):

        # emptying next layer
        next_layer.clear()

        # computing number of nodes to generate for following layer based on relationship function
        num_nodes_to_generate_for_cur_layer = rng.node_count_generation_by_top_layer_small(layer_counter)
        layer_counter += 1

        # generating nodes and creating next topological layer
        par_list = gen_layer_nodes(teams, employees, task_baseline_time, num_nodes_to_generate_for_cur_layer, node_id_counter, team_idx_counter)
        next_layer = par_list[0]
        node_id_counter = par_list[1]
        team_idx_counter = par_list[2]

        if update_task_baseline_deltas:
                for node in next_layer:
                    task_deltas[node[0]] = node[1]['baseline_delta']

        DAG.add_nodes_from(next_layer)

        if len(cur_layer) == 0: # edge case for the first node (cur_layer doesn't exist yet)
            cur_layer = next_layer.copy()
            continue
        
        # generating edges between current and next layer
        layer_task_edges = gen_layer_edges(DAG, cur_layer=cur_layer, next_layer=next_layer)

        if update_task_baseline_deltas:
            task_edges += layer_task_edges

        # assigning next layer to be current layer for next iteration
        cur_layer = next_layer.copy()

    if update_task_baseline_deltas:
        np.save('task_baseline_deltas.npy', task_deltas)
        np.save('task_edges.npy', task_edges)

    return DAG


def gen_DAG_from_file(nodes_file, edges_file, teams, employees):
    try:
        nodes = np.load('task_baseline_deltas.npy', allow_pickle='TRUE').item()
        edges = np.load('task_edges.npy', allow_pickle='TRUE')

    except:
        print(f'Failed to open {nodes_file} and {edges_file}')
        return None

    DAG = nx.DiGraph()

    team_idx_counter = 0
    node_list = []

    # adding nodes
#region
    for node in nodes:
        employee = random.sample(sorted(teams[team_idx_counter].employees), 1)[0]
        exp_coefficient, nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[employee].exp_years)
        local_delta = round(exp_coefficient * nodes[node], 5) # baseline delta = node[1]

        nc_occured = False
        if random.random() < nc_prob:
            nc_occured = True
            local_delta = rng.adjust_local_delta_based_on_nc(local_delta)

        temp_node = (node, {'baseline_delta': nodes[node],
                                           'local_delta': local_delta,
                                            'global_delta': 0, 
                                            'team': team_idx_counter, 
                                            'emp_ID': employee, 
                                            'exp_coefficient' : round(exp_coefficient, 5), 
                                            'nc_prob': round(nc_prob, 5), 
                                            'nc_occured': nc_occured})
        node_list.append(temp_node)
        team_idx_counter = (team_idx_counter + 1) % len(teams)

    DAG.add_nodes_from(node_list)
#endregion

    # adding edges
    for edge in edges:
        DAG.add_edge(edge[0], edge[1])

    return DAG


'''
Main point for simulation - filling out the global deltas based on predecessor attributes per node
Input: DAG node set with empty global_delta fields
Output: DAG node set with generated global_delta fields
'''
'''#Global Delta Simulation Pseudo Code - Jonah's Proposed method

#Some details:
# 1. the topologocal sort is custome to be able to add randomness to it
# 2. the sorting mechanism uses a queue/bucket method to shuffle tasks within a topological layer before adding them to the 'sort_results' list
# 3. the sort_results list will dictate the order of "execution" - the order in which the RCPSP/Global Delta assinging mechanism takes tasks to place them into the "execution schedule"

# 1. Custom Topological Sort for random "queue" ordering
def topological_sort_with_random_priority(DAG):
    # Step 1: Initialize in-degree dictionary
    INITIALIZE an empty dictionary --> `in_degree`, to store the 'in-degree' (number of predecessors) for each task (node)
    FOR each task 'v' in the DAG:
        SET in_degree[v] = 0  # Initially, assume no predecessors for the task
        #where 'v' is the task/node we are assessing to establish the number of predecessors

    # Step 2: Update in-degree for each node based on edges (dependencies)
    FOR each edge (u, v) in the DAG (from 'u' to 'v'):
        INCREMENT in_degree[v] by 1  # 'v' has one more prerequisite, where 'u' in the predecessor of task 'v'

    # Step 3: Initialize queue with tasks that have no predecessors
    INITIALIZE an empty list --> `queue`
    FOR each task 'v' in the DAG:
        IF in_degree[v] == 0:  # Task has no predecessors (zero in-degree)
            ADD (task_duration, random_priority, v) to `queue` where:
                - task_duration = DAG.nodes[v]['local_delta']
                - random_priority = a random value (random.random()) to shuffle the queue later

    # Step 4: Initialize an empty list `sort_results` to store the sorted tasks, these tasks will be sorted such that a successor is never listed before its predecessor
    #The order within the topological layer will be random on purpose as to simulate the stochasticity of real production execution
    INITIALIZE an empty list --> `sort_results`

    # Step 5: Process the queue of tasks
    WHILE `queue` is not empty:
        SHUFFLE the `queue` to randomize the priority of tasks at the same level
        POP the first task 'v' from `queue` (ignoring its duration and random priority)
        ADD 'v' to the `sort_results` list  # 'v' is now scheduled for execution

        # Step 6: Update the in-degree of each successor of v
        FOR each task 'w' that is a successor of 'v' (i.e., tasks that depend on 'v'):
            DECREMENT in-degree[w] by 1  # u has one less prerequisite to be fulfilled
            IF in-degree[w] == 0:  # All predecessors for task 'u' are now complete
                ADD (task_duration, random_priority, w) to `queue` where:
                    - task_duration = DAG.nodes[w]['local_delta']
                    - random_priority = a new random value

        SHUFFLE the `queue` again to maintain randomness in task selection

    # Step 7: Return the sorted task list
    RETURN `sort_results`  # A topologically sorted list of tasks, with random priority where possible
    



def rcpsp_solver_with_buffer(DAG, min_buffer, max_buffer):
    # Step 1: Initialize an empty list `schedule` to store the final task schedule
    INITIALIZE an empty list --> `schedule` to store tuples (task, start_time)

    # Step 2: Track when each resource (employee) is available
    INITIALIZE a dictionary --> `resource_availability` where:
        FOR each task in the DAG:
            SET resource_availability[employee] = 0
            # This means every employee is initially available at time 0

    # Step 3: Get the topologically sorted tasks (with random priority) from `topological_sort_with_random_priority`
    SET `sorted_tasks` = topological_sort_with_random_priority(DAG)

    # Step 4: Schedule each task in the sorted order
    FOR each 'task' in `sorted_tasks`:
        # Get the employee (resource) assigned to this 'task'
        SET employee = DAG.nodes[task]['emp_ID']

        # Step 5: Find the earliest start time based on task dependencies (predecessors)
        INITIALIZE `earliest_start` = 0
        IF the 'task' has predecessors:
            FOR each predecessor task 'pred' of this 'task':
                FIND the `end_time` of 'pred' in `schedule`
                CALCULATE end_time_of_pred = start_time_of_pred + duration_of_pred
                ADD a random buffer between some 'min_buffer' (15min) and 'max_buffer' (24 hours) to simulate delays
                SET earliest_start = maximum of (end_time_of_pred + buffer) for all predecessors

        # Step 6: Determine when the employee is available to start the task
        GET the availability time of the employee (resource_availability[employee])
        ADD a random delay (resource_delay) to simulate resource unavailability

        # The final start time is the later of the earliest_start (dependencies) and employee availability
        SET start_time = maximum of (earliest_start, resource_availability[employee]) + resource_delay

        # Step 7: Record the task's start time in the `schedule`
        ADD (task, start_time) to the `schedule` list

        # Step 8: Update the availability of the employee
        # After an employee was assigned to their first job, the resource availiability "time" become that job's completion time (start_time + duration of task)
        SET resource_availability[employee] = start_time + duration_of_task

    # Step 9: Return the final schedule
    RETURN `schedule`  # A list of (task, start_time) tuples for the entire project
'''


def simulation_global_delta_process_DAG(DAG):

#Modify topological sorting with random selection
def topological_sort_with_random_priority(DAG: nx.DiGraph) -> List[int]:
    in_degree = {v: 0 for v in DAG}
    for _, v in DAG.edges():
        in_degree[v] += 1

    # Initial queue with tasks that have zero in-degree (no predecessors)
    queue = [(DAG.nodes[v]['local_delta'], random.random(), v) for v in DAG if in_degree[v] == 0]

    result = []
    while queue:
        random.shuffle(queue)  # Shuffle to make it random instead of time-based priority
        _, _, v = queue.pop(0)  # Select a random task from the current layer
        result.append(v)
        for u in DAG.successors(v):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append((DAG.nodes[u]['local_delta'], random.random(), u))
        random.shuffle(queue)  # Keep shuffling to ensure randomness

    return result

#Modified RCPSP Solver with random buffer
def rcpsp_solver_with_buffer(DAG: nx.DiGraph, min_buffer: int = 15, max_buffer: int = 420) -> List[Tuple[int, int]]:
    schedule = []
    resource_availability = {DAG.nodes[node]['emp_ID']: 0 for node in DAG.nodes()}
    sorted_tasks = topological_sort_with_random_priority(DAG)

    for task in sorted_tasks:
        resource = DAG.nodes[task]['emp_ID']
        predecessors = list(DAG.predecessors(task))

        # Find the earliest start time considering predecessor tasks
        earliest_start = max([next((s for t, s in schedule if t == pred), 0) + DAG.nodes[pred]['local_delta']
                              + random.randint(min_buffer, max_buffer) for pred in predecessors] + [0])

        # Add a random delay to simulate resource unavailability or delays
        resource_delay = random.randint(15, 1440)

        # Schedule the task, considering the resource's availability
        start_time = max(earliest_start, resource_availability[resource]) + resource_delay

        # Record the schedule
        schedule.append((task, start_time))

        # Update resource availability after the task is completed
        resource_availability[resource] = start_time + DAG.nodes[task]['local_delta']

    return schedule

#Recalculate task attributes (local_delta) for each iteration
def recalculate_local_deltas(DAG, teams, employees, task_baseline_times):
    for node in DAG.nodes:
        employee_id = DAG.nodes[node]['emp_ID']
        employee = employees[employee_id]
        baseline_delta = random.choice(task_baseline_times)

        # Recalculate experience coefficient and probability of error
        exp_coefficient, nc_prob = adjusted_task_time_and_prob_of_error(employee.exp_years)
        local_delta = round(exp_coefficient * baseline_delta, 5)

        # Simulate potential non-conformance (nc_occured)
        nc_occured = False
        if random.random() < nc_prob:
            nc_occured = True
            local_delta = adjust_local_delta_based_on_nc(local_delta)

        # Update node attributes
        DAG.nodes[node]['local_delta'] = local_delta
        DAG.nodes[node]['exp_coefficient'] = round(exp_coefficient, 5)
        DAG.nodes[node]['nc_prob'] = round(nc_prob, 5)
        DAG.nodes[node]['nc_occured'] = nc_occured

# Simulate multiple schedules
def simulate_schedules(DAG: nx.DiGraph, teams, employees, task_baseline_times, num_runs: int = 1000) -> pd.DataFrame:
    all_schedules = []

    for run_id in range(num_runs):
        # Recalculate task deltas for this iteration
        recalculate_local_deltas(DAG, teams, employees, task_baseline_times)

        # Generate schedule for this run
        schedule = rcpsp_solver_with_buffer(DAG)

        for task, start_time in schedule:
            end_time = start_time + DAG.nodes[task]['local_delta']
            '''all_schedules.append({
                'project_ID': run_id,
                'task_id': task,
                'start_time': start_time,
                'global_delta': end_time,
                'emp_ID': DAG.nodes[task]['emp_ID']
            })'''
            all_schedules.append({
                'project_ID': run_id,
                'task_id': task,
                'local_delta': DAG.nodes[task]['local_delta'],
                'start_time': start_time,
                'global_delta': end_time,
                'emp_ID': DAG.nodes[task]['emp_ID'],
                'team': DAG.nodes[task]['team'],  # Add the team
                'exp_coefficient': DAG.nodes[task]['exp_coefficient'],  # Add exp_coefficient
                'nc_prob': DAG.nodes[task]['nc_prob'],  # Add nc_prob
                'nc_occured': DAG.nodes[task]['nc_occured']  # Add nc_occured (True/False)
            })

    return pd.DataFrame(all_schedules)

# Main function to run the simulation
def main():
    # Number of teams and employees
    TEAM_COUNT = 4
    EMPLOYEE_COUNT = 10

    # Generating task baseline times
    task_baseline_times = [i * 15 for i in range(1, 29)]

    # Generate employees and teams (this comes from the provided functions)
    EMPLOYEES = gen_employees(EMPLOYEE_COUNT)
    TEAMS = gen_teams(TEAM_COUNT, EMPLOYEES)

    # Generate the DAG with task attributes (local_delta = time, emp_ID = resource)
    DAG = gen_DAG(num_top_layers=5, teams=TEAMS, employees=EMPLOYEES, task_baseline_time=task_baseline_times, update_task_baseline_deltas=False)

    # Simulate the schedules for multiple runs
    schedule_data = simulate_schedules(DAG, TEAMS, EMPLOYEES, task_baseline_times, num_runs=1000)

    # Print and save the results
    print(schedule_data)
    schedule_data.to_csv('schedule_simulations.csv', index=False)

if __name__ == "__main__":
    main()

    # node_bucket = set()

    # # inserting first node's ID into bucket
    # node_bucket.add(0)

    # while len(node_bucket) > 0:

    #     # list of successors to be added for next iteration of processing (neighbors)
    #     to_be_added = set()

    #     # looking at every current node; this is where resource contention reolution will be happening
    #     for node in node_bucket:
    #         print(node, DAG[node])

    #         for successor in DAG[node]:
    #             to_be_added.add(successor)

    #     print(to_be_added)

    #     node_bucket.clear()
    #     for i in to_be_added:
    #         node_bucket.add(i)

    #     print(node_bucket)
    #     print('======')


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