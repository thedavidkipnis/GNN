#Dependencies
import random
import networkx as nx
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

#Functions for making random graph and random attributes (for the sake of showing how RCPSP solvers work)
def create_dag(num_nodes: int = 100) -> nx.DiGraph:
    G = nx.gnc_graph(num_nodes)
    return G

'''
,'11','12','13','14','15','16','17',
'18','19','20','21','22','23','24','25',
'26','27','28','29','30','31','32','33',
'34','35','36','37','38','39','40','41',
'42','43','44','45','46','47','48','49',
'50','51','52','53','54','55','56','57',
'58','59','60','61','62','63','64','65',
'66','67','68','69','70','71','72','73',
'74','75','76','77','78','79','80','81',
'82','83','84','85','86','87','88','89',
'90','91','92','93','94','95','96','97',
'98','99','100','101','102','103','200',
'104','105','106','107','196','197','198','199',
'108','109','110','111','112','113','114','115',
'116','117','118','119','120','121','122','123',
'124','125','126','127','128','129','130','131',
'132','133','134','135','136','137','138','139',
'140','141','142','143','144','145','146','147',
'148','149','150','151','152','153','154','155',
'156','157','158','159','160','161','162','163',
'164','165','166','167','168','169','170','171',
'172','173','174','175','176','177','178','179',
'180','181','182','183','184','185','186','187',
'188','189','190','191','192','193','194','195']'''

def assign_task_attributes(G: nx.DiGraph) -> Dict[int, Dict]:
    resources = ['1','2','3','4','5','6','7','8','9','10']



    return {node: {'time': random.randint(30, 420), 'resource': random.choice(resources)} for node in G.nodes()}

#Data Visualzation (visualizing final schedule and dag if need be)
def visualize_schedule(schedule: List[Tuple[int, int]], task_attributes: Dict[int, Dict]):
    fig, ax = plt.subplots(figsize=(100, 20))
    resources = sorted(set(attr['resource'] for attr in task_attributes.values()))

    for task, start_time in schedule:
        resource = task_attributes[task]['resource']
        duration = task_attributes[task]['time']
        resource_index = resources.index(resource)

        ax.barh(resource_index, duration, left=start_time, height=0.5, align='center',
                color=plt.cm.Set3(resource_index / len(resources)), alpha=0.8)
        ax.text(start_time + duration/2, resource_index, f'Task {task}',
                ha='center', va='center', color='black', fontweight='bold')

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels(resources)
    ax.set_xlabel('Time')
    ax.set_ylabel('Resource')
    ax.set_title('RCPSP Schedule')
    plt.tight_layout()
    plt.show()

'''def visualize_dag(G: nx.DiGraph):
    for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
      for node in nodes:
          G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.show()'''

#Best Solver - OR-Tools CP Solver for RCPSP
from ortools.sat.python import cp_model

def rcpsp_with_cp(G, task_attributes):
    model = cp_model.CpModel()

    # Decision variables: start times for each task
    task_starts = {task: model.NewIntVar(0, sum(attr['time'] for attr in task_attributes.values()), f'start_{task}')
                   for task in G.nodes()}

    # Resource usage and intervals for each task
    task_intervals = {}
    for task in G.nodes():
        duration = task_attributes[task]['time']
        resource = task_attributes[task]['resource']
        end_time = model.NewIntVar(0, sum(attr['time'] for attr in task_attributes.values()), f'end_{task}')
        task_intervals[task] = model.NewIntervalVar(task_starts[task], duration, end_time, f'interval_{task}')

    # Precedence constraints: A task must finish before its successors can start
    for task in G.nodes():
        for successor in G.successors(task):
            model.Add(task_starts[successor] >= task_starts[task] + task_attributes[task]['time'])

    # Resource constraints: Only one task per resource at any time
    resource_to_tasks = {}
    for task, attr in task_attributes.items():
        resource = attr['resource']
        if resource not in resource_to_tasks:
            resource_to_tasks[resource] = []
        resource_to_tasks[resource].append(task_intervals[task])

    for resource, intervals in resource_to_tasks.items():
        model.AddNoOverlap(intervals)  # Ensures no two tasks use the same resource at the same time

    # Objective: Minimize the makespan (the maximum end time of all tasks)
    makespan = model.NewIntVar(0, sum(attr['time'] for attr in task_attributes.values()), 'makespan')
    model.AddMaxEquality(makespan, [task_starts[task] + task_attributes[task]['time'] for task in G.nodes()])
    model.Minimize(makespan)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Makespan: {solver.Value(makespan)}')
        schedule = []
        for task in G.nodes():
            start_time = solver.Value(task_starts[task])
            schedule.append((task, start_time))
        return schedule
    else:
        print('No solution found.')
        return []

#Shitty RCPSP Solvers, potentially for sim and other other shit#
#Sorting algos for priority assignment pre-solver usage
#Sorting version 1 --> sorting by topological order & task time
def topological_sort_with_priority(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> List[int]:
    in_degree = {v: 0 for v in G}
    for _, v in G.edges():
        in_degree[v] += 1

    queue = [(task_attributes[v]['time'], random.random(), v) for v in G if in_degree[v] == 0]
    queue.sort(reverse=True)

    result = []
    while queue:
        _, _, v = queue.pop(0)
        result.append(v)
        for u in G.successors(v):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append((task_attributes[u]['time'], random.random(), u))
        queue.sort(reverse=True)

    return result

#Sorting version 2 --> sorting by calculating longest path from each node, then priority queue based on the critical path length 
#(longer path gets higher priority)
def compute_longest_path_lengths(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> Dict[int, int]:
    # Initialize all nodes with a path length of zero
    longest_paths = {v: 0 for v in G}

    # Topologically sort the graph
    topological_order = list(nx.topological_sort(G))

    # Calculate the longest path for each node
    for v in topological_order:
        for u in G.successors(v):
            longest_paths[u] = max(longest_paths[u], longest_paths[v] + task_attributes[v]['time'])

    return longest_paths

def topological_sort_with_critical_path(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> List[int]:
    in_degree = {v: 0 for v in G}
    for _, v in G.edges():
        in_degree[v] += 1

    # Compute the longest path lengths (i.e., critical path) for prioritization
    longest_paths = compute_longest_path_lengths(G, task_attributes)

    # Priority queue based on the critical path length (longer path gets higher priority)
    queue = [(longest_paths[v], random.random(), v) for v in G if in_degree[v] == 0]
    queue.sort(reverse=True)

    result = []
    while queue:
        _, _, v = queue.pop(0)
        result.append(v)
        for u in G.successors(v):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append((longest_paths[u], random.random(), u))
        queue.sort(reverse=True)

    return result


#Shitty Solver 1 --> RCPSP Solver that places tasks in schedule one topological layer at a time based on pririty values derived from sorting algos
def rcpsp_solver_relaxed(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> List[Tuple[int, int]]:
    schedule = []
    resource_availability = {resource: 0 for resource in set(attr['resource'] for attr in task_attributes.values())}

#Can choose between these two function for sorting algos
    sorted_tasks = topological_sort_with_priority(G, task_attributes)
    #sorted_tasks = topological_sort_with_critical_path(G, task_attributes)

    for task in sorted_tasks:
        resource = task_attributes[task]['resource']
        predecessors = list(G.predecessors(task))
        earliest_start = max([next((s for t, s in schedule if t == pred), 0) + task_attributes[pred]['time']
                              for pred in predecessors] + [0])
        start_time = max(earliest_start, resource_availability[resource])

        schedule.append((task, start_time))
        resource_availability[resource] = start_time + task_attributes[task]['time']

    return schedule


#Shitty Solver 2 --> RCPSP Solver with Preemptive Scheduling: Looks ahead for idle resource windows and preemptivly schedules if possible
def rcpsp_solver_preemptive(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> List[Tuple[int, int]]:
    schedule = []
    resource_availability = {resource: 0 for resource in set(attr['resource'] for attr in task_attributes.values())}

    sorted_tasks = topological_sort_with_critical_path(G, task_attributes)

    for task in sorted_tasks:
        resource = task_attributes[task]['resource']
        predecessors = list(G.predecessors(task))

        # Calculate the earliest possible start time based on predecessors
        earliest_start = max([next((s for t, s in schedule if t == pred), 0) + task_attributes[pred]['time']
                              for pred in predecessors] + [0])

        # Look ahead for idle resource windows and preempt if possible
        start_time = max(earliest_start, resource_availability[resource])
        schedule.append((task, start_time))
        resource_availability[resource] = start_time + task_attributes[task]['time']

    return schedule

#Create New Dag & Attributes with random generator
G = create_dag()
task_attributes = assign_task_attributes(G)
print(task_attributes)

#Run Solver and Visualize Schedule
def main():
# Choose your Fighter
    #schedule = rcpsp_solver_relaxed(G, task_attributes)
    #schedule = rcpsp_solver_preemptive(G, task_attributes)
    schedule = rcpsp_with_cp(G, task_attributes)


    # Find the task with the maximum start time
    max_task, max_start_time = max(schedule, key=lambda x: x[1])

    # Print the details for the task with the maximum start time
    print(f"Task {max_task}: Start Time = {max_start_time}, Duration = {task_attributes[max_task]['time']}, Resource = {task_attributes[max_task]['resource']}")


    #print("Task Schedule:")
    #for task, start_time in schedule:
        #print(f"Task {task}: Start Time = {start_time}, Duration = {task_attributes[task]['time']}, Resource = {task_attributes[task]['resource']}")

    #visualize_dag(G)
    visualize_schedule(schedule, task_attributes)

if __name__ == "__main__":
    main()