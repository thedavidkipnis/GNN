
#Dependencies
import random
import networkx as nx
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import itertools
import pandas as pd

#Function for making specific Graph for testing
import itertools
import matplotlib.pyplot as plt
import networkx as nx

#subset_sizes = [1, 3, 5, 5, 4, 3, 2, 4, 4, 3, 1]

subset_sizes = [1,3,10,8,9,9,4,4,3,2,
                1,3,2,2,4,4,3,4,2,1]

'''subset_sizes = [1,3,10,15,15,22,23,28,33,36,42,37,38,37,33,36,31,34,31,29,31,26,25,28,23,26,25,20,23,21,21,19,
                18,16,14,12,10,11,9,7,3,5,4,2,8,8,16,17,24,23,23,20,24,19,20,20,20,17,18,20,
                17,16,19,17,17,13,16,13,16,15,12,11,10,11,10,11,8,9,7,5,6,4,7,5,3,6,3,3,2,1]
'''


'''def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.DiGraph()  # Changed to DiGraph
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from((u, v) for u in layer1 for v in layer2)  # Directed edges
    return G

G = multilayered_graph(*subset_sizes)'''

# Probabilities for each layer
layer_probabilities = [
    1, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6,
    1, 1, 0.6, 0.6, 0.5, 0.5, 0.7, 0.9, 0.85, 1
]

# subset_sizes = [1, 3, 5, 5, 4, 3, 2, 4, 4, 3, 1]
subset_sizes = [1,3,10,8,9,9,4,4,3,2,
                1,3,2,2,4,4,3,4,2,1]

def multilayered_graph_layered_prob(*subset_sizes, layer_probabilities):
    """
    Create a multilayered directed graph where each layer has its own probability
    for connecting to future layers.

    Args:
        subset_sizes: Sizes of each layer.
        layer_probabilities: List of probabilities for each layer to control random connection.

    Returns:
        G: A directed graph with nodes organized in layers and random connections between them.
    """
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]

    G = nx.DiGraph()  # Directed graph
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)

    # Add connections between adjacent layers using respective probabilities
    for i, (layer1, layer2) in enumerate(nx.utils.pairwise(layers)):
        p = layer_probabilities[i]  # Get the probability for the current layer
        for u in layer1:
            for v in layer2:
                if random.random() < p:  # Add edge with probability p
                    G.add_edge(u, v)

    return G

# Generate the graph
G = multilayered_graph_layered_prob(*subset_sizes, layer_probabilities=layer_probabilities)


# This function
for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
        G.nodes[node]["layer"] = layer

# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer")

fig, ax = plt.subplots(figsize=(20,10))
nx.draw_networkx(G, pos=pos, ax=ax, node_size=1, node_color = 'white' , edge_color = 'white', font_color = 'black' )
ax.set_facecolor('#1AA7EC')
fig.tight_layout()
plt.show()

#Functions for making graph attributes
def assign_task_attributes(G: nx.DiGraph) -> Dict[int, Dict]:
  resources = ['1','2','3','4','5','6','7','8','9','10']


  return {node: {'time': random.randint(30, 420), 'resource': random.choice(resources)} for node in G.nodes()}

#Data Visualzation
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

task_attributes = assign_task_attributes(G)
print(task_attributes)

# Modify topological sorting with random selection
def topological_sort_with_random_priority(G: nx.DiGraph, task_attributes: Dict[int, Dict]) -> List[int]:
    in_degree = {v: 0 for v in G}
    for _, v in G.edges():
        in_degree[v] += 1

    # Initial queue with tasks that have zero in-degree (no predecessors)
    queue = [(task_attributes[v]['time'], random.random(), v) for v in G if in_degree[v] == 0]

    result = []
    while queue:
        random.shuffle(queue)  # Shuffle to make it random instead of time-based priority
        _, _, v = queue.pop(0)  # Select a random task from the current layer
        result.append(v)
        for u in G.successors(v):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append((task_attributes[u]['time'], random.random(), u))
        random.shuffle(queue)  # Keep shuffling to ensure randomness

    return result

# Modified RCPSP Solver with random buffer
def rcpsp_solver_with_buffer(G: nx.DiGraph, task_attributes: Dict[int, Dict], min_buffer: int = 15, max_buffer: int = 420) -> List[Tuple[int, int]]:
    schedule = []
    resource_availability = {resource: 0 for resource in set(attr['resource'] for attr in task_attributes.values())}
    sorted_tasks = topological_sort_with_random_priority(G, task_attributes)

    for task in sorted_tasks:
        resource = task_attributes[task]['resource']
        predecessors = list(G.predecessors(task))
        earliest_start = max([next((s for t, s in schedule if t == pred), 0) + task_attributes[pred]['time']
                              + random.randint(min_buffer, max_buffer) for pred in predecessors] + [0])
        start_time = max(earliest_start, resource_availability[resource])

        schedule.append((task, start_time))
        resource_availability[resource] = start_time + task_attributes[task]['time']

    return schedule

#Run SOlver and Visualize Scheudule
def main():
    schedule = rcpsp_solver_with_buffer(G, task_attributes)


    # Find the task with the maximum start time
    max_task, max_start_time = max(schedule, key=lambda x: x[1])

    # Print the details for the task with the maximum start time
    print(f"Task {max_task}: Start Time = {max_start_time}, Duration = {task_attributes[max_task]['time']}, Resource = {task_attributes[max_task]['resource']}")


    print("Task Schedule:")
    #for task, start_time in schedule:
        #print(f"Task {task}: Start Time = {start_time}, Duration = {task_attributes[task]['time']}, Resource = {task_attributes[task]['resource']}")

    #visualize_dag(G)
    visualize_schedule(schedule, task_attributes)

if __name__ == "__main__":
    main()

import pandas as pd
def simulate_schedules(G: nx.DiGraph, task_attributes: Dict[int, Dict], num_runs: int = 5000) -> pd.DataFrame:
    all_schedules = []

    for run_id in range(num_runs):
        task_attributes = assign_task_attributes(G)
        schedule = rcpsp_solver_with_buffer(G, task_attributes)
        for task, start_time in schedule:
            end_time = start_time + task_attributes[task]['time']
            all_schedules.append({
                'run_id': run_id,
                'task_id': task,
                'start_time': start_time,
                'end_time': end_time,
                'resource': task_attributes[task]['resource']
            })

    return pd.DataFrame(all_schedules)

# Run the simulation and store in a DataFrame
def main():
    schedule_data = simulate_schedules(G, task_attributes, num_runs=5000)

    # Saving to CSV or just printing the dataframe for inspection
    print(schedule_data)
    schedule_data.to_csv('schedule_simulations.csv', index=False)

if __name__ == "__main__":
    main()

x = pd.read_csv('schedule_simulations.csv')

filtered_data = x[x['task_id'] == 78]

# Plot a histogram of column 'a' values
plt.hist(filtered_data['end_time'], bins=25, color='blue', edgecolor='black')
plt.title("Histogram of 'a' where 'b' = 78")
plt.xlabel("Values in 'a'")
plt.ylabel("Frequency")
plt.show()