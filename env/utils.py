import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import random

def get_mapped_node(map, i):
    return np.where(map[i] == 1)[0][0]

def from_mapping_to_matrix(mapping, n_devices):
    m = np.zeros((len(mapping), n_devices))
    for i in range(len(mapping)):
        m[i, mapping[i]] = 1
    return m

def from_matrix_to_mapping(m):
    return [get_mapped_node(m, i) for i in range(m.shape[0])]


def visualize_dag(G, widths, height):
    pos = {}
    max_width = max(widths)
    max_height = height

    height_incr = 2 / (max_height + 1)
    width_incr = 2 / max_width

    total_operator = sum(widths)

    pos[0] = np.array([-1, -1])
    pos[total_operator + 1] = np.array([1, -1])

    cnt = 1
    cur_height = -1 + height_incr
    cur_width = -1
    for i in range(height):
        for j in range(widths[i]):
            pos[cnt] = np.array([cur_height, cur_width])
            cur_width += width_incr
            cnt += 1
        cur_width = -1
        cur_height += height_incr

    nx.draw(G, pos)
    plt.show()

    return


def visualize_task_power_consumption(cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]

    plt.figure(figsize=(12, 6))
    plt.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    plt.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    plt.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    plt.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
            plt.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')


    plt.xlabel('Task')
    plt.ylabel('Frequency / Voltage')
    plt.title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    plt.xticks(tasks, [f"Task {task}" for task in tasks])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def visualize_task_power_consumption_with_connectivity(cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages, task_edges):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]

    plt.figure(figsize=(12, 6))
    
    plt.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    plt.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    plt.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    plt.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
        plt.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')

    for edge in task_edges:
        task_from, task_to = edge
        plt.plot([task_from, task_to], [device_frequencies[task_from], device_frequencies[task_to]], color='black', linestyle='--', alpha=0.5)

    plt.xlabel('Task')
    plt.ylabel('Frequency / Voltage')
    plt.title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    plt.xticks(tasks, [f"Task {task}" for task in tasks])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def visualize_task_power_consumption_with_dag(G, cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages, vmin, vmax, evmin, evmax, name, width):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]
    
    for i, task in enumerate(tasks):
        G.nodes[task]['frequency'] = device_frequencies[i]
        G.nodes[task]['adjusted_frequency'] = adjusted_frequencies[i]
        G.nodes[task]['voltage'] = device_voltages[i]
        G.nodes[task]['adjusted_voltage'] = adjusted_voltages[i]
    
    P = G.copy()
    for n in nx.topological_sort(P):
        if P.in_degree(n) == 0:
            P.nodes[n]['level'] = 0
        else:
            P.nodes[n]['level'] = max([P.nodes[v]['level'] for v in P.predecessors(n)]) + 1
    
    pos = nx.multipartite_layout(P, subset_key='level')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, 7), gridspec_kw={'width_ratios': [2, 3]})
    
    ax1.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    ax1.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    ax1.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    ax1.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
        ax1.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Frequency / Voltage')
    ax1.set_title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    ax1.set_xticks(tasks)
    ax1.set_xticklabels([f"V{task}" for task in tasks])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    node_colors = [G.nodes[n]['frequency'] for n in G.nodes()]
    edge_colors = [G.edges[e]['bytes'] if 'bytes' in G.edges[e] else 0 for e in G.edges()]
    edge_widths = [G.edges[e]['bytes'] / max(edge_colors) * 2 if 'bytes' in G.edges[e] else 1 for e in G.edges()]

    nx.draw_networkx_nodes(P, pos, node_color=node_colors, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax, node_size=300, ax=ax2)
    nx.draw_networkx_edges(P, pos, edgelist=G.edges(), width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=evmin, edge_vmax=evmax, ax=ax2)
    
    nx.draw_networkx_labels(P, pos, labels={n: f"V{n}" for n in G.nodes()}, font_size=12, ax=ax2)
    
    cb1 = plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.bwr, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)), ax=ax2, shrink=0.75)
    cb1.set_label('Task Compute Requirement', size=13)
    cb2 = plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=mpl.colors.Normalize(vmin=evmin, vmax=evmax)), ax=ax2, shrink=0.75)
    cb2.set_label('Data Transmission (bytes)', size=13)

    ax2.set_title(f"Task Connectivity (Graph)")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()



def visualize_task_power_consumption_with_arrows_and_colorbar(G, cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages, vmin, vmax, evmin, evmax, name, width):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]
    
    fig, ax1 = plt.subplots(figsize=(width, 7))
    
    ax1.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    ax1.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    ax1.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    ax1.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
        ax1.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Frequency / Voltage')
    ax1.set_title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    ax1.set_xticks(tasks)
    ax1.set_xticklabels([f"V{task}" for task in tasks])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for u, v in G.edges():
        bytes_value = G.edges[u, v]['bytes'] if 'bytes' in G.edges[u, v] else 0
        if bytes_value > 0:
            x_start = u
            x_end = v  
            y_start = max(device_frequencies[u], adjusted_frequencies[u]) + 5 
            y_end = max(device_frequencies[v], adjusted_frequencies[v]) + 5   
            arrow_color = plt.cm.Blues(bytes_value / max(evmax, 1))  
            ax1.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=dict(facecolor=arrow_color, edgecolor='black', arrowstyle='->', lw=1.5))
    
    # cb2 = plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=mpl.colors.Normalize(vmin=evmin, vmax=evmax)), ax=ax1, shrink=0.75)
    # cb2.set_label('Data Transmission (bytes)', size=13)

    ax1.text(0.95, 0.05, 'Arrows represent data communication', ha='right', va='bottom', transform=ax1.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()


def visualize_task_power_consumption_with_arrows_and_colorbar1(G, cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages, vmin, vmax, evmin, evmax, name, width):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]
    
    fig, ax1 = plt.subplots(figsize=(width, 7))
    
    ax1.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    ax1.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    ax1.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    ax1.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
        ax1.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Frequency / Voltage')
    ax1.set_title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    ax1.set_xticks(tasks)
    ax1.set_xticklabels([f"V{task}" for task in tasks])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for u, v in G.edges():
        bytes_value = G.edges[u, v]['bytes'] if 'bytes' in G.edges[u, v] else 0
        if bytes_value > 0:
            x_start = u  
            x_end = v    
            y_start = max(device_frequencies[u], adjusted_frequencies[u]) + 5  
            y_end = max(device_frequencies[v], adjusted_frequencies[v]) + 5      
            normalized_bytes = (bytes_value - evmin) / (evmax - evmin) if evmax != evmin else 0
            arrow_color = plt.cm.Blues(normalized_bytes)  
            ax1.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), 
                         arrowprops=dict(facecolor=arrow_color, edgecolor='black', arrowstyle='->', lw=1.5))
    
    
    cb2 = plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=mpl.colors.Normalize(vmin=evmin, vmax=evmax)), ax=ax1, shrink=0.75)
    cb2.set_label('Data Transmission (bytes)', size=13)

    ax1.text(0.95, 0.05, 'Arrows represent data communication (bytes)', ha='right', va='bottom', transform=ax1.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()


def visualize_task_power_consumption_with_arrows(G, cur_mapping, device_frequencies, adjusted_frequencies, device_voltages, adjusted_voltages, vmin, vmax, evmin, evmax, name, width):
    tasks = list(range(len(cur_mapping)))
    adjusted_frequencies = [freq if freq is not None else 0 for freq in adjusted_frequencies]
    adjusted_voltages = [volt if volt is not None else 0 for volt in adjusted_voltages]
    
    fig, ax1 = plt.subplots(figsize=(width, 7))
    
    ax1.bar(tasks, device_frequencies, color='orange', label='Device Frequency')
    ax1.bar(tasks, adjusted_frequencies, color='blue', alpha=0.6, label='Adjusted Frequency')
    ax1.plot(tasks, device_voltages, color='red', marker='o', label='Device Voltage', linestyle='-', linewidth=2)
    ax1.plot(tasks, adjusted_voltages, color='green', marker='o', label='Adjusted Voltage', linestyle='-', linewidth=2)

    for i in range(len(tasks)):
        ax1.text(i, max(device_frequencies[i], adjusted_frequencies[i]) - 20, f"dev{cur_mapping[i]}", ha='center', fontsize=8, color='white')
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Frequency / Voltage')
    ax1.set_title('Frequency and Voltage Devices Use When Running Tasks Compared To Their Default Settings')
    ax1.set_xticks(tasks)
    ax1.set_xticklabels([f"V{task}" for task in tasks])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for u, v in G.edges():
        bytes_value = G.edges[u, v]['bytes'] if 'bytes' in G.edges[u, v] else 0
        if bytes_value > 0:
            x_start = u  
            x_end = v    
            y_start = max(device_frequencies[u], adjusted_frequencies[u]) + 5  
            y_end = max(device_frequencies[v], adjusted_frequencies[v]) + 5      
            arrow_color = plt.cm.Blues(bytes_value / max(evmax, 1))  
            ax1.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=dict(facecolor=arrow_color, edgecolor='black', arrowstyle='->', lw=1.5))
            
            ax1.text((x_start + x_end) / 2, (y_start + y_end) / 2, f"{bytes_value}", color='black', fontsize=10, ha='center')

    plt.tight_layout()
    plt.show()


def graph_dag_structure(v,
                        alpha,
                        seed,
                        conn_prob=0.1,
                        visualize=False):
    np.random.seed(seed)
    random.seed(seed)

    height_mean = np.sqrt(v) / alpha
    height = int(np.ceil(np.random.uniform(height_mean * 0.8, height_mean * 1.2)))

    width_mean = alpha * np.sqrt(v)
    widths = []

    for i in range(height):
        widths.append(int(np.ceil(np.random.uniform(0, 2 * width_mean))))

    total_operator = sum(widths)

    G = nx.DiGraph()
    G.add_node(0)
    cnt = 1
    nodes = [[] for i in range(height + 2)]
    nodes[0].append(0)
    for i in range(height):
        for j in range(widths[i]):
            nodes[i + 1].append(cnt)
            G.add_node(cnt)
            for k in set().union(*nodes[:i + 1]):
                if np.random.rand() < conn_prob:
                    G.add_edge(k, cnt)
            cnt += 1

    nodes[-1].append(total_operator + 1)
    G.add_node(total_operator + 1)

    # make sure the depth equals height
    critical_path = [random.choice(n) for n in nodes]
    for x, y in zip(critical_path, critical_path[1:]):
        G.add_edge(x, y)

    # Validity checking, if any node in the middle has 0 indegree or 0 outdegree,
    # randomly connect
    for i, layer in enumerate(nodes):
        if i > 0 and i < height + 1:
            for node in layer:
                if G.out_degree(node) == 0:
                    G.add_edge(node, random.choice([a for n in nodes[i + 1:] for a in n]))
                if G.in_degree(node) == 0:
                    G.add_edge(random.choice([a for n in nodes[:i] for a in n]), node)

    if visualize:
        visualize_dag(G, widths, height)

    widths.insert(0, 1)
    widths.append(1)
    height += 2
    return G, widths, height


def generate_graph(alpha,
                   v,
                   connect_prob,
                   seed,
                   num_types,
                   avg_compute,
                   avg_bytes,
                   b_comp=0.2,
                   b_comm=0.2):
    G, widths, height = graph_dag_structure(v, alpha, seed, connect_prob)
    np.random.seed(seed)

    # compute requirement for each dag node
    for n in G.nodes:
        G.nodes[n]['compute'] = np.random.uniform(avg_compute * (1 - b_comp / 2), avg_compute * (1 + b_comp / 2))
        G.nodes[n]['h_constraint'] = np.random.choice(range(num_types))
        G.nodes[n]['h_frequency'] = np.random.uniform(10, 30)
        G.nodes[n]['h_voltage'] = np.random.uniform(5, 35)

    # Communication requirement (bytes) for each dag edge
    for edge in G.edges:
        G.edges[edge]['bytes'] = np.random.uniform(avg_bytes * (1 - b_comm / 2), avg_bytes * (1 + b_comm / 2))

    G.graph['alpha'] = alpha
    G.graph['v'] = v
    G.graph['connect_prob'] = connect_prob
    G.graph['seed'] = seed
    G.graph['num_types'] = num_types
    G.graph['avg_compute'] = avg_compute
    G.graph['avg_bytes'] = avg_bytes
    G.graph['b_comp'] = b_comp
    G.graph['b_comm'] = b_comm

    return G


def generate_network(n_devices,
                     seed,
                     num_types=5,
                     type_prob=0.3,
                     avg_speed=3,
                     avg_bw=200,
                     avg_delay=10,
                     b_bw=0.2,
                     b_speed=0.2
                     ):
    np.random.seed(seed)
    delay = np.random.uniform(0, 2 * avg_delay, (n_devices, n_devices))
    avg_comm = 1 / avg_bw
    comm_speed = np.random.uniform(avg_comm * (1 - b_bw / 2), avg_comm * (1 + b_bw / 2), (n_devices, n_devices))
    for i in range(n_devices):
        for j in range(i, n_devices):
            if i == j:
                delay[i][j] = 0
                comm_speed[i][j] = 0
            else:
                delay[i][j] = delay[j][i]
                comm_speed[i][j] = comm_speed[j][i]

    # speed for each device
    speed = np.random.uniform(avg_speed * (1 - b_speed / 2), avg_speed * (1 + b_speed / 2), n_devices)

    device_constraints = {}
    device_frequencies = {}
    device_voltages = {}

    for i in range(n_devices):
        device_constraints[i] = []
        for j in range(num_types):
            if np.random.rand() < type_prob:
                device_constraints[i].append(j)
        if len(device_constraints[i]) == 0:
            device_constraints[i].append(np.random.choice(range(num_types)))
    types = set(range(num_types)) - set().union(*[device_constraints[i] for i in range(n_devices)])
    for t in types:
        dev_set = np.random.choice(n_devices, int(n_devices * type_prob), replace=False).tolist()
        for d in dev_set:
            device_constraints[d].append(t)
    
    for i in range(n_devices):
        device_frequencies[i] = np.random.choice(range(30, 120))
        device_voltages[i] = np.random.uniform(35, 55)

    network = {}
    network["delay"] = delay
    network["comm_speed"] = comm_speed
    network["speed"] = speed
    network["device_constraints"] = device_constraints
    network["device_frequencies"] = device_frequencies
    network["device_voltages"] = device_voltages
    network['para'] = {}

    network['para']['n_devices'] = n_devices
    network['para']["seed"] = seed
    network['para']["num_types"] = num_types
    network['para']["type_prob"] = type_prob
    network['para']["avg_speed"] = avg_speed
    network['para']['avg_bw'] = avg_bw
    network['para']['avg_delay'] = avg_delay
    network['para']["b_bw"] = b_bw
    network['para']['b_speed'] = b_speed

    return network

def program_data_fn(v, alpha, p, num_types, avg_compute, avg_byte, b_comp, b_comm):
    return  f"v_{v}_alpha_{alpha}_connp_{p}_ntype_{num_types}_compute_{avg_compute}_bytes_{avg_byte}_bcomp_{b_comp}_bcomm_{b_comm}.pkl"

def network_data_fn(n, num_type, avg_speed, avg_bw, avg_delay, p, b_bw, b_speed):
    return  f"ndevice_{n}_ntype_{num_type}_speed_{avg_speed}_bw_{avg_bw}_delay_{avg_delay}_tprob_{p}_bbw_{b_bw}_bspeed_{b_speed}.pkl"


def generate_networks(n_devices,
                      type_probs,
                      avg_speeds,
                      avg_bws,
                      avg_delays,
                      b_bws,
                      b_speeds,
                      number=25,
                      num_type=5,
                      seeds=None,
                      save=False):
    if save:
        network_path = "./data/device_networks"
        if not os.path.exists(network_path):
            os.mkdir(network_path)
    else:
        res = {}
    for n in n_devices:
        for p in type_probs:
            for avg_speed in avg_speeds:
                for avg_bw in avg_bws:
                    for avg_delay in avg_delays:
                        for b_bw in b_bws:
                            for b_speed in b_speeds:
                                networks = []
                                if not seeds:
                                    seeds  = range(number)
                                for seed in seeds:
                                    network = generate_network(n, seed, num_type, p, avg_speed, avg_bw, avg_delay, b_bw,
                                                               b_speed)
                                    networks.append(network)

                                network_fn = network_data_fn(n, num_type, avg_speed, avg_bw, avg_delay, p, b_bw, b_speed)
                                if save:
                                    network_save_path = os.path.join(network_path, network_fn)
                                    to_pickle(network_save_path, networks)
                                else:
                                    res[network_fn] = networks
    if save:
        return
    return res


def generate_programs(alphas,
                      vs,
                      connect_probs,
                      avg_computes,
                      avg_bytes,
                      b_comps,
                      b_comms,
                      number=25,
                      num_types=5,
                      seeds=None,
                      save=False):
    if save:
        op_path = "./data/op_networks/"
        if not os.path.exists(op_path):
            os.mkdir(op_path)
    else:
        res = {}

    for alpha in alphas:
        for v in vs:
            for p in connect_probs:
                for avg_compute in avg_computes:
                    for avg_byte in avg_bytes:
                        for b_comp in b_comps:
                            for b_comm in b_comms:
                                programs = []
                                if not seeds:
                                    seeds = range(number)
                                for seed in seeds:
                                    G = generate_graph(alpha, v, p, seed, num_types, avg_compute, avg_byte, b_comp,
                                                       b_comm)
                                    programs.append(G)

                                op_fn = program_data_fn(v, alpha, p, num_types, avg_compute, avg_byte, b_comp, b_comm)
                                if save:
                                    op_fn = os.path.join(op_path, op_fn)
                                    to_pickle(op_fn, programs)
                                else:
                                    res[op_fn] = programs
    if save:
        return
    return res



def to_pickle(save_path, res):
    with open(save_path, 'wb') as handle:
        pickle.dump(res, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        res = pickle.load(handle)
    return res


def network_fn_filter(network_path,
                   n_devices=[20],
                   type_probs=[0.2],
                   avg_speeds=[5],
                   avg_bws=[100],
                   avg_delays=[10],
                   b_bws=[0.2],
                   b_speeds=[0.2],
                   num_types=[5]):

    fns = os.listdir(network_path)
    res = []
    for fn in fns:
        if '.pkl' not in fn:
            continue
        token = fn.split("_")
        ndevice = int(token[1])
        ntype = int(token[3])
        speed = int(token[5])
        bw = int(token[7])
        delay = int(token[9])
        tprob = float(token[11])
        bbw = float(token[13])
        bspeed = float(token[15][:-4])
        if ndevice not in n_devices:
            continue
        elif ntype not in num_types:
            continue
        elif tprob not in type_probs:
            continue
        elif speed not in avg_speeds:
            continue
        elif bw not in avg_bws:
            continue
        elif delay not in avg_delays:
            continue
        elif bbw not in b_bws:
            continue
        elif bspeed not in b_speeds:
            continue
        else:
            res.append(fn)

    return res

def close_to_any(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))

def program_fn_filter(op_path,
                   vs=[20],
                   alphas=[0.2],
                   connect_probs=[0.2],
                   avg_computes=[100],
                   avg_bytes=[10],
                   b_comps=[0.2],
                   b_comms=[0.2],
                   num_types=[5]):

    fns = os.listdir(op_path)
    res = []
    for fn in fns:
        if '.pkl' not in fn:
            continue
        token = fn.split("_")
        v = int(token[1])
        alpha = float(token[3])
        connp = float(token[5])
        ntype = int(token[7])
        compute = int(token[9])
        byte = int(token[11])
        bcomp = float(token[13])
        bcomm = float(token[15][:-4])
        if v not in vs:
            continue
        elif not close_to_any(alpha, alphas):
            continue
        elif not close_to_any(connp, connect_probs):
            continue
        elif ntype not in num_types:
            continue
        elif compute not in avg_computes:
            continue
        elif byte not in avg_bytes:
            continue
        elif not close_to_any(bcomp, b_comps):
            continue
        elif not close_to_any(bcomm, b_comms):
            continue
        else:
            res.append(fn)

    return res

