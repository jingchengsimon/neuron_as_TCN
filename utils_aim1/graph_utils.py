import networkx as nx
import re
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import json

from Bio import Phylo
from Bio.Phylo.PhyloXML import Phylogeny, Clade

def create_directed_graph(all_sections, all_segments, section_df):
    parent_list, parent_index_list = [], []
    
    basal_branch_init_sec_list, apic_branch_init_sec_list = [], []

    for i, section_segment_list in enumerate(all_sections):
        section = section_segment_list[0].sec
        section_id = i
        section_name = section.psection()['name']
        match = re.search(r'\.(.*?)\[', section_name)
        section_type = match.group(1)
        L = section.psection()['morphology']['L']

        parent_list.append(section_name)
        parent_index_list.append(section_id)

        if i == 0:
            parent_name = 'None'
            parent_id = 0
            branch_idx = -1
            
        else:
            parent = section.parentseg().sec #section.psection()['morphology']['parent'].sec
            parent_name = parent.psection()['name']
            parent_id = parent_index_list[parent_list.index(parent_name)]
        
        # Define branch index (for basal and apical)
        if section_type == 'dend':
            if (parent_id ==  0) & (section not in basal_branch_init_sec_list):
                basal_branch_init_sec_list.append(section)
                branch_idx = basal_branch_init_sec_list.index(section)
            elif (parent_id != 0) & (section_df[section_df['section_name'] == parent.psection()['name']]['branch_idx'].values[0] is not None):
                branch_idx = section_df[section_df['section_name'] == parent.psection()['name']]['branch_idx'].values[0]
            else:
                branch_idx = -1
        
        elif section_type == 'apic':
            if (parent_id == 121) & (section not in apic_branch_init_sec_list):
                apic_branch_init_sec_list.append(section)
                branch_idx = apic_branch_init_sec_list.index(section)
            elif (parent_id != 121) & (section_df[section_df['section_name'] == parent.psection()['name']]['branch_idx'].values[0] is not None):
                branch_idx = section_df[section_df['section_name'] == parent.psection()['name']]['branch_idx'].values[0]
            else:
                branch_idx = -1

        # create data
        data_to_append = {'parent_id': parent_id,
                'section_id': section_id,
                'parent_name': parent_name,
                'section_name': section_name,
                'length': L,
                'branch_idx': branch_idx,
                'section_type': section_type}
        
        section_df = pd.concat([section_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
        
    DiG = nx.from_pandas_edgelist(section_df, source='parent_id', target='section_id',
                                 create_using=nx.DiGraph(), edge_attr='length')
    
    segment_DiG = nx.DiGraph()

    # Step 1: Add intra-section edges
    for section_id, segment_list in enumerate(all_sections):
        num_segments = len(segment_list)
        
        for seg_idx in range(num_segments):
            segment_index = all_segments.index(segment_list[seg_idx])
            segment_DiG.add_node(segment_index)

            # Add intra-section edge
            if seg_idx > 0:
                prev_seg = all_segments.index(segment_list[seg_idx - 1]) 
                segment_DiG.add_edge(prev_seg, segment_index)

    # Step 2: Add inter-section edges (based on parent section)
    for section_id, section_segment_list in enumerate(all_sections):
        section = section_segment_list[0].sec  # assume all segs in list share same section
        if section_id == 0:
            continue  # skip root

        parent_section = section.parentseg().sec
        parent_name = parent_section.psection()['name']
        
        # Locate parent section id
        try:
            parent_id = parent_index_list[parent_list.index(parent_name)]
        except ValueError:
            continue  # parent not found

        # Connect parent section's last segment → this section's first segment
        parent_seg_end = all_segments.index(all_sections[parent_id][-1])  # last segment of parent section
        child_seg_start = all_segments.index(section_segment_list[0]) # first segment of current section
        
        segment_DiG.add_edge(parent_seg_end, child_seg_start)

    # Remove cycles from the graph
    remove_cycles(DiG)
    remove_cycles(segment_DiG)

    # Convert the NetworkX graph to a PhyloXML tree
    # root = 0
    # root_clade = nx_to_clade(DiG, root)
    # phylo_tree = Phylogeny.from_clade(root_clade, rooted=True)

    # # Save the PhyloXML tree to an XML file
    # with open("L5_morphology.xml", "w") as xml_file:
    #     Phylo.write([phylo_tree], xml_file, "phyloxml")

    # segment_DiG_array = nx.to_numpy_array(segment_DiG, nodelist=sorted(segment_DiG.nodes()))  
    # with open('segment_adjacency_matrix.json', 'w') as f:
    #     json.dump(segment_DiG_array.tolist(), f, indent=2)

    return section_df, DiG

def set_graph_order(G, root_tuft):
    
    class_dict_soma = calculate_out_degree(G, 0)

    G_tuft = get_subgraph(G, root_tuft)
    class_dict_tuft = calculate_out_degree(G_tuft, root_tuft)

    return class_dict_soma, class_dict_tuft

def remove_cycles(DiG):
    """Remove cycles from the directed graph."""
    try:
        while True:
            cycle = nx.find_cycle(DiG, orientation='original')
            # print(f"Cycle detected: {cycle}")
            DiG.remove_edge(*cycle[0][:2])
            # print(f"Removed edge: {cycle[0][:2]}")
    except nx.NetworkXNoCycle:
        pass
        # print("No more cycles detected.")

def nx_to_clade(graph, node, visited=None):
    """Recursively convert NetworkX nodes to Bio.Phylo Clade nodes."""
    if visited is None:
        visited = set()
    if node in visited:
        raise ValueError(f"Cycle detected at node {node}")
    visited.add(node)
    
    clade = Clade(name=node)
    for child in graph.successors(node):  # Only directed edges from parent to child
        edge_data = graph.get_edge_data(node, child)
        weight = edge_data['length'] if edge_data else 1.0
        child_clade = nx_to_clade(graph, child, visited)
        child_clade.branch_length = weight
        clade.clades.append(child_clade)
    return clade

def plot_graph(G):
    plt.figure()

    # tree layout
    # Find all descendants of node 121 (apical nexus)
    tuft_nodes = nx.descendants(G, 121) if 121 in G.nodes else set()
    tuft_nodes.add(121)  # Add the apical nexus node itself to the set

    node_colors = []
    for node in G.nodes:
        if node == 0:
            node_colors.append('lightgreen') # lightgreen color for soma node
        elif 1 <= node < 85:
            node_colors.append('skyblue') # skyblue color for basal nodes
        elif node in tuft_nodes:
            node_colors.append('pink')  # pink color for tuft nodes
        else:
            node_colors.append('lightgray')  # lightgray color for nodes outside specified ranges

    # Set positions and draw nodes with specified colors
    pos = graphviz_layout(G, prog="circo")
    nx.draw(G, pos, with_labels=False, node_size=100, node_color=node_colors)

    # Add text labels for each node (representing the node index)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=10, font_color="black")

    
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

    # 标记节点的 order
    # nx.draw_networkx_labels(G, pos, labels=order, font_size=12, font_color='red')
    
    plt.title("Graph Visualization")
    plt.show()

def get_subgraph(G, node):
    subgraph_nodes = nx.descendants(G, node) | {node}   # 使用descendants函数获取所有后代节点
    subgraph = G.subgraph(subgraph_nodes)  # 创建一个包含指定节点及其后代节点的子图

    return subgraph

def calculate_out_degree(G, root_node=0):
    order_dict = nx.single_source_shortest_path_length(G, root_node)

    out_degree = dict(G.out_degree())
    fork_points = {node for node, degree in out_degree.items() if node != root_node and degree > 1}
    
    ## distance to the soma
    class_dict = {}
    # Exclude the root node
    # for node in order_dict.keys():
    for node in (key for key in order_dict.keys() if key != root_node):
        forks_in_path = sum(1 for n in nx.shortest_path(G, source=root_node)[node][1:-1] if n in fork_points)
        order = forks_in_path
        if order not in class_dict:
            class_dict[order] = []
        class_dict[order].append(node)

    return class_dict