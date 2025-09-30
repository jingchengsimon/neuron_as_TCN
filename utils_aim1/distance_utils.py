from neuron import h

def distance_synapse_mark_compare(dis_syn_from_ctr, dis_mark_from_ctr):
    # 创建一个包含原始索引的列表
    original_indices = list(range(len(dis_syn_from_ctr)))
    index = []

    for value in dis_mark_from_ctr:
        # 计算与value差值最小的元素的索引
        min_index = min(original_indices, key=lambda i: abs(dis_syn_from_ctr[i] - value)) 
        # 将该索引加入结果列表，并从original_indices中移除
        index.append(min_index)
        original_indices.remove(min_index)
    
    return index

def recur_dist_to_soma(sec, loc, initial=True): 
        
    if h.SectionRef(sec).has_parent():
        sec_len = sec.L * loc if initial else sec.L
        parent_sec = h.SectionRef(sec).parent
        return sec_len + recur_dist_to_soma(parent_sec, loc, initial=False)
    else:
        # If there is no parent, the section is the soma, return half lenth of the soma
        # Calculate the distance from the center of the soma
        return sec.L * 0.5 
    
def recur_dist_to_root(sec, loc, root, initial=True): 

    if sec == root:
        # Calculate the distance from the apical nexus (the end of the apical trunk)
        return 0
       
    if h.SectionRef(sec).has_parent():
        sec_len = sec.L * loc if initial else sec.L
        parent_sec = h.SectionRef(sec).parent
        return sec_len + recur_dist_to_root(parent_sec, loc, root, initial=False)
    
