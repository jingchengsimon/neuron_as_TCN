import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing
import itertools
from itertools import product
import networkx as nx

from utils_aim1.graph_utils import set_graph_order
from utils_aim1.generate_init_firing_utils import generate_init_firing


class CellWithNetworkx:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, epoch_idx):
    
        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_soma_inh = 0

        # 随机种子设置
        self.spk_epoch_idx = epoch_idx
        self.epoch_idx = 42  # 固定种子用于位置
        self.rnd = np.random.default_rng(self.epoch_idx)
        random.seed(self.epoch_idx)
        
        # 频率计算
        spk_rnd = np.random.default_rng(self.spk_epoch_idx)
        ratio = spk_rnd.uniform(0.4, 1.6)
        self.FREQ_EXC = bg_exc_freq * ratio
        self.FREQ_INH = self.FREQ_EXC * bg_inh_freq/bg_exc_freq

        self.SIMU_DURATION = SIMU_DURATION
        self.STIM_DURATION = STIM_DURATION

        # 数据结构初始化
        self.section_synapse_df = pd.DataFrame(columns=['section_id_synapse', 'section_synapse', 
                                                       'segment_synapse', 'loc', 'type', 'region'], dtype=object)
        self.lock = threading.Lock()
        
        # 加载神经元结构
        section_data = np.load('section_df.npy', allow_pickle=True)
        self.section_df = pd.DataFrame(section_data, columns=['parent_id', 'section_id', 'parent_name', 
                                                             'section_name', 'length', 'branch_idx', 'section_type'])
        self.DiG = nx.read_graphml('DiG.graphml')
        self.DiG = nx.relabel_nodes(self.DiG, {str(i): i for i in range(len(self.DiG.nodes()))})
        
        # 创建区域sections
        self.sections_basal = [(row, ) for _, row in self.section_df[self.section_df['section_type'] == 'dend'].iterrows()]
        self.sections_apical = [(row, ) for _, row in self.section_df[self.section_df['section_type'] == 'apic'].iterrows()]
        self.sections_soma = [(row, ) for _, row in self.section_df[self.section_df['section_type'] == 'soma'].iterrows()]
        
        # 设置section顺序
        self.root_tuft_idx = 121
        self.class_dict_soma, self.class_dict_tuft = set_graph_order(self.DiG, self.root_tuft_idx)
        self.sec_tuft_idx = list(itertools.chain(*self.class_dict_tuft.values()))

    def add_synapses(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh, num_syn_soma_inh):
        # 保存突触数量
        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        self.num_syn_soma_inh = num_syn_soma_inh
        
        # 添加突触
        self.add_single_synapse(num_syn_basal_exc, 'basal', 'exc')
        self.add_single_synapse(num_syn_apic_exc, 'apical', 'exc')
        self.add_single_synapse(num_syn_basal_inh, 'basal', 'inh')
        self.add_single_synapse(num_syn_apic_inh, 'apical', 'inh')
        self.add_single_synapse(num_syn_soma_inh, 'soma', 'inh')
                                    
    def add_single_synapse(self, num_syn, region, sim_type):
        syn_type = 'A' if sim_type == 'exc' else 'B'
        region_map = {'basal': ('dend', self.sections_basal), 
                      'apical': ('apic', self.sections_apical), 
                      'soma': ('soma', self.sections_soma)}
        
        section_type, sections = region_map[region]
        section_length = self.section_df[self.section_df['section_type'] == section_type]['length'].values

        def generate_synapse(_):
            section_row = random.choices(sections, weights=section_length)[0][0]
            loc = self.rnd.uniform()
            
            data = {'section_id_synapse': section_row['section_id'],
                   'section_synapse': section_row['section_name'],
                   'segment_synapse': f"{section_row['section_name']}({loc})",
                   'loc': loc, 'type': syn_type, 'region': region}
            
            with self.lock:
                self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame([data], dtype=object)], ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(executor.map(generate_synapse, range(num_syn)))

    def add_inputs(self, spat_condition, input_ratio_basal_apic, num_func_group):
        exc_firing_rate_array, inh_firing_rate_array = generate_init_firing(
            self.section_synapse_df, self.SIMU_DURATION, self.FREQ_EXC, self.FREQ_INH,
            input_ratio_basal_apic, num_func_group, self.spk_epoch_idx, spat_condition)
        
        return np.concatenate((exc_firing_rate_array, inh_firing_rate_array), axis=0)
      
def generate_simu_params_REAL(spat_cond='clus'):

    NUM_SYN_BASAL_EXC = [10042] # [10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] # [16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    NUM_SYN_SOMA_INH = [150]

    SIMU_DURATION = [400] # 1000 ms
    STIM_DURATION = [400] # 1000 ms

    bg_exc_channel_type = ['AMPANMDA']
    spat_condition = [spat_cond] # clus or distr

    ## Invivo params
    bg_exc_freq = [1] # 1.3 basal ->0.5 1/4
    bg_inh_freq = [4] # 4 basal the ratio of exc/inh is 1:4 for both basal and apical
    input_ratio_basal_apic = [1]
    num_func_group = [10]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH,
                                SIMU_DURATION,
                                STIM_DURATION,
                                spat_condition, 
                                bg_exc_freq, 
                                bg_inh_freq, 
                                input_ratio_basal_apic, 
                                bg_exc_channel_type, 
                                num_func_group))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'NUM_SYN_SOMA_INH',
                                            'SIMU DURATION',
                                            'STIM DURATION',
                                            'synaptic spatial condition',
                                            'background excitatory frequency',
                                            'background inhibitory frequency',
                                            'input ratio of basal to apical',
                                            'background excitatory channel type',
                                            'number of functional groups'])
    
    df['folder_tag'] = (df.index + 1).astype(str)
    params_list = [{**df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list

def build_cell(**params):
    
    # 从参数中提取需要的值
    NUM_SYN_BASAL_EXC = params['NUM_SYN_BASAL_EXC']
    NUM_SYN_APIC_EXC = params['NUM_SYN_APIC_EXC']
    NUM_SYN_BASAL_INH = params['NUM_SYN_BASAL_INH']
    NUM_SYN_APIC_INH = params['NUM_SYN_APIC_INH']
    NUM_SYN_SOMA_INH = params['NUM_SYN_SOMA_INH']
    SIMU_DURATION = params['SIMU DURATION']
    STIM_DURATION = params['STIM DURATION']
    bg_exc_freq = params['background excitatory frequency']
    bg_inh_freq = params['background inhibitory frequency']
    epoch = params['epoch']

    spat_condtion = params['synaptic spatial condition']
    input_ratio_basal_apic = params['input ratio of basal to apical']
    bg_exc_channel_type = params['background excitatory channel type']
    num_func_group = params['number of functional groups']
    
    swc_file_path = '../NeuronWithNetworkx/modelFile/cell1.asc'

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, epoch)
    
    cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                       NUM_SYN_APIC_EXC, 
                       NUM_SYN_BASAL_INH, 
                       NUM_SYN_APIC_INH,
                       NUM_SYN_SOMA_INH)
    
    # firing_rate_array = cell1.add_inputs(spat_condtion, input_ratio_basal_apic, bg_exc_channel_type, num_func_group)
    
    return cell1 # firing_rate_array

def run_simulation_batch(num_runs=2, epoch=1, rebuild_cell=False):
    """批量运行仿真"""
    all_firing_arrays = []
    cell = None
    
    try:
        print(f"使用epoch={epoch}作为随机种子...")
        
        for run_idx in range(num_runs):
            if (run_idx+1) % 50 == 0:
                print(f"  执行第 {run_idx+1}/{num_runs} 次运行")
            params = generate_simu_params_REAL('clus')[0]
            params['epoch'] = epoch  # 使用外部传入的epoch作为随机种子
            
            if rebuild_cell or cell is None:
                # 重建模式：每次都创建；优化模式：只在第一次创建
                cell = build_cell(**params)
            
            # 生成firing_rate_array
            firing_rate_array = cell.add_inputs(params['synaptic spatial condition'], 
                                               params['input ratio of basal to apical'], 
                                               params['number of functional groups'])
            
            all_firing_arrays.append(firing_rate_array)
        
        if all_firing_arrays:
            combined_array = np.stack(all_firing_arrays, axis=0)
            return combined_array
        else:
            return np.array([])
            
    except Exception as e:
        print(f"仿真运行过程中出现错误: {e}")
        return np.array([])

if __name__ == "__main__":
    print("=== 测试rebuild_cell=False (优化模式) ===")
    # 运行3次，使用epoch=42作为随机种子
    all_firing_arrays = run_simulation_batch(num_runs=3, epoch=42, rebuild_cell=False)
    print(f"结果形状: {all_firing_arrays.shape}")
    
    # print("\n=== 测试rebuild_cell=True (重建模式) ===")
    # # 运行2次，使用epoch=100作为随机种子
    # all_firing_arrays = run_simulation_batch(num_runs=2, epoch=100, rebuild_cell=True)
    # print(f"结果形状: {all_firing_arrays.shape}")



    