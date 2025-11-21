from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils_aim1.generate_pink_noise import make_noise

# Match segments with function groups of pink noise
def generate_init_firing(section_synapse_df, DURATION, FREQ_EXC, FREQ_INH,
                         input_ratio_basal_apic, num_func_group, spk_epoch_idx, spat_condition):

    sec_syn_bg_exc_df = section_synapse_df[section_synapse_df['type'].isin(['A'])]
    num_syn_bg_exc = len(sec_syn_bg_exc_df)

    segments_dend_df = pd.read_csv('all_segments_dend.csv')
    num_func_group = 2 # segments_dend_df.shape[0] #num_func_group # (26,000/5)/100 = 52

    pink_noise_array = make_noise(num_traces=num_func_group, num_samples=DURATION, spk_epoch_idx=spk_epoch_idx, scale=0.5)
    
    num_segments = segments_dend_df.shape[0]
    exc_firing_rate_array = np.zeros((num_segments, DURATION))  # Shape: (NUM_SYN_BASAL_EXC, DURATION)
    inh_firing_rate_array = np.zeros((num_segments + 1, DURATION))  # Shape: (NUM_SYN_BASAL_EXC, DURATION)
    
    def process_section(i):    
        section = sec_syn_bg_exc_df.iloc[i]
        
        if spat_condition == 'clus':
            # Use np.random.poisson to generate the spike counts independently
            # Remember to divide by 1000 to get the rate per ms

            section_synapse, loc = section['section_synapse'], section['loc']
            matching_segments = segments_dend_df[segments_dend_df['section_name'] == str(section_synapse)]
                    
            if not matching_segments.empty:
                # 找到最接近的 x_position
                x_positions = matching_segments['x_position'].values
                closest_idx = np.argmin(np.abs(x_positions - loc))
                segment_index = matching_segments.index[closest_idx]

            region_index = 0 if section['region'] == 'basal' else 1

            # Random choose a pink noise trace and rectify it to remove negative values and rescaled its mean to 1
            pink_noise = pink_noise_array[region_index]
            pink_noise[pink_noise<0] = 0
            pink_noise = pink_noise/np.mean(pink_noise)

            # variation_factor = 10  # 可调，越大分布越宽        
            # pink_noise = (pink_noise - 1) * variation_factor + 1
            # pink_noise[pink_noise < 0] = 0
            # pink_noise = pink_noise / np.mean(pink_noise)

            exc_firing_rate = FREQ_EXC/1000 * pink_noise if section['region'] == 'basal' else FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise
            exc_firing_rate_array[segment_index] += exc_firing_rate

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(executor.map(process_section, range(num_syn_bg_exc)))
        # executor.map(process_section, range(num_syn_bg_exc))

    inh_delay = 4
    inh_firing_rate_array[1:,inh_delay:] = FREQ_INH/1000 * exc_firing_rate_array[:,:DURATION-inh_delay]/np.mean(exc_firing_rate_array[:,:DURATION-inh_delay])
    inh_firing_rate_array[0,inh_delay:] = FREQ_INH/1000 * exc_firing_rate_array[0,:DURATION-inh_delay]/np.mean(exc_firing_rate_array[:,:DURATION-inh_delay]) # np.mean(exc_firing_rate_array[0])

    return exc_firing_rate_array, inh_firing_rate_array

