import numpy as np
from scipy import signal as ss
from scipy.stats import zscore
import matplotlib.pyplot as plt

def make_noise(num_traces=10, num_samples=1000, spk_epoch_idx=42, scale=0.5):
    np.random.seed(spk_epoch_idx)
    
    num_samples = num_samples+2000
    # Normalised Frequencies
    # fv = np.linspace(0, 1, 40)
    # Amplitudes Of '1/f'                                
    # a = 1/(1+2*fv)                
    # Filter Numerator Coefficients              
    # b = ss.firls(43, fv, a)                                   

    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    A = [1, -2.494956002,   2.017265875,  -0.522189400]

    invfn = np.zeros((num_traces,num_samples))

    for i in np.arange(0, num_traces):
        # Create White Noise
        wn = np.random.normal(loc=1, scale=scale, size=num_samples)  # scale=0.5
        # Create '1/f' Noise
        invfn[i,:] = zscore(ss.lfilter(B, A, wn))                          

    return invfn[:,2000:]

def visualize_pink_noise(DURATION=1000):
    FREQ_EXC = 8000/639
    for scale in [20]:
        sample_list = make_noise(num_traces=2, num_samples=DURATION, scale=scale)

        # sample = sample_list[2]
        # # print(np.mean(sample))

        # sample[sample<0] = 0
        # sample = sample/np.mean(sample)
        # # print(np.mean(sample))
        
        # # spk_rnd = np.random.default_rng(42)
        # # counts_ori = np.random.default_rng(42).poisson(FREQ_EXC/1000, DURATION)


        spk_rnd = np.random.default_rng(42)
        num_basal_segments = 262 #639
        num_apic_segments = 639 - 262
        spike_train_bg_list = []
        for _ in range(num_basal_segments):
            sample = sample_list[0]
            sample[sample<0] = 0
            sample = sample/np.mean(sample)
            counts = spk_rnd.poisson(FREQ_EXC/1000 * sample)
            spike_train_bg = np.where(counts >= 1)[0] # ndarray
            mask = spk_rnd.choice([True, False], size=spike_train_bg.shape, p=[0.5, 0.5])
            spike_train_bg = spike_train_bg[mask]
            spike_train_bg_list.append(spike_train_bg)
        
        for _ in range(num_apic_segments):
            sample = sample_list[1]
            sample[sample<0] = 0
            sample = sample/np.mean(sample)
            counts = spk_rnd.poisson(FREQ_EXC/1000 * sample)
            spike_train_bg = np.where(counts >= 1)[0] # ndarray
            mask = spk_rnd.choice([True, False], size=spike_train_bg.shape, p=[0.5, 0.5])
            spike_train_bg = spike_train_bg[mask]
            spike_train_bg_list.append(spike_train_bg)
        
        n_rows = 4
        plt.subplots(figsize=(10, 6.5), nrows=n_rows, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 0.5, 2, 2]})
        plt.subplot(n_rows,1,1)
        plt.ylim(0, 4)
        plt.plot(sample_list[0],color='b',alpha=0.7,label='basal')
        plt.plot(sample_list[1],color='r',alpha=0.7,label='apical')
        plt.legend()

        # plt.subplot(4,1,2)
        # plt.ylim(0, 1)
        # plt.plot(counts_ori)

        plt.subplot(n_rows,1,2)
        plt.ylim(0, 1)
        plt.plot(counts)

        plt.subplot(n_rows,1,3)
        for i in range(num_basal_segments):
            plt.vlines(spike_train_bg_list[i], i+0.5, i+1.5, color='b',linewidth=5)

        plt.subplot(n_rows,1,4)
        for i in range(num_apic_segments):
            plt.vlines(spike_train_bg_list[i], i+0.5, i+1.5, color='r',linewidth=5)
        # print(spike_train_bg)

        plt.tight_layout()

        # Remove top and right spines
        axs = plt.gcf().get_axes()
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # plt.show()
