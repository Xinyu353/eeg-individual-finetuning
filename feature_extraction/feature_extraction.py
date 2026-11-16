import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import matplotlib.pyplot as plt
from EntropyHub import SpecEn, ApEn
import os
import scipy

def hjorth_parameters(signal, epsilon=1e-10):
    """
    Calculate Hjorth parameters
    
    Parameters:
        signal: 输入信号
        epsilon: 数值稳定性参数，防止除以零
    Returns:
        activity: 信号的方差
        mobility: 信号一阶差分的标准差与信号标准差的比值
        complexity: 信号二阶差分的mobility与一阶差分的mobility的比值
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    # 计算差分
    first_diff = np.diff(signal)
    second_diff = np.diff(first_diff)
    
    # 计算方差
    var_signal = np.var(signal)
    var_first = np.var(first_diff)
    var_second = np.var(second_diff)
    
    # 计算三个参数
    activity = var_signal
    mobility = np.sqrt(var_first / (var_signal + epsilon))
    complexity = np.sqrt(var_second / (var_first + epsilon)) / (mobility + epsilon)
    
    return activity, mobility, complexity

def petrosian_fractal_dimension(signal, epsilon=1e-10):
    """Calculate Petrosian Fractal Dimension"""
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    N = len(signal)
    # 使用更高效的方式计算符号变化
    diff_signal = np.diff(signal)
    M = np.sum(diff_signal[1:] * diff_signal[:-1] < 0)
    
    PFD = np.log(N + epsilon) / (np.log(N + epsilon) + np.log((N + epsilon) / (N + 0.4 * M + epsilon)))
    return PFD

def psd_bandpower(signal, fs=256, bands=None):
    """
    Calculate PSD bandpower for different frequency bands
    
    Parameters:
        signal: 输入信号
        fs: 采样率
        bands: 频带定义字典
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),    # Delta: 0.5-4 Hz
            'theta': (4, 8),      # Theta: 4-8 Hz
            'alpha': (8, 13),     # Alpha: 8-13 Hz
            'beta': (13, 30),     # Beta: 13-30 Hz
            'gamma': (30, 40),    # Gamma: 30-40 Hz
            'all': (0, 40)        # All: 0-40 Hz
        }
    
    # 窗口长度选择
    nperseg = min(len(signal), fs)  # 使用1秒数据
    noverlap = nperseg // 2
    
    # 使用scipy.signal.welch而不是signal.welch
    f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 使用梯形积分计算带功率
    bandpower = {}
    for band, (low, high) in bands.items():
        idx_band = np.logical_and(f >= low, f <= high)
        bandpower[band] = np.trapz(Pxx[idx_band], f[idx_band])
    
    return tuple(bandpower.get(band, 0) for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all'])

def baseline_correction(signal, times, baseline_window=(-400, 0)):
    """
    对单个trial进行基线校正
    
    Parameters:
        signal: 输入信号
        times: 时间点（ms）
        baseline_window: 基线期窗口
    """
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    baseline_mean = np.mean(signal[baseline_mask])
    return signal - baseline_mean

def mexican_hat(times, lag, scale):
    """Mexican hat wavelet function"""
    t = (times - lag) / scale
    return (1 - 16 * t ** 2) * np.exp(-8 * t ** 2)

def compute_Wst(data, times, plot_on=True, save_path=None):
    """
    计算小波变换
    
    Parameters:
    -----------
    data : array-like
        输入信号
    times : array-like
        时间点数组
    plot_on : bool
        是否绘制时频图
    save_path : str
        保存图片的路径
        
    Returns:
    --------
    Wst : array-like
        小波变换结果
    tlag : array-like
        时间轴
    scale : array-like
        尺度轴
    """
    if data.shape != times.shape:
        raise ValueError("Data and times should be same!")

    srate = compute_srate(times)
    tlag = np.linspace(0, 1000, len(times))
    scale = np.linspace(1, 3500, len(times))

    Wst = np.zeros((len(scale), len(tlag)))

    for ti, t in enumerate(tlag):
        for si, s in enumerate(scale):
            wavelet = mexican_hat(times, t, s)
            Wst[si, ti] = np.sum(data * wavelet) / np.sqrt(s)

    if plot_on:
        # 创建1x2的子图，调整右图比例为1:1
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        plt.rcParams['font.family'] = 'Arial'
        
        # 左图：ERP曲线
        ax1.plot(times, data, color='blue', linewidth=1.5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude (µV)')
        ax1.set_title('ERP Waveform')
        # 设置x轴范围
        ax1.set_xlim(times[0], times[-1])
        # 添加零线
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        # 添加刺激开始线
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 右图：WST时频图
        im = ax2.imshow(Wst, aspect='auto', extent=[tlag[0], tlag[-1], scale[0], scale[-1]], 
                       origin='lower', cmap='coolwarm')
        plt.colorbar(im, ax=ax2)
        ax2.contour(tlag, scale, Wst, colors='k', linewidths=0.5)
        
        # 找到局部最大值
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(Wst, size=3) == Wst
        max_points = np.where(local_max)
        # 只保留显著的最大值点（可以调整阈值）
        threshold = np.max(Wst) * 0.5  # 使用最大值的50%作为阈值
        significant_max = Wst[max_points] > threshold
        max_points = (max_points[0][significant_max], max_points[1][significant_max])
        
        # 绘制局部最大值点
        ax2.scatter(tlag[max_points[1]], scale[max_points[0]], 
                   color='black', s=20, marker='o', alpha=0.6)
        
        ax2.set_ylabel('Scale (ms)')
        ax2.set_xlabel('Time lag (ms)')
        ax2.set_title('WST Color Map')
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
        plt.close()

    return Wst, tlag, scale

def compute_srate(times):
    """计算采样率"""
    return 1000 / np.mean(np.diff(times))

def validate_p3(amplitude, latency, background_std):
    """
    验证检测到的P3是否有效
    """
    if amplitude is None or latency is None:
        return False
        
    # 检查潜伏期是否在合理范围
    if not (250 <= latency <= 500):
        return False
    
    # 检查振幅是否显著（降低阈值）
    if amplitude < 1.5 * background_std:
        return False
    
    return True

def extract_p3_features(signal, times, plot=False, save_path=None):
    """
    提取P3特征
    
    Parameters:
        signal: 输入信号
        times: 时间点序列
        plot: 是否绘制时频图
        save_path: 图像保存路径
    Returns:
        p3_amplitude: P3振幅
        p3_latency: P3潜伏期
    """
    # 计算小波变换
    Wst, tlag, scale = compute_Wst(signal, times, plot_on=plot, save_path=save_path)
    
    # 定义P3搜索窗口
    p3_window = (400 <= tlag) & (tlag <= 500)
    
    if np.any(p3_window):
        # 在P3时间窗口中搜索
        window_data = Wst[:, p3_window]
        
        if window_data.size > 0:
            # 使用稳健小波面积法
            # 计算每个尺度下的时间积分
            area_by_scale = np.trapz(window_data, axis=1)
            # 找到最大积分对应的尺度
            max_scale_idx = np.argmax(area_by_scale)
            p3_amplitude = area_by_scale[max_scale_idx]
            
            # 获取潜伏期（使用最大值点的时间）
            p3_latency = tlag[p3_window][np.argmax(window_data[max_scale_idx])]
            
            return p3_amplitude, p3_latency
    
    return np.nan, np.nan

def process_single_trial(signal, times, plot=False, save_path=None):
    """
    处理单个trial的数据
    
    Parameters:
    -----------
    signal : array-like
        原始信号
    times : array-like
        时间点序列
    plot : bool
        是否绘制时频图
    save_path : str
        图像保存路径
    """
    # 进行基线校正
    signal = baseline_correction(signal, times)
    
    # 计算小波变换
    Wst, tlag, scale = compute_Wst(signal, times, plot_on=plot, save_path=save_path)
    
    # 提取特征
    p3_amp, p3_lat = extract_p3_features(signal, times, plot=plot, save_path=save_path)
    
    return {
        'P3_Amplitude': p3_amp,
        'P3_Latency': p3_lat
    }

def calculate_band_entropy(signal, fs=256, bands=None):
    """
    Calculate spectral entropy for different frequency bands
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
    
    # 使用更长的窗口来提高频率分辨率
    nperseg = min(len(signal), 4 * fs)  # 4秒窗口
    noverlap = nperseg // 2
    
    # 计算功率谱密度
    f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, 
                               noverlap=noverlap, window='hann')
    
    # 对高频成分进行衰减（模拟1/f噪声校正）
    Pxx = Pxx / (f + 1)
    
    # 计算各频带的熵
    band_entropy = {}
    for band_name, (low, high) in bands.items():
        # 找到频带对应的索引
        idx_band = np.logical_and(f >= low, f <= high)
        if np.any(idx_band):
            # 提取该频带的功率谱并归一化
            Pxx_band = Pxx[idx_band]
            Pxx_band = Pxx_band / np.sum(Pxx_band)
            # 计算该频带的熵
            band_entropy[band_name] = entropy(Pxx_band)
        else:
            band_entropy[band_name] = 0
    
    # 计算总体谱熵
    Pxx_norm = Pxx / np.sum(Pxx)
    band_entropy['total'] = entropy(Pxx_norm)
    
    return band_entropy

def extract_all_features(signal, times, fs=256):
    """Extract all features from a signal"""
    # 进行基线校正
    signal = baseline_correction(signal, times)
    
    # Basic features
    activity, mobility, complexity = hjorth_parameters(signal)
    pfd = petrosian_fractal_dimension(signal)
    
    # PSD features
    delta_psd, theta_psd, alpha_psd, beta_psd, gamma_psd, all_psd = psd_bandpower(signal, fs=fs)
    
    # 计算熵特征
    entropy_features = calculate_band_entropy(signal, fs=fs)
    
    # ERP features (P3)
    p3_amplitude, p3_latency = extract_p3_features(signal, times)
    
    return {
        'Activity': activity,
        'Mobility': mobility,
        'Complexity': complexity,
        'Petrosian Fractal Dimension': pfd,
        'P3 Amplitude': p3_amplitude,
        'P3 Latency': p3_latency,
        'Delta PSD': delta_psd,
        'Theta PSD': theta_psd,
        'Alpha PSD': alpha_psd,
        'Beta PSD': beta_psd,
        'Gamma PSD': gamma_psd,
        'All PSD': all_psd,
        'SpecEntropy': entropy_features['total'],
        'Delta Entropy': entropy_features['delta'],
        'Theta Entropy': entropy_features['theta'],
        'Alpha Entropy': entropy_features['alpha'],
        'Beta Entropy': entropy_features['beta'],
        'Gamma Entropy': entropy_features['gamma']
    }

def extract_erp_features(Wst, tlag, scale, times):
    """
    提取ERP特征（P1, N1, P3的振幅和潜伏期）
    
    Parameters:
    -----------
    Wst : array-like
        小波变换结果
    tlag : array-like
        时间轴
    scale : array-like
        尺度轴
    times : array-like
        时间点序列
        
    Returns:
    --------
    p1_amp : float
        P1振幅
    n1_amp : float
        N1振幅
    p3_amp : float
        P3振幅
    p1_lat : float
        P1潜伏期
    n1_lat : float
        N1潜伏期
    p3_lat : float
        P3潜伏期
    """
    # P1窗口 (50-150ms)
    p1_window = (120 <= tlag) & (tlag <= 130)
    # N1窗口 (100-200ms)
    n1_window = (175 <= tlag) & (tlag <= 185)
    # P3窗口 (300-600ms)
    p3_window = (400 <= tlag) & (tlag <= 500)
    
    # 初始化返回值
    p1_amp, n1_amp, p3_amp = np.nan, np.nan, np.nan
    p1_lat, n1_lat, p3_lat = np.nan, np.nan, np.nan
    
    # 提取P1特征
    if np.any(p1_window):
        window_data = Wst[:, p1_window]
        if window_data.size > 0:
            max_scale_idx = np.argmax(np.max(window_data, axis=0))
            max_time_idx = np.argmax(window_data[max_scale_idx])
            p1_amp = window_data[max_scale_idx, max_time_idx]
            p1_lat = tlag[p1_window][max_time_idx]
    
    # 提取N1特征
    if np.any(n1_window):
        window_data = Wst[:, n1_window]
        if window_data.size > 0:
            min_scale_idx = np.argmin(np.min(window_data, axis=0))
            min_time_idx = np.argmin(window_data[min_scale_idx])
            n1_amp = window_data[min_scale_idx, min_time_idx]
            n1_lat = tlag[n1_window][min_time_idx]
    
    # 提取P3特征
    if np.any(p3_window):
        window_data = Wst[:, p3_window]
        if window_data.size > 0:
            max_scale_idx = np.argmax(np.max(window_data, axis=0))
            max_time_idx = np.argmax(window_data[max_scale_idx])
            p3_amp = window_data[max_scale_idx, max_time_idx]
            p3_lat = tlag[p3_window][max_time_idx]
    
    return p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat

def preprocess_signal(signal):
    # 检查NaN值
    if np.any(np.isnan(signal)):
        return None
    # 检查数据长度
    if len(signal) != 409:
        return None
    return signal

def process_csv_file(input_file, output_file, fs=256):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Get the first column (sample IDs)
        sample_ids = df.iloc[:, 0]
        
        # Get the time series data (all columns except the first)
        time_series_data = df.iloc[:, 1:]
        
        # Create time points array
        times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000
        
        # Initialize list to store features
        features_list = []
        
        # Process each row
        for idx, (sample_id, row) in enumerate(zip(sample_ids, time_series_data.itertuples(index=False))):
            try:
                if idx % 100 == 0:
                    print(f'Processing sample {idx + 1}/{len(sample_ids)}')
                
                # Convert row to numpy array
                signal = np.array(row)
                
                # Preprocess signal
                signal = preprocess_signal(signal)
                if signal is None:
                    print(f"Warning: Invalid signal at index {idx}")
                    continue
                
                # Extract features
                features = extract_all_features(signal, times, fs=fs)
                
                # Add sample ID to features
                features['SampleID'] = sample_id
                
                # Append to features list
                features_list.append(features)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Convert features list to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Reorder columns to put SampleID first
        cols = ['SampleID'] + [col for col in features_df.columns if col != 'SampleID']
        features_df = features_df[cols]
        
        # Save features to CSV
        features_df.to_csv(output_file, index=False)
        print(f'Features saved to {output_file}')
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return 

def process_averaged_trials(df, srate=256, scale_factor=1, trials_per_group=20, plot=False, save_path=None, random_seed=None):
    """
    将trials分组并计算平均值，然后使用小波方法提取特征
    
    Parameters:
    -----------
    df : DataFrame
        输入数据，包含EEG信号
    srate : int
        采样率
    scale_factor : float
        尺度因子
    trials_per_group : int
        每组包含的trials数量
    plot : bool
        是否绘制时频图
    save_path : str
        图像保存路径
    random_seed : int
        随机种子，用于打乱trials
    Returns:
    --------
    DataFrame : 包含特征的DataFrame
    """
    # 判断是否有participant编号
    if len(df.columns) == 410:  # 有编号的情况
        time_series_data = df.iloc[:, 1:]  # 其余列是EEG数据
    elif len(df.columns) == 409:  # 没有编号的情况
        time_series_data = df  # 所有列都是EEG数据
    else:
        raise ValueError(f"Unexpected number of columns: {len(df.columns)}")
    
    # 放大EEG数据
    time_series_data = time_series_data * scale_factor
    
    # 创建时间点数组
    times = np.linspace(-0.4, 1.2, time_series_data.shape[1]) * 1000  # 转换为毫秒
    
    # 转换为numpy数组
    signals = time_series_data.values
    
    # 随机打乱trials
    if random_seed is not None:
        np.random.seed(random_seed)
        n_trials = signals.shape[0]
        indices = np.random.permutation(n_trials)
        signals = signals[indices]
    else:
        indices = np.arange(signals.shape[0])
    
    features_list = []
    
    # 计算可以完整处理的组数
    n_complete_groups = signals.shape[0] // trials_per_group
    
    # 处理完整的组
    for group_idx in range(n_complete_groups):
        start_idx = group_idx * trials_per_group
        end_idx = start_idx + trials_per_group
        
        # 获取当前组的trials
        group_signals = signals[start_idx:end_idx]
        
        # 对每个trial进行基线校正
        corrected_signals = np.array([baseline_correction(signal, times) for signal in group_signals])
        
        # 计算平均值
        averaged_signal = np.mean(corrected_signals, axis=0)
        
        # 设置当前组的保存路径
        current_save_path = f"{save_path}_group_{group_idx}.png" if save_path else None
        
        # 计算小波变换并提取特征
        Wst, tlag, scale = compute_Wst(averaged_signal, times, plot_on=plot, save_path=current_save_path)
        p3_amp, p3_lat = extract_p3_features(averaged_signal, times, plot=plot, save_path=current_save_path)
        
        # 创建特征字典
        features = {
            'GroupID': group_idx,
            'TrialIndices': f"{indices[start_idx]}-{indices[end_idx-1]}",  # 保存原始索引
            'P3_Amplitude': p3_amp,
            'P3_Latency': p3_lat
        }
        features_list.append(features)
        
        # 打印进度
        print(f"Processed group {group_idx + 1}/{n_complete_groups}")
        if plot and current_save_path:
            print(f"Saved plot to {current_save_path}")
    
    # 转换为DataFrame
    return pd.DataFrame(features_list) 