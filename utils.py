import os
import h5py

def split_list(lst, chunks):
    result = []
    chunk_size = int(len(lst) / chunks)
    for i in range(0, len(lst), chunk_size):
        result.append(lst[i:i+chunk_size])
    return result

def list_h5_files(data_folder):
    # Get a list of all files in the folder
    files = os.listdir(data_folder)
    
    # Filter the list of files by extension
    h5_files = [f for f in files if f.endswith('.h5')]
    
    return h5_files

def count_videos(data_folder):
    list_files = list_h5_files(data_folder)
    n_videos = 0
    for file_name in list_files:
        data_path = os.path.join(data_folder, file_name)
        try:
            with h5py.File(data_path, "r") as f:
                for class_label in f.keys():
                    n_videos = n_videos + len(f[class_label].keys())
        except:
            pass
    return n_videos

def get_batch_sizes_in_power_of_2(powers_of_two: range):
    batch_sizes = []
    for i in powers_of_two:
        batch_sizes.append(2**i)
    return batch_sizes