import os
from os.path import join, abspath, dirname, splitext
import random
import csv
import numpy as np
import dawdreamer as daw
from soundfile import write
import pandas as pd
import random
from scipy import stats
from scipy.signal import hilbert
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

class InvalidRIRException(Exception):
    """Custom exception for invalid RIRs."""
    pass

def load_config(file_name="config.yaml"):
    path = join(dirname(abspath(__file__)), file_name)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_impulse(length, sample_rate):
    impulse = np.zeros((int(length * sample_rate),), dtype=np.float32)
    impulse[0] = 1.0
    impulse = np.stack([impulse, impulse], axis=1).T
    return impulse

def load_presets(path):
    presets = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            presets.append(file_path)
    return presets

def calc_rt60(h, sr=48000, rt='t30'): 
    """
    RT60 measurement routine acording to Schroeder's method [1].

    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

    Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    h_abs = np.abs(h) / np.max(np.abs(h))

    # Schroeder integration
    sch = np.cumsum(h_abs[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch)+1e-20)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / sr
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60

def validate_rir(rir, sr):
    """
    Validates a Room Impulse Response (RIR).
    Raises an exception if the RIR is invalid.
    """
    # 1. Check for NaN or Inf values
    if np.isnan(rir).any() or np.isinf(rir).any():
        raise InvalidRIRException("Invalid RIR: Contains NaN or Inf values.")

    # 2. Check amplitude range
    if np.max(np.abs(rir)) > 1.0:
        raise InvalidRIRException("RIR contains values outside the range [-1.0, 1.0].")
    
    # 3. Check DC offset (mean value close to zero)
    dc_offset = np.mean(rir)
    if abs(dc_offset) >= 1e-4:
        raise InvalidRIRException(f"DC offset is too high: {dc_offset}")

    # 4. Check energy 
    energy = np.sum(rir**2)
    if energy < 1e-16: 
        raise InvalidRIRException(f"RIR energy too low_ {energy}")

    # 5. Check Lenght (should not be empty)
    if len(rir) == 0 or rir.size == 0:
        raise InvalidRIRException("RIR is empty or has no valid length.")
    
    #rt60 check
    min_rt60 = 0.1
    max_rt60 = 100
    rt60 = calc_rt60(h=rir,sr=sr)
    if not min_rt60 <= rt60 <= max_rt60:
        raise InvalidRIRException(f"Unrealistic RT60: {rt60}s")

    
    envelope = np.abs(hilbert(rir))
    normalized_env = envelope / np.max(envelope)

    early_energy_threshold = 0.8  # Weniger streng als 0.8
    late_energy_threshold = 0.2  # Höher als 0.1
    if np.max(normalized_env[:500]) > early_energy_threshold and np.mean(normalized_env[2000:]) < late_energy_threshold:
        raise InvalidRIRException("RIR energy decays too quickly.")
    
    early_energy_fraction = 0.6  # Höher als 0.5
    if np.sum(np.abs(rir[:1000])) > early_energy_fraction * np.sum(np.abs(rir)):
        raise InvalidRIRException("RIR energy is too concentrated at the beginning.")
    
    return rt60

def normalize(rir):
    if np.max(np.abs(rir)) < 0.1:
        rir = 0.1 * rir / np.max(np.abs(rir))
    elif np.max(np.abs(rir)) > 0.7:
        rir = 0.7 * rir / np.max(np.abs(rir))
    return rir

def render_rir(engine, graph, rir_len, sr, output_path, rir_name):
    engine.load_graph(graph) 
    engine.render(rir_len)
    rir = normalize(engine.get_audio().T)
    rt60 = validate_rir(rir, sr)
    write(join(output_path, rir_name), rir, sr, subtype="FLOAT")
    return rt60

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help='Target directory of the dataset')
    parser.add_argument("--preset_dir", type=str, required=True, help='Directory of the effect presets')
    parser.add_argument("--target_sr", type=int, default=48000, help='Sampling rate')
    parser.add_argument("--buffer_size", type=int, default=512)
    parser.add_argument("--rir_len", type=int, default=10, help='Duration of the resulting RIR in seconds')
    parser.add_argument("--name", type=str, default="ReverbFx", help='Name of the dataset')
    parser.add_argument("--var_num", type=int, default=15, help='Number of parameter variations per preset')
    parser.add_argument("--save_state", type=bool, help='Saves the randomized presets as an state')
    args = parser.parse_args()

preset_dir = args.preset_dir
target_dir = args.target_dir
sr = args.target_sr
rir_len = args.rir_len
buffer_size = args.buffer_size

config = load_config()
plugin_dirs = config["plugin_dirs"]
force_params = config["force_params"]

output_base_path = join(target_dir, args.name)
try:
    os.makedirs(output_base_path)
except Exception as e:
    raise Exception(f"Error creating output folder: {e}")

if(args.save_state):
    out_preset_base = join(output_base_path, "States")
    os.makedirs(out_preset_base)

output_rir_path = join(output_base_path, "RIRs")
os.makedirs(output_rir_path)

log_file = join(output_rir_path, "ART_RIR_LOG.csv")
with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["RIR","Plugin Name","Preset Name", "Randomized Parameters (name_id_value)", "rt60"])

error_log_file = join(output_rir_path, "ERROR_LOG.csv")
with open(error_log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["RIR","Plugin Name","Preset Name", "Randomized Parameters (name_id_value)","Error"])


num_presets = {}
engine = daw.RenderEngine(sr, buffer_size)
impulse = engine.make_playback_processor("impulse", create_impulse(rir_len, sr))

random.seed(42)

for plugin_name, plugin_path in plugin_dirs.items():
    
    if(args.save_state):
        out_preset_path = join(out_preset_base, plugin_name)
        os.makedirs(out_preset_path, exist_ok=True)

    output_path = join(output_rir_path, plugin_name)
    os.makedirs(output_path, exist_ok=True)
    
    plugin = engine.make_plugin_processor(plugin_name, plugin_path)

    graph = [
        (impulse, []),

        (plugin, [impulse.get_name()])
    ]

    presets = load_presets(join(preset_dir, plugin_name))

    forced_plugin_params = force_params[plugin_name]

    parameters = []

    #Search for all parameters, which can be randomized 
    for para in plugin.get_parameters_description():
        if (para["isAutomatable"] and not para["isDiscrete"] and (para["index"] not in forced_plugin_params.keys())):
            parameters.append({
            "name": para["name"],
            "id": para["index"]
        })
    print(f"{len(parameters)} parameters found:")

    for param in parameters:
        print(f"  {param['name']} (ID: {param['id']})")

    num_presets[plugin_name] = len(presets)
 
    for preset in presets:
        if preset.endswith(".fxp"):
            plugin.load_preset(preset)
        elif preset.endswith(".vstpreset"):
            plugin.load_vst3_preset(preset)

        preset_name = os.path.splitext(os.path.basename(preset))[0]

        for id, value in forced_plugin_params.items():
            plugin.set_parameter(id, value)
        
        x=args.var_num #number of randomizations
        if plugin_name == "Protoverb":
            x=5

        for i in range(x):
            parameter_changes = ""
            default_values = []

            if i!=0:
                #Select two random settings to adjust
                if len(parameters) > 2:
                    select_params = random.sample(parameters, 2)
                else:
                    select_params = parameters
                    
                for param in select_params:
                    
                    try:
                        default_values.append((param["id"],plugin.get_parameter(param["id"])))
                        random_value = random.uniform(0,1)
                        plugin.set_parameter(param["id"], random_value)
                        parameter_changes += f"{param['name']}_{param['id']}_{random_value}"
                        
                    except Exception as e:
                        print(f"Error setting parameter '{param['name']}' (ID: {param['id']}): {e}, plugin: {plugin_name}")
                                     

            rir_name = f"{plugin_name}_{preset_name}_{i:02}.wav"
            
            try:
                rt60 = render_rir(engine, graph, rir_len, sr, output_path, rir_name)
                if args.save_state:
                    plugin.save_state(join(out_preset_path, rir_name))
                with open(log_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([rir_name, plugin_name, preset_name, parameter_changes, rt60])
            except Exception as e:
                print(f"Skipping RIR {rir_name} due to error: {e}")
                with open(error_log_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([rir_name, plugin_name, preset_name, parameter_changes, e])

            #Set randomized parameters back to their original values
            if i!=0:
                for dv in default_values:
                    idx, value = dv
                    plugin.set_parameter(idx, value)

    engine.remove_processor(plugin_name)



print(f"{args.name} successfully built.")
for plugin, count in num_presets.items():
    print(f"{plugin}: {count} presets")



