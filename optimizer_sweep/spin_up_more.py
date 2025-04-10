import os
import re
from optimizer_sweep.external_monitor import parse_command_file, calculate_data_tag
from utils_simp import grab_best_run

# Define the root directory to search.
root_dir = "optimizer_sweep"

# This regex assumes filenames follow a pattern such as:
#   exp<digits>_<optimizer>sweep_<model_size>_<target>.py
pattern = re.compile(r"exp\d+_([a-z]+)sweep_(\d+M)_(\d+)\.py$", re.IGNORECASE)

# Set to store unique optimizers
optimizer_name = 'soap'
# Base parameters to filter files.
base_model_size = '130m'
base_target_chinchilla = 2
# New parameters for the generated script.
real_model_size = '130m'
real_target_chinchilla = 4



def rewrite(optimizer_name, base_model_size, base_target_chinchilla, real_model_size, real_target_chinchilla, optimizer_name_2 = None):
    # Walk through the directory tree.
    try:
        if optimizer_name_2 is None:
            optimizer_name_2 = optimizer_name
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # We look for files ending with '_130M_1.py'
                if filename.endswith(f"_{optimizer_name}sweep_{base_model_size.upper()}_{base_target_chinchilla}.py"):
                    match = pattern.match(filename)
                    if match:
                        # Parse the file for configuration data.
                        full_file_path = os.path.join(dirpath, filename)
                        parsed_data = parse_command_file(full_file_path)
                        first_baseline = parsed_data['baseline_config']
                        sweep_grids = parsed_data['sweep_grids']
                        model_size = parsed_data['model_size']
                        target_chinchilla = parsed_data['target_chinchilla']
                        optimizer = parsed_data['optimizer_name']
                        
                        # Calculate additional tags if needed.
                        target_data, data_size = calculate_data_tag(model_size, target_chinchilla)
                        tags = (model_size, data_size, optimizer)
                        
                        # Retrieve the best configuration.
                        current_best_config, approximate_best_config_list = grab_best_run(first_baseline.keys(), tags)
                        print(optimizer, current_best_config)
                        
                        # Create the new filename. Digits are fixed to 1.
                        new_filename = f"exp725_{optimizer_name_2}sweep_{real_model_size.upper()}_{real_target_chinchilla}.py"
                        new_file_path = os.path.join(root_dir, new_filename)
                        print(f"key_of_optimizer['{optimizer_name_2}'] = {list(first_baseline.keys())}")
                        
                        # Generate the new script using current_best_config.
                        with open(new_file_path, "w") as new_file:
                            new_file.write("from optimizer_sweep.template import template\n")
                            new_file.write("\n")
                            new_file.write("if __name__ == '__main__':\n")
                            new_file.write(f"    sweep_grids = {sweep_grids}\n")
                            new_file.write(f"    baseline_config = {current_best_config}\n")
                            new_file.write(f"    model_size = '{real_model_size}'\n")
                            new_file.write(f"    target_chinchilla = {real_target_chinchilla}\n")
                            new_file.write("    my_suffix = None\n")
                            new_file.write(f"    template(model_size, target_chinchilla, '{optimizer_name_2}', baseline_config, sweep_grids, random_suffix=my_suffix)\n")
                        
                        print(f"Generated new script: {new_file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # rewrite('soap', '130m', 4, '130m', 2)
    # rewrite('soap', '130m', 2, '130m', 4)
    # rewrite('adamw', '130m', 8, '300m', 1)
    # rewrite('adamw', '130m', 8, '520m', 1)
    # rewrite('soape', '130m', 1, '130m', 1)
    # rewrite('soape', '130m', 1, '130m', 2)
    # rewrite('soape', '130m', 4, '130m', 4)
    # rewrite('soape', '130m', 4, '130m', 8)
    # rewrite('muon', '130m', 4, '300m', 1)
    # rewrite('muon', '130m', 8, '520m', 1)
    # rewrite('soape', '130m', 8, '300m', 1)
    # rewrite('soape', '130m', 8, '520m', 1)
    # do this for cautious
    # rewrite('cautious', '130m', 1, '130m', 2)
    # rewrite('cautious', '130m', 4, '130m', 4)
    # rewrite('cautious', '130m', 4, '130m', 8)
    # rewrite('cautious', '300m', 1, '520m', 1)
    # rewrite('nadamw', '130m', 1, '130m', 2)
    # rewrite('kron', '130m', 4, '130m', 4)
    # for data_and_model in [('130m', 1), ('130m', 2), ('130m', 4), ('300m', 1), ('520m', 1)]:
    #     rewrite('adamw', data_and_model[0], data_and_model[1], data_and_model[0], data_and_model[1], 'mini')
    # rewrite('mini', '130m', 4, '130m', 8)
    # rewrite('muon', '130m', 8, '130m', 16)
    rewrite('adamw', '130m', 8, '130m', 16)
    # rewrite('soape', '130m', 8, '130m', 16)
    # rewrite('sophia', '130m', 1, '130m', 2)
    # rewrite("sophia", "130m", 8, '300m', 1)
    # rewrite("sophia", "130m", 8, '520m', 1)
    