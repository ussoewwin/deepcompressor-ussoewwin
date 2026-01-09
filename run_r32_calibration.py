
import os
import sys

# Ensure current directory is in python path
sys.path.append(os.getcwd())

from deepcompressor.app.diffusion.dataset.collect.calib import collect, CollectConfig
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.data import get_dataset

def main():
    print("Starting Flux R32 Calibration with Local Fixes...")
    
    # Use the parser to load configs exactly as the CLI does
    # This ensures all defaults and overrides are applied correctly
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    
    # Arguments mimicking the command line
    args = [
        "--config", "examples/diffusion/configs/model/flux.1-dev-r32.yaml",
        "--config", "examples/diffusion/configs/collect/qdiff-r32.yaml"
    ]
    
    configs, _, _, _, _ = parser.parse_known_args(args)
    ptq_config = configs[""]
    collect_config = configs["collect"]
    
    # Calculate output path
    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Target Cache Directory: {collect_dirpath}")
    
    # Load dataset
    dataset = get_dataset(
        collect_config.data_path,
        max_dataset_size=collect_config.num_samples,
        return_gt=ptq_config.pipeline.task in ["canny-to-image"],
        repeat=1,
    )
    
    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    
    # Run collection/calibration
    collect(ptq_config, dataset=dataset)
    print("Calibration Completed Successfully.")

if __name__ == "__main__":
    main()
