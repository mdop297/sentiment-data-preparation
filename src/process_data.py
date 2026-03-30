import os
from pathlib import Path
from src.config_schemas.data_processing.dataset_cleaners_schema import DatasetCleanerManagerConfig
from src.config_schemas.data_processing_config_schema import DataProcessingConfig
from src.utils.config_utils import custom_instantiate, get_config, get_pickle_config
from src.utils.gcp_utils import access_secret_version
from src.utils.data_utils import get_raw_data_with_version
from dask.distributed import Client
from hydra.utils import instantiate

from src.utils.io_utils import write_yaml_file
from src.utils.utils import get_logger
import dask.dataframe as dd

from google.oauth2 import service_account
from google.auth.transport.requests import Request


def process_raw_data(
    df_partition: dd.core.DataFrame, dataset_cleaner_manager: DatasetCleanerManagerConfig
) -> dd.core.Series:
    processed_partition: dd.core.Series = df_partition["text"].apply(dataset_cleaner_manager)
    return processed_partition


@get_pickle_config(config_path="src/configs/auto_generated", config_name="data_processing_config")
def process_data(config: DataProcessingConfig) -> None:
    logger = get_logger(name=Path(__file__).name)
    logger.info("Processing raw data ...")
    
    processed_data_save_dir = config.processed_data_save_dir
    
    logger.info("instantiating cluster...")
    cluster = custom_instantiate(config.dask_cluster)
    logger.info("instantiated cluster!")

    client = Client(cluster)
    try: 
        # print(20 * "===")
        # from omegaconf import OmegaConf

        # yaml_config = OmegaConf.to_yaml(config)
        # print(yaml_config)
        # print(20 * "===")
        # return

        # github_access_token = access_secret_version(project_id="mdop-cybulde", secret_id="cybulde-github-access-token")

        # get_raw_data_with_version(
        #     version=config.version,
        #     data_local_save_dir=config.data_local_save_dir,
        #     dvc_remote_repo=config.dvc_remote_repo,
        #     dvc_data_folder=config.dvc_data_folder,
        #     github_user_name=config.github_user_name,
        #     github_access_token=github_access_token,
        # )

        dataset_reader_manager = instantiate(config.dataset_reader_manager)
        dataset_cleaner_manager = instantiate(config.dataset_cleaner_manager)
        df = dataset_reader_manager.read_data(config.dask_cluster.n_workers)
        print(df.compute().head())


        logger.info("Cleaning data ...")
        df = df.assign(cleaned_text = df.map_partitions(process_raw_data, dataset_cleaner_manager=dataset_cleaner_manager, meta=("text", "object")))
        df = df.compute()

        train_parquet_path = os.path.join(processed_data_save_dir, "train.parquet")
        dev_parquet_path = os.path.join(processed_data_save_dir, "dev.parquet")
        test_parquet_path = os.path.join(processed_data_save_dir, "test.parquet")

        df[df['split'] == "train"].to_parquet(train_parquet_path)
        df[df['split'] == "dev"].to_parquet(dev_parquet_path)
        df[df['split'] == "test"].to_parquet(test_parquet_path)

        docker_info = {"docker_image": config.docker_image_name, "docker_tag": config.docker_image_tag}
        docker_info_save_path = os.path.join(processed_data_save_dir, 'docker_info.yaml')
        write_yaml_file(docker_info_save_path, docker_info)

        logger.info("Data processing finished!")


    finally:
        logger.info("Closing dask client and cluster...")
        client.close()
        cluster.close()


if __name__ == "__main__":
    service_account_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not service_account_file:
        raise ValueError("The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    
    # Refresh the credentials to ensure they are valid
    credentials.refresh(Request())
    
    print(f"Using service account file: {service_account_file}")
    print(f"{credentials.valid}")

    # Call your data processing function
    process_data()
