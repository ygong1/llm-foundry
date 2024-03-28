from typing import List, Optional
from ygong.mosaic.scaling_config import ScalingConfig
from .wsfs import WSFSIntegration
from mcli import RunConfig
import os


class TrainingConfig:
    def __init__(
            self,
            name: str,
            commands: List[str],
            image: Optional[str] = None):
        self.name = name
        self.mlflow_experimentName = f"/Users/yu.gong@databricks.com/{name}"
        self.commands = commands
        self.image = image if image is not None else 'mosaicml/llm-foundry:2.2.1_cu121_flash2-latest'
        self.hacky_integrations = [
            {
                'integration_type': 'git_repo',
                'git_repo': 'ygong1/llm-foundry',
                'path': '~/llm-foundry',
                'git_branch': 'prototype',
                'pip_install': '-e .[gpu]',
                'ssh_clone': False
            },
            {
                'integration_type': 'pip_packages',
                'packages': ['pynvml', 'mosaicml-streaming[databricks]'],
            },
    ]
        
    def toRunConfig(self, scalingConfig: ScalingConfig, wsfs: Optional[WSFSIntegration] = None) -> RunConfig:
        if wsfs is not None:
            commands = wsfs.get_setup_command() + self.commands
            env_variables = {
                'DATABRICKS_HOST': os.environ['WORKSPACE_URL'],
                'DATABRICKS_TOKEN': os.environ['MLFLOW_TRACKING_TOKEN']
            }
        else:
            commands = self.commands
            env_variables = {}
                
        return RunConfig(
            name=self.name,
            image=self.image,
            command="\n".join(["cat /mnt/config/usercommand.bash"] + commands),
            compute=scalingConfig.toCompute,
            integrations=self.hacky_integrations,
            env_variables=env_variables,
        )
