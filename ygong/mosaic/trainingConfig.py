from typing import List, Optional
from ygong.mosaic.scaling_config import ScalingConfig
from .wsfs import WSFSIntegration
from mcli import RunConfig
import os


class TrainingConfig:
    def __init__(
            self,
            name: str,
            entry_point: Optional[str] = None,
            parameters: Optional[dict] = None,
            commands: Optional[list]= None,
            image: Optional[str] = None,
            priority: str = 'high',
            preemptible: bool = False, 
            retry_on_system_failure: bool = False):
        self.name = name
        self.mlflow_experimentName = f"/Users/yu.gong@databricks.com/{name}"
        self.debug_commands = ["cat /mnt/config/usercommand.bash", 'echo "\n===================\n"',"cat /mnt/config/parameters.yaml"]
        
        self.parameters = parameters if parameters is not None else {}
        if entry_point is not None and commands is None:
            self.commands = [f"~/llm-foundry/scripts/train/launcher.py {entry_point} /mnt/config/parameters.yaml"]
        elif entry_point is None and commands is not None:
            self.commands = commands
        else:
            raise ValueError("Either entry_point or commands must be provided and they cannot be provided at the same time.")
        
        self.image = image if image is not None else 'mosaicml/llm-foundry:2.2.1_cu121_flash2-latest'
        self.priority = priority
        self.preemptible = preemptible
        self.retry_on_system_failure = retry_on_system_failure
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
            command="\n".join(self.debug_commands + commands),
            compute=scalingConfig.toCompute,
            integrations=self.hacky_integrations,
            env_variables=env_variables,
            scheduling={
                'priority': self.priority,
                'preemptible': self.preemptible,
                'retry_on_system_failure': self.retry_on_system_failure
            },
            parameters=self.parameters
        )
