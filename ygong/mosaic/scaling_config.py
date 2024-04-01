class ScalingConfig:
    def __init__(
            self,
            gpusNum: int,
            poolName: str,
            priority: str = 'high',
            preemptible: bool = False, 
            retry_on_system_failure: bool = False):
        self.gpusNum = gpusNum
        self.poolName = poolName
        self.priority = priority
        self.preemptible = preemptible
        self.retry_on_system_failure = retry_on_system_failure

    @property
    def toCompute(self):
        return {
            'gpus': self.gpusNum,
            'gpu_type': 'a100_80gb',
            'cluster': self.poolName
        }
    
    @property
    def toScheduling(self):
        return {
            'priority': self.priority,
            'preemptible': self.preemptible,
            'retry_on_system_failure': self.retry_on_system_failure
        }
