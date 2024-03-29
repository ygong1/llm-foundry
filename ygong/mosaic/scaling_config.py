class ScalingConfig:
    def __init__(self, gpusNum: int, poolName: str):
        # TODO: validate the inputs
        self.gpusNum = gpusNum
        self.poolName = poolName

    @property
    def toCompute(self):
        return {
            'gpus': self.gpusNum,
            'gpu_type': 'a100_80gb',
            'cluster': self.poolName
        }
