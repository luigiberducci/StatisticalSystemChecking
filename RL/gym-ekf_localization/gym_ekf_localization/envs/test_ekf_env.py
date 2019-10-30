import ipdb
from EKFLocEnv import EKFLocEnv

def test_ekf_env():
    env = EKFLocEnv()
    env.render()

    env.step(50)
    env.render()

if __name__=='__main__':
    test_ekf_env()