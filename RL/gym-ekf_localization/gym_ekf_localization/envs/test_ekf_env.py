import ipdb
from EKFLocReducedEnv import EKFLocReducedEnv

def test_ekf_env():
    env = EKFLocReducedEnv()

    k = 0
    for j in range(10):
        env.reset()
        for i in range(100):
            env.step(1)
            if env.is_done:
                k = k + 1
                env.render()
                break
        env.render()

    ipdb.set_trace()

if __name__=='__main__':
    test_ekf_env()