from pettingzoo.test import api_test

from rps import env

aec_env = env()
api_test(aec_env, num_cycles=10, verbose_progress=True  )
