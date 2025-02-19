import os
from log_states import try_get_luxai_root_path
from tqdm.auto import tqdm

FROM = 'state_logs_0'
TO = 'state_logs'

LUXAI_ROOT_PATH = try_get_luxai_root_path()
for train_test in ('train', 'test'):
    print(f'Merging {train_test} states')
    for state in tqdm(os.listdir(LUXAI_ROOT_PATH / FROM / train_test)):
        os.system(f'mv {LUXAI_ROOT_PATH / FROM / train_test / state} {LUXAI_ROOT_PATH / TO / train_test}/')
