import pickle
from pathlib import Path


states_dir = Path('/Users/artemveshkin/dev/luxai-s3/state_logs/processed_states')
with open(states_dir / '1_1.pickle', 'rb') as fh:
    logged = pickle.load(fh)

print(logged['state']['space'][''])