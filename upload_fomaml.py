"""Upload FOMAML v2 files to A2 (A1 was shut down; A2 is the clone)."""
import paramiko, os

HOST = 'connect.cqa1.seetacloud.com'
PORT = 29463
USER = 'root'
PW = 'UglX1RnukHl0'
BASE = r'C:\Users\wangj\Desktop\闯红灯预测'

FILES = [
    ('src/prediction/flow_chain_official.py', '/root/red-light-prediction/src/prediction/flow_chain_official.py'),
    ('src/prediction/flow_chain.py', '/root/red-light-prediction/src/prediction/flow_chain.py'),
    ('src/perception_model.py', '/root/red-light-prediction/src/perception_model.py'),
    ('src/modulation_net.py', '/root/red-light-prediction/src/modulation_net.py'),
    ('data/dataset.py', '/root/red-light-prediction/data/dataset.py'),
    ('train_fomaml.py', '/root/red-light-prediction/train_fomaml.py'),
    ('scripts/eval_fomaml.py', '/root/red-light-prediction/scripts/eval_fomaml.py'),
    ('scripts/eval_flowchain_domain.py', '/root/red-light-prediction/scripts/eval_flowchain_domain.py'),
    ('scripts/smoke_fomaml.py', '/root/red-light-prediction/scripts/smoke_fomaml.py'),
    ('scripts/test_eval_runtime.py', '/root/red-light-prediction/scripts/test_eval_runtime.py'),
    ('run_smoke.sh', '/root/red-light-prediction/run_smoke.sh'),
    ('run_smoke_eval.sh', '/root/red-light-prediction/run_smoke_eval.sh'),
    ('run_eval_full.sh', '/root/red-light-prediction/run_eval_full.sh'),
    ('run_test_runtime.sh', '/root/red-light-prediction/run_test_runtime.sh'),
    ('scripts/inspect_fomaml_ckpt.py', '/root/red-light-prediction/scripts/inspect_fomaml_ckpt.py'),
    ('run_v4_train_eval.sh', '/root/red-light-prediction/run_v4_train_eval.sh'),
    ('run_smoke_v4.sh', '/root/red-light-prediction/run_smoke_v4.sh'),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
print('Connected')

# Ensure remote parent dirs exist BEFORE uploading
sftp = c.open_sftp()
remote_dirs = sorted({os.path.dirname(r) for _, r in FILES})
for d in remote_dirs:
    try:
        sftp.mkdir(d)
    except IOError:
        pass  # already exists

for local_rel, remote in FILES:
    local = os.path.join(BASE, local_rel)
    if os.path.exists(local):
        sftp.put(local, remote)
        print(f'  Uploaded: {local_rel} → {remote}')
    else:
        print(f'  MISSING: {local}')
sftp.close()
c.close()
print('Done')
