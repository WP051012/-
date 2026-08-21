"""Upload conditional FlowChain files to instance A1 (port 44037).

New first-version model: signal + geometry + scene + goal conditioning on top
of the frozen FlowChain backbone. NO meta-learning / ModulationNet / AdaBN.
"""
import paramiko, os

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'
BASE = r'C:\Users\wangj\Desktop\闯红灯预测'

FILES = [
    ('data/dataset.py', '/root/red-light-prediction/data/dataset.py'),
    ('src/context_encoders.py', '/root/red-light-prediction/src/context_encoders.py'),
    ('src/conditional_flowchain.py', '/root/red-light-prediction/src/conditional_flowchain.py'),
    ('src/prediction/flow_chain.py', '/root/red-light-prediction/src/prediction/flow_chain.py'),
    ('src/prediction/flow_chain_official.py', '/root/red-light-prediction/src/prediction/flow_chain_official.py'),
    ('scripts/train_conditional.py', '/root/red-light-prediction/scripts/train_conditional.py'),
    ('scripts/eval_conditional.py', '/root/red-light-prediction/scripts/eval_conditional.py'),
    ('scripts/test_conditional.py', '/root/red-light-prediction/scripts/test_conditional.py'),
    ('run_conditional.sh', '/root/red-light-prediction/run_conditional.sh'),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
print('Connected')

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
        print(f'  Uploaded: {local_rel} -> {remote}')
    else:
        print(f'  MISSING: {local}')
sftp.close()
c.close()
print('Done')
