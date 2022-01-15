from azureml.core import Workspace, Datastore
from azureml.train.dnn import PyTorch
from azureml.core import Experiment, Environment, ScriptRunConfig, Dataset
from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration
from azureml.train.estimator import Estimator
from sys import argv
script, i, seed, dataset, algorithm = argv
#setup cluster
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

from azureml.core.compute import ComputeTarget
from azureml.contrib.core.compute.k8scompute import AksCompute
for key, target in ws.compute_targets.items():
    if type(target) is AksCompute:
        print('Found compute target:{}\ttype:{}\tprovisioning_state:{}\tlocation:{}'.format(target.name, target.type, target.provisioning_state, target.location))

compute_target = ComputeTarget(workspace=ws, name="itpscusv100cl")# researchvc-eus
experiment_name = '%s_%s_d%s_seed%s'%(dataset, algorithm, str(i), str(seed))
#experiment_name = '%s_%s_e2e_seed%s'%(dataset, str(i), str(seed))
print(experiment_name)
Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name='yifan_data',  # just a name to refer the Datastore
    account_name="yifanzhang",
    container_name="data",
    account_key="pv0wggRvdq2Xf1hMmXWqlz0xm0hmaugghPFfqrD5G2J8BQJ7If6/9G2RAMjjv7o/21RZATGVvUfKiQ9g+Yvduw==")
ds = Datastore(ws, "yifan_data")
Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name='yifan_model',  # just a name to refer the Datastore
    account_name="yifanzhang",
    container_name="model",
    account_key="pv0wggRvdq2Xf1hMmXWqlz0xm0hmaugghPFfqrD5G2J8BQJ7If6/9G2RAMjjv7o/21RZATGVvUfKiQ9g+Yvduw==")
ds_model = Datastore(ws, "yifan_model")
experiment = Experiment(ws, name=experiment_name)

config = Estimator(
    compute_target=compute_target,
    use_gpu=False,
    custom_docker_image="mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04", 
    source_directory='./',
    entry_script='train_submit.py', # './DGdata/'
    script_params={ '--data_dir': ds.path('./DGdata') , '--gen_dir':ds_model.path('./DG-Net/mnist_gen.pkl'), '--dataset':dataset, '--test_envs':i, '--stage':1,'--algorithm': algorithm, '--seed':seed},
    pip_packages=['torch','torchvision', 'pyyaml', 'tqdm', 'wilds','imageio']
)

# set up pytorch environment
# env = Environment.from_conda_specification(name='disdg_yifan1',file_path='/home/v-yifanzhang/DisDG/submit/environment.yml')
# config.run_config.environment = env

compute_config = K8sComputeConfiguration()
compute_config.configuration = {
    'enable_ipython': False, 
    'enable_tensorboard': False, 
    'enable_ssh': True,
    'gpu_count':1,
    'preemption_allowed':False,
    # 'ssh_public_key' : 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDesrvA7BerK+5Ko3yZ6Yg1kKlPGzYFT+0j5PXrrralgTq5EFDLf6UzmgwzyRq6zXYGu4IXMrJOopcacKwU68stzW78u/8jzlxzf5cKpqvZOjFyiUCUyn1rWEGuvUV0GBGmuZEYuIgEXyhBpVGV3K6nRkDbEjBISMeIPN+NXeHIUcodVTJcdly9+kFGSPtBUGTf6D/jCOL1AqS3ti0+fss9Q2n4Y6W5QpI2X+7qbIuIkg82/gbPehIN6ua54ojhejY6d5GNE+5eAw6aIV6/KejNfjWMGiAeepa7t0znQ8v6Dow+i1YdNFogq/wfe5yiGry3b4Gnwx04RX9cpHRmO9q3 cbzhang@server'
    'ssh_public_key' : 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICIlTNE5k4t6TqtiLv17bmepthbZmldge88YpKmxfvvL yifanzhang@BPQ4BV2',
}
config.run_config.cmk8scompute = compute_config

#setup jobs
run = experiment.submit(config)
# run.wait_for_completion(show_output=True)
run.get_tags()