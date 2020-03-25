import shutil
import os
import subprocess
import torch
from util.workspace import Workspace
from util.dataset import Dataset
from parameters import Parameters
from util.train_emb import train_test, unload_dll


p = Parameters()
params = p.get_parameters()


workspace = Workspace(params.w, params.em, params.exp_id)

if os.path.exists(workspace.base):
	shutil.rmtree(workspace.base)

os.makedirs(workspace.base)
os.makedirs(workspace.result_dir)


# fresh start generate train data from raw for mining and embedding
dataset = Dataset(workspace, train_ratio=params.tr_ratio, shuffle=True, load_existing_test_files=params.load_existing_test_files, load_existing_test_files_sparsity=params.load_existing_test_files_sparsity)

global_iters = params.g_iters

print(str(params))
params_dump_file = open(workspace.base + "/params.txt", "w")
n = params_dump_file.write(str(params))
params_dump_file.close()


for iter_id in range(global_iters):
	print("Global Iter: ", iter_id)

	# run embedding model
	if iter_id == 0:

		tester, train_dataloader, test_dataloader = train_test(workspace, params, iter_id, params.gpu_id, is_pretrain=False)

		if not os.path.isfile(workspace._base +'/rules.txt.sorted'):
			subprocess.run(["java", "-XX:-UseGCOverheadLimit", "-Xmx100G", "-jar", "mining/build.jar", "-rt", workspace.model_dir,"-w", workspace.base, "-em",params.em, "-ew", "0.3" ])
			shutil.copy(workspace.base + '/rules.txt.sorted', workspace._base +'/rules.txt.sorted', follow_symlinks=True)
	else:
		tester, train_dataloader, test_dataloader = train_test(workspace, params, iter_id, params.gpu_id, is_pretrain=True)

	# embedding saved to result folder
	# run mining system

	# subprocess.run(["java", "-XX:-UseGCOverheadLimit", "-Xmx100G", "-jar", "mining/build.jar", "-rt", workspace.model_dir,"-w", workspace.base, "-em",params.em, "-ew", "0.3", "-pca", "true" ])

	# infer new facts
	mining_log_fpath = workspace.mining_log_dir + "/iter-" + str(iter_id)
	os.makedirs(mining_log_fpath, exist_ok=True)

	subprocess.run(["bash", "infer.sh", workspace.base, workspace._base +'/rules.txt.sorted', str(params.m_top_rules), workspace.mining_new_facts_fpath, mining_log_fpath])

	# read data from mining log

	# new facts in /new_facts.txt
	# read and add to train2id.txt
	#con = init_OpenKE(workspace, params, 0, params.gpu_id, is_pretrain=True, is_sans_mining=True)

	#dataset.augment_inferred_data_random(tester)
	dataset.augment_inferred_data()

	dataset.gen_type_constraints()

	# unload dll

	# unload dll
	unload_dll(train_dataloader)
	unload_dll(test_dataloader)