import os
import ctypes
import importlib
import json

from openke.config import Trainer, Tester
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


def _is_loaded(libp):
	# libp = os.path.abspath(lib)
	ret = os.system("lsof -w -p %d | grep %s > /dev/null" % (os.getpid(), libp))
	return ret == 0


def _dlclose(handle):
	libdl = ctypes.CDLL("libdl.so")
	libdl.dlclose(handle)



def unload_dll(con):
	# unload dll
	handle = con.lib._handle
	del con.lib
	while _is_loaded(con.base_file):
		_dlclose(handle)


def set_pretrain_model(pretrain_model_path, is_pretrain):
	if is_pretrain:
		try:
			f = open(pretrain_model_path, "r")
			pretrain_model = json.load(f)
			f.close()
			print("Using pretrained weights")
			return pretrain_model
		except FileNotFoundError:
			print("Not using pretrained weights")
			return None
	else:
		print("Not using pretrained weights")
		return None

def train_test(workspace, params, iter_id, gpu_id, is_test=True, is_pretrain=True, is_sans_mining=False):
	# dataloader for training
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	train_dataloader = TrainDataLoader(
		in_path=workspace.base + "/",
		nbatches=params.em_batch,
		threads=8,
		sampling_mode=params.sampling_mode,
		bern_flag=params.em_bern,
		filter_flag=1,
		neg_ent=params.ent_neg_rate,
		neg_rel=0)

	# dataloader for test
	test_dataloader = TestDataLoader(workspace.base + "/", "link")

	# define the model

	models_module = importlib.import_module('openke.module.model')
	em_model_name = getattr(models_module, params.em)

	em_model = em_model_name(
		ent_tot=train_dataloader.get_ent_tot(),
		rel_tot=train_dataloader.get_rel_tot(),
		dim=params.em_dim,
		pretrain_model=set_pretrain_model(workspace.model_fpath, is_pretrain)
		)

	# define the loss function
	# model = NegativeSampling(
	# 	model=em_model,
	# 	loss=MarginLoss(margin=5.0),
	# 	batch_size=train_dataloader.get_batch_size()
	# )
	# define the loss function
	model = NegativeSampling(
		model=em_model,
		loss=SigmoidLoss(adv_temperature=2),
		batch_size=train_dataloader.get_batch_size(),
		regul_rate=0.0
	)

	# train the model
	trainer = Trainer(model=model, data_loader=train_dataloader, train_times=params.em_iters,
	                  alpha=params.em_lr, use_gpu=True, opt_method=params.em_opt)
	trainer.run()
	if not os.path.isdir(workspace.model_dir):
		os.mkdir(workspace.model_dir)
	em_model.save_embedding_matrix(workspace.model_fpath)
	check_path = os.path.join(workspace.checkpoint_dir, params.em + ".json")
	print("Saving Embedding matrix done")
	if not os.path.isdir(workspace.checkpoint_dir):
		os.mkdir(workspace.checkpoint_dir)
	em_model.save_checkpoint(check_path)
	print("Saving checkpoint done")

	# test the model
	em_model.load_checkpoint(check_path)
	tester = Tester(model=em_model, data_loader=test_dataloader, use_gpu=True)
	mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
	os.makedirs(workspace.embedding_log_dir, exist_ok=True)
	with open(workspace.result_dir + "lp_no_typ.txt", "a") as inp:
		inp.write("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n" % (mrr, mr, hit10, hit3, hit1))
		inp.close()
	# acc, thres = tester.run_triple_classification()
	# print("Triplet classification acc : %f \t threshold : %f \n" % (acc, thres))
	#
	#
	# with open(workspace.result_dir + "tc.txt", "a") as inp:
	# 	inp.write("tc_acc:\t %f \n" % (acc))
	# 	inp.close()

	return tester, train_dataloader, test_dataloader


# def train_test(workspace, params, iter_id, gpu_id, is_test=True, is_pretrain=True, is_sans_mining=False):
# 	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
# 	con = config.Config()
# 	con.set_use_gpu(True)
# 	con.set_in_path(workspace.base + "/")
# 	con.set_work_threads(8)
# 	con.set_train_times(params.em_iters)
# 	con.set_nbatches(params.em_batch)
# 	con.set_alpha(params.em_lr)
# 	con.set_bern(params.em_bern)
# 	con.set_dimension(params.em_dim)
# 	con.set_margin(1.0)
# 	con.set_lmbda(params.lmbda)
# 	con.set_ent_neg_rate(params.ent_neg_rate)
# 	con.set_rel_neg_rate(0)
# 	con.set_opt_method(params.em_opt)
# 	con.set_save_steps(2000)
# 	con.set_valid_steps(params.em_vsteps)
# 	con.set_early_stopping_patience(3)
# 	con.set_checkpoint_dir(workspace.checkpoint_dir)
# 	if is_sans_mining:
# 		con.set_result_dir(workspace.sans_mining_model_dir)
# 	else:
# 		con.set_result_dir(workspace.model_dir)
# 	con.set_test_link(True)
# 	con.set_test_triple(True)
# 	con.set_log_instance(workspace.embedding_log_dir + "/iter-" + str(iter_id))
#
# 	if is_pretrain:
# 		con.set_pretrain_model(workspace.model_fpath)
#
# 	con.init()
# 	models_module = importlib.import_module('openke.module.model')
# 	model = getattr(models_module, params.em)
# 	con.set_train_model(model)
# 	con.train()
# 	if is_test:
# 		print("Testing...")
# 		con.set_test_model(model)
# 		con.test()
# 		print("Finish test")
#
# 	# # unload dll
# 	# handle = con.lib._handle
# 	# del con.lib
# 	# while _is_loaded(con.base_file):
# 	# 	_dlclose(handle)
# 	return con




