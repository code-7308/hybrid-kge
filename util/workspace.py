import os


class Workspace:
	def __init__(self, workspace_dir, model_name,exp_id):
		self.dataset_name = os.path.basename(workspace_dir)
		self.model_name = model_name
		self._base = workspace_dir
		self.base = workspace_dir + "/" +str(exp_id)
		self.result_dir = self.base + "/result/"
		self.embedding_test_results_fpath = self.result_dir + "embedding_test_results.txt"
		self.model_fpath = self.result_dir+"model/"+model_name+".json"
		self.embedding_log_dir = self.result_dir + "log/embedding/"
		self.mining_log_dir = self.result_dir + "log/mining/"
		self.checkpoint_dir = self.result_dir + "checkpoint/"
		self.model_dir = self.result_dir + "model"
		self.sans_mining_model_dir = self._base + "/base_models"

		# dataset related fpaths
		self.entities_description_fpath = self.base + "/entities_description.txt"
		self.base_ideal_data_fpath = self._base + "/ideal.data.txt"
		self.ideal_data_fpath = self.base + "/ideal.data.txt"


		self.mining_train_fpath = self.base + "/train.txt"
		self.mining_test_fpath = self.base + "/test.txt"
		self.mining_valid_fpath = self.base + "/valid.txt"
		self.mining_meta_fpath = self.base + "/meta.txt"
		self.mining_e_desc_fpath = self.base + "/e_desc.txt"
		self.mining_new_facts_fpath = self.base + "/new_facts.txt"

		self.embedding_train_fpath = self.base + "/train2id.txt"
		self.embedding_train_fpath_init = self._base + "/train2id_init.txt"
		self.embedding_test_fpath = self.base + "/test2id.txt"
		self.embedding_valid_fpath = self.base + "/valid2id.txt"
		self.embedding_entity2id_fpath = self.base + "/entity2id.txt"
		self.embedding_relation2id_fpath = self.base + "/relation2id.txt"
		self.embedding_type_constrain_fpath = self.base + "/type_constrain.txt"

