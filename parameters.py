import argparse


class Parameters():
	def __init__(self):
		self.parser = argparse.ArgumentParser(description='Model Trainer')
		self.parser.add_argument('--exp_id', type=str, default="74", help='Experiment ID')
		self.parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
		self.parser.add_argument('--load_existing_test_files', type=int, default=1, help="Load existing test files")
		self.parser.add_argument('--load_existing_test_files_sparsity', type=int, default=0, help="Load existing test files sparsity dataset")

		self.parser.add_argument('--w', type=str, default="../scratch/benchmarks/FB15K237", help='Workspace Directory')
		self.parser.add_argument('--em', type=str, default="RotatE", help='Embedding model name')
		self.parser.add_argument('--g_iters', type=int, default=10, help='Global iterations')
		self.parser.add_argument('--tr_ratio', type=float, default=0.92, help='Train ratio')

		# embedding related
		self.parser.add_argument('--em_iters', type=int, default=100, help='Embedding model iterations')
		self.parser.add_argument('--em_vsteps', type=int, default=105, help='Valid Steps')
		self.parser.add_argument('--em_dim', type=int, default=512, help='Embedding dimension')
		self.parser.add_argument('--em_batch', type=int, default=1000, help='Batch size')
		self.parser.add_argument('--em_lr', type=float, default=2e-5, help='Learning rate')
		self.parser.add_argument('--em_bern', type=int, default=0, help='Bern Sampling')
		self.parser.add_argument('--em_opt', type=str, default="adam", help='Optimizer')
		self.parser.add_argument('--ent_neg_rate', type=int, default=64, help='Neg Entity rate')
		self.parser.add_argument('--lmbda', type=float, default=0.09, help='Regularization lambda')
		self.parser.add_argument('--sampling_mode', type=str, default="cross", help='normal / cross')


		# mining related
		self.parser.add_argument('--m_top_rules', type=int, default=3100, help='Number of top rules to infer')

	def get_parameters(self):
		args = self.parser.parse_args()
		return args