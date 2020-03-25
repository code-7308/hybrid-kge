import os
import shutil
from sklearn.model_selection import train_test_split
import random
import numpy as np
from scipy.special import softmax
from scipy import stats

class Dataset:
	def __init__(self, workspace, train_ratio=0.8, shuffle=True, load_existing_test_files_sparsity=0,load_existing_test_files=0):
		self.workspace = workspace
		self.entities = {}
		self.entities_arr = []
		self.relations = {}
		self.relations_arr = []
		self.types = {}
		self.types_arr = []
		self._copy_ideal_data()
		self.ideal_data_encoded = self.generate_meta_file()
		if load_existing_test_files_sparsity:
			print("Loading existing sparse files")
			self.load_existing_test_files_sparsity()
		elif load_existing_test_files:
			print("Loading existing files")
			self.load_existing_test_files()
		else:
			self.generate_train_test_files(train_ratio, shuffle)
		self.gen_type_constraints()
		shutil.copy(self.workspace.embedding_train_fpath, self.workspace.embedding_train_fpath_init,
		            follow_symlinks=True)

	def load_existing_test_files(self):
		print("Loading existing valid and test files ...")
		train_encoded = self._read_test_files(self.workspace._base + "/train_existing.txt")
		valid_encoded = self._read_test_files(self.workspace._base + "/valid_existing.txt")
		test_encoded = self._read_test_files(self.workspace._base + "/test_existing.txt")

		self.tr_encoded = train_encoded
		self.val_encoded = valid_encoded
		self.tes_encoded = test_encoded

		print("Train Size :", len(train_encoded))
		print("Valid Size :", len(valid_encoded))
		print("Test Size :", len(test_encoded))
		# mining files
		with open(self.workspace.mining_train_fpath, 'w') as out:
			for v in range(len(train_encoded)):
				out.write("%d\t%d\t%d\n" % train_encoded[v])
			out.close()
		with open(self.workspace.mining_valid_fpath, 'w') as out:
			for v in range(len(valid_encoded)):
				out.write("%d\t%d\t%d\n" % valid_encoded[v])
			out.close()
		with open(self.workspace.mining_test_fpath, 'w') as out:
			for v in range(len(test_encoded)):
				out.write("%d\t%d\t%d\n" % test_encoded[v])

		# Gen description data
		if os.path.isfile(self.workspace.entities_description_fpath):
			print('Processing entities description')
			entity2id = {}
			e = None
			with open(self.workspace.mining_meta_fpath, 'r') as f:
				line = f.readline()
				e = int(line.split("\t")[0])
				for i in range(e):
					entity2id[f.readline().strip()] = i
				f.close()
			data = {}
			with open(self.workspace.entities_description_fpath, 'r') as f:
				for line in f.readlines():
					arr = line.strip().split("\t")
					if arr[0] in entity2id:
						data[entity2id[arr[0]]] = arr[1]
					else:
						print('Invalid entity in description file:', arr[0])
				f.close()
			with open(self.workspace.mining_e_desc_fpath, "w") as f:
				for i in range(e):
					if i in data:
						f.write("%d\t%s\n" % (i, data[i]))
					else:
						f.write("%d\t\n" % i)
				f.close()

		# embedding format train test files
		with open(self.workspace.embedding_entity2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.entities_arr)))
			for e in self.entities_arr:
				out.write("%s\t%d\n" % (e, self.entities[e]))

		with open(self.workspace.embedding_relation2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.relations_arr)))
			for r in self.relations_arr:
				out.write("%s\t%d\n" % (r, self.relations[r]))

		with open(self.workspace.embedding_train_fpath, 'w') as out:
			out.write("%d\n" % (len(train_encoded)))  # number of triples
			for inst in train_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_test_fpath, 'w') as out:
			out.write("%d\n" % (len(test_encoded)))  # number of triples
			for inst in test_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_valid_fpath, 'w') as out:
			out.write("%d\n" % (len(valid_encoded)))  # number of triples
			for inst in valid_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))



		print("Done generating train test files ...")

	def load_existing_test_files_sparsity(self):
		print("Loading existing valid and test files ...")
		train_encoded = self._read_test_files(self.workspace._base + "/train_main.txt")
		valid_encoded = self._read_test_files(self.workspace._base + "/valid_sparsity_0.995.txt")
		test_encoded = self._read_test_files(self.workspace._base + "/test_sparsity_0.995.txt")

		self.tr_encoded = train_encoded
		self.val_encoded = valid_encoded
		self.tes_encoded = test_encoded

		print("Train Size :", len(train_encoded))
		print("Valid Size :", len(valid_encoded))
		print("Test Size :", len(test_encoded))
		# mining files
		with open(self.workspace.mining_train_fpath, 'w') as out:
			for v in range(len(train_encoded)):
				out.write("%d\t%d\t%d\n" % train_encoded[v])
			out.close()
		with open(self.workspace.mining_valid_fpath, 'w') as out:
			for v in range(len(valid_encoded)):
				out.write("%d\t%d\t%d\n" % valid_encoded[v])
			out.close()
		with open(self.workspace.mining_test_fpath, 'w') as out:
			for v in range(len(test_encoded)):
				out.write("%d\t%d\t%d\n" % test_encoded[v])

		# Gen description data
		if os.path.isfile(self.workspace.entities_description_fpath):
			print('Processing entities description')
			entity2id = {}
			e = None
			with open(self.workspace.mining_meta_fpath, 'r') as f:
				line = f.readline()
				e = int(line.split("\t")[0])
				for i in range(e):
					entity2id[f.readline().strip()] = i
				f.close()
			data = {}
			with open(self.workspace.entities_description_fpath, 'r') as f:
				for line in f.readlines():
					arr = line.strip().split("\t")
					if arr[0] in entity2id:
						data[entity2id[arr[0]]] = arr[1]
					else:
						print('Invalid entity in description file:', arr[0])
				f.close()
			with open(self.workspace.mining_e_desc_fpath, "w") as f:
				for i in range(e):
					if i in data:
						f.write("%d\t%s\n" % (i, data[i]))
					else:
						f.write("%d\t\n" % i)
				f.close()

		# embedding format train test files
		with open(self.workspace.embedding_entity2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.entities_arr)))
			for e in self.entities_arr:
				out.write("%s\t%d\n" % (e, self.entities[e]))

		with open(self.workspace.embedding_relation2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.relations_arr)))
			for r in self.relations_arr:
				out.write("%s\t%d\n" % (r, self.relations[r]))

		with open(self.workspace.embedding_train_fpath, 'w') as out:
			out.write("%d\n" % (len(train_encoded)))  # number of triples
			for inst in train_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_test_fpath, 'w') as out:
			out.write("%d\n" % (len(test_encoded)))  # number of triples
			for inst in test_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_valid_fpath, 'w') as out:
			out.write("%d\n" % (len(valid_encoded)))  # number of triples
			for inst in valid_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))



		print("Done generating train test files ...")


	def _read_test_files(self, path):
		with open(path, "r") as fin:
			data = fin.readlines()
			fin.close()

		data_encoded = []
		num_err = 0
		for line in data:
			arr = line.strip().split('\t')
			if len(arr) != 3:
				# print('err:', line)
				num_err += 1
				continue
			s, p, o = arr
			if p == '<subClassOf>':
				continue

			if p == '<type>':
				continue
			data_encoded.append((self.entities[s], self.relations[p], self.entities[o]))

		return data_encoded


	def _copy_ideal_data(self):
		newPath = shutil.copy(self.workspace.base_ideal_data_fpath ,self.workspace.ideal_data_fpath)

	def generate_meta_file(self):
		print("Loading Ideal dataset ...")
		with open(self.workspace.ideal_data_fpath, "r") as fin:
			ideal_data = fin.readlines()
			fin.close()

		data_encoded = []
		types_encoded = []
		num_err = 0
		for line in ideal_data:
			arr = line.strip().split('\t')
			if len(arr) != 3:
				# print('err:', line)
				num_err += 1
				continue
			s, p, o = arr
			if p == '<subClassOf>':
				continue
			if s not in self.entities:
				self.entities[s] = len(self.entities_arr)
				self.entities_arr.append(s)
			if p == '<type>':
				if o not in self.types:
					self.types[o] = len(self.types_arr)
					self.types_arr.append(o)
				types_encoded.append((self.entities[s], self.types[o]))
				continue
			if o not in self.entities:
				self.entities[o] = len(self.entities_arr)
				self.entities_arr.append(o)
			if p not in self.relations:
				self.relations[p] = len(self.relations_arr)
				self.relations_arr.append(p)

			data_encoded.append((self.entities[s], self.relations[p], self.entities[o]))

		# generate meta data
		with open(self.workspace.mining_meta_fpath, 'w') as out:
			out.write("%d\t%d\t%d\n" % (len(self.entities_arr), len(self.relations_arr), len(self.types_arr)))
			for v in self.entities_arr:
				out.write("%s\n" % v)
			for v in self.relations_arr:
				out.write("%s\n" % v)
			for v in self.types_arr:
				out.write("%s\n" % v)
			for v in range(len(types_encoded)):
				out.write("%d\t%d\n" % types_encoded[v])
		print("Done generating meta data file ...")
		return data_encoded

	def generate_train_test_files(self,  train_ratio, shuffle):
		_train, test_encoded = train_test_split(self.ideal_data_encoded, train_size=train_ratio, shuffle=shuffle, random_state=0)
		train_encoded, valid_encoded = train_test_split(_train, train_size=train_ratio, shuffle=shuffle, random_state=0)
		self.tr_encoded = train_encoded
		self.val_encoded = valid_encoded
		self.tes_encoded = test_encoded

		print("Train Size :", len(train_encoded))
		print("Valid Size :", len(valid_encoded))
		print("Test Size :", len(test_encoded))
		# mining files
		with open(self.workspace.mining_train_fpath, 'w') as out:
			for v in range(len(train_encoded)):
				out.write("%d\t%d\t%d\n" % train_encoded[v])
			out.close()
		with open(self.workspace.mining_valid_fpath, 'w') as out:
			for v in range(len(valid_encoded)):
				out.write("%d\t%d\t%d\n" % valid_encoded[v])
			out.close()
		with open(self.workspace.mining_test_fpath, 'w') as out:
			for v in range(len(test_encoded)):
				out.write("%d\t%d\t%d\n" % test_encoded[v])

		# Gen description data
		if os.path.isfile(self.workspace.entities_description_fpath):
			print('Processing entities description')
			entity2id = {}
			e = None
			with open(self.workspace.mining_meta_fpath, 'r') as f:
				line = f.readline()
				e = int(line.split("\t")[0])
				for i in range(e):
					entity2id[f.readline().strip()] = i
				f.close()
			data = {}
			with open(self.workspace.entities_description_fpath, 'r') as f:
				for line in f.readlines():
					arr = line.strip().split("\t")
					if arr[0] in entity2id:
						data[entity2id[arr[0]]] = arr[1]
					else:
						print('Invalid entity in description file:', arr[0])
				f.close()
			with open(self.workspace.mining_e_desc_fpath, "w") as f:
				for i in range(e):
					if i in data:
						f.write("%d\t%s\n" % (i, data[i]))
					else:
						f.write("%d\t\n" % i)
				f.close()

		# embedding format train test files
		with open(self.workspace.embedding_entity2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.entities_arr)))
			for e in self.entities_arr:
				out.write("%s\t%d\n" % (e, self.entities[e]))

		with open(self.workspace.embedding_relation2id_fpath, 'w') as out:
			out.write("%d\n" % (len(self.relations_arr)))
			for r in self.relations_arr:
				out.write("%s\t%d\n" % (r, self.relations[r]))

		with open(self.workspace.embedding_train_fpath, 'w') as out:
			out.write("%d\n" % (len(train_encoded)))  # number of triples
			for inst in train_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_test_fpath, 'w') as out:
			out.write("%d\n" % (len(test_encoded)))  # number of triples
			for inst in test_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		with open(self.workspace.embedding_valid_fpath, 'w') as out:
			out.write("%d\n" % (len(valid_encoded)))  # number of triples
			for inst in valid_encoded:
				e1, r, e2 = inst  # all ids
				out.write("%d\t%d\t%d\n" % (e1, e2, r))

		print("Done generating train test files ...")

	def gen_type_constraints(self):
		lef = {}
		rig = {}
		rellef = {}
		relrig = {}

		triple = open(self.workspace.embedding_train_fpath, "r")
		valid = open(self.workspace.embedding_valid_fpath, "r")
		test = open(self.workspace.embedding_test_fpath, "r")

		tot = (int)(triple.readline())
		for i in range(tot):
			content = triple.readline()
			h, t, r = content.strip().split()
			if not (h, r) in lef:
				lef[(h, r)] = []
			if not (r, t) in rig:
				rig[(r, t)] = []
			lef[(h, r)].append(t)
			rig[(r, t)].append(h)
			if not r in rellef:
				rellef[r] = {}
			if not r in relrig:
				relrig[r] = {}
			rellef[r][h] = 1
			relrig[r][t] = 1

		tot = (int)(valid.readline())
		for i in range(tot):
			content = valid.readline()
			h, t, r = content.strip().split()
			if not (h, r) in lef:
				lef[(h, r)] = []
			if not (r, t) in rig:
				rig[(r, t)] = []
			lef[(h, r)].append(t)
			rig[(r, t)].append(h)
			if not r in rellef:
				rellef[r] = {}
			if not r in relrig:
				relrig[r] = {}
			rellef[r][h] = 1
			relrig[r][t] = 1

		tot = (int)(test.readline())
		for i in range(tot):
			content = test.readline()
			h, t, r = content.strip().split()
			if not (h, r) in lef:
				lef[(h, r)] = []
			if not (r, t) in rig:
				rig[(r, t)] = []
			lef[(h, r)].append(t)
			rig[(r, t)].append(h)
			if not r in rellef:
				rellef[r] = {}
			if not r in relrig:
				relrig[r] = {}
			rellef[r][h] = 1
			relrig[r][t] = 1

		test.close()
		valid.close()
		triple.close()

		f = open(self.workspace.embedding_type_constrain_fpath, "w")
		f.write("%d\n" % (len(rellef)))
		for i in rellef:
			f.write("%s\t%d" % (i, len(rellef[i])))
			for j in rellef[i]:
				f.write("\t%s" % (j))
			f.write("\n")
			f.write("%s\t%d" % (i, len(relrig[i])))
			for j in relrig[i]:
				f.write("\t%s" % (j))
			f.write("\n")
		f.close()

		rellef = {}
		totlef = {}
		relrig = {}
		totrig = {}
		# lef: (h, r)
		# rig: (r, t)
		for i in lef:
			if not i[1] in rellef:
				rellef[i[1]] = 0
				totlef[i[1]] = 0
			rellef[i[1]] += len(lef[i])
			totlef[i[1]] += 1.0

		for i in rig:
			if not i[0] in relrig:
				relrig[i[0]] = 0
				totrig[i[0]] = 0
			relrig[i[0]] += len(rig[i])
			totrig[i[0]] += 1.0

		s11 = 0
		s1n = 0
		sn1 = 0
		snn = 0
		f = open(self.workspace.embedding_test_fpath, "r")
		tot = (int)(f.readline())
		for i in range(tot):
			content = f.readline()
			h, t, r = content.strip().split()
			rign = rellef[r] / totlef[r]
			lefn = relrig[r] / totrig[r]
			if (rign < 1.5 and lefn < 1.5):
				s11 += 1
			if (rign >= 1.5 and lefn < 1.5):
				s1n += 1
			if (rign < 1.5 and lefn >= 1.5):
				sn1 += 1
			if (rign >= 1.5 and lefn >= 1.5):
				snn += 1
		f.close()

		f = open(self.workspace.embedding_test_fpath, "r")
		f11 = open(self.workspace.base + "/1-1.txt", "w")
		f1n = open(self.workspace.base + "/1-n.txt", "w")
		fn1 = open(self.workspace.base + "/n-1.txt", "w")
		fnn = open(self.workspace.base + "/n-n.txt", "w")
		fall = open(self.workspace.base + "/test2id_all.txt", "w")
		tot = (int)(f.readline())
		fall.write("%d\n" % (tot))
		f11.write("%d\n" % (s11))
		f1n.write("%d\n" % (s1n))
		fn1.write("%d\n" % (sn1))
		fnn.write("%d\n" % (snn))
		for i in range(tot):
			content = f.readline()
			h, t, r = content.strip().split()
			rign = rellef[r] / totlef[r]
			lefn = relrig[r] / totrig[r]
			if (rign < 1.5 and lefn < 1.5):
				f11.write(content)
				fall.write("0" + "\t" + content)
			if (rign >= 1.5 and lefn < 1.5):
				f1n.write(content)
				fall.write("1" + "\t" + content)
			if (rign < 1.5 and lefn >= 1.5):
				fn1.write(content)
				fall.write("2" + "\t" + content)
			if (rign >= 1.5 and lefn >= 1.5):
				fnn.write(content)
				fall.write("3" + "\t" + content)
		fall.close()
		f.close()
		f11.close()
		f1n.close()
		fn1.close()
		fnn.close()
		print("Done generating type_constrain files ...")

	def augment_inferred_data(self):
		new_facts_encoded = set()
		with open(self.workspace.mining_new_facts_fpath, 'r') as inp:
			data = inp.readlines()
			for line in data:
				arr = line.strip().split('\t')
				if arr[3] == "null":
					continue
				new_facts_encoded.add((self.entities[arr[0]], self.entities[arr[2]], self.relations[arr[1]]))  # e1, e2, r ids
			inp.close()

		if (self.workspace.dataset_name == "FB15K"):
			v = []
			for inst in self.val_encoded:
				e1, r, e2 = inst  # all ids
				v.append((e1, e2, r))

			new_facts_encoded = new_facts_encoded.union(set(v))
		# read current train2id file
		curr_train_encoded = set()
		with open(self.workspace.embedding_train_fpath, 'r') as inp:
			curr_size = int(inp.readline().strip())
			for i in range(curr_size):
				e1, e2, r = [int(x) for x in inp.readline().strip().split("\t")]
				curr_train_encoded.add((e1, e2, r))
			inp.close()

		new_train_encoded = curr_train_encoded.union(new_facts_encoded)

		with open(self.workspace.embedding_train_fpath, 'w') as out:
			out.write(str(len(new_train_encoded))+"\n")
			for inst in new_train_encoded:
				out.write("%d\t%d\t%d\n" % (inst[0], inst[1], inst[2]))
			out.close()

		print("Augmented %d new inferred triples" % (len(new_train_encoded)-len(curr_train_encoded)))



	def softmax(self, x, sigma=1.0):
		return np.exp(x/sigma) / sum( np.exp(x/sigma))


	def self_guided_sampling(self,con, infer_facts, rid, thresh):
		batch_h = []
		batch_t = []
		batch_r = []
		for f in infer_facts:
			h = f[0]
			t = f[1]
			r = f[2]
			batch_h.append(h)
			batch_t.append(t)
			batch_r.append(r)
			if not r == rid:
				print("error")

		res = con.predict_triple_batch(np.array(batch_h).reshape(1,-1),
		                         np.array(batch_t).reshape(1,-1),
		                         np.array(batch_r).reshape(1,-1))

		#print(thresh)
		out = res.squeeze()
		#print(stats.describe(out))
		msk = (out < thresh).tolist() ## these are predicted to be true.
		if msk is list():
			return [infer_facts[i] for i in range(len(infer_facts)) if (msk[i])]
		else:
			return [infer_facts[i] for i in range(len(infer_facts)) if msk]

	def get_r_to_so(self,facts):
		r_to_so = {}

		for t in facts:
			sid = t[0]
			oid = t[1]
			rid = t[2]
			if not rid in r_to_so:
				r_to_so[rid] = []
			r_to_so[rid].append((sid, oid, rid))
		return r_to_so


	def augment_inferred_data_random(self,tester):

		# read current train2id file
		curr_train_encoded = set()
		with open(self.workspace.embedding_train_fpath_init, 'r') as inp:
			curr_size = int(inp.readline().strip())
			for i in range(curr_size):
				e1, e2, r = [int(x) for x in inp.readline().strip().split("\t")]
				curr_train_encoded.add((e1, e2, r))
			inp.close()

		print("Initial Train Set Size |G_0| : ", str(len(curr_train_encoded)))

		new_facts_encoded = set()
		with open(self.workspace.mining_new_facts_fpath, 'r') as inp:
			data = inp.readlines()
			for line in data:
				arr = line.strip().split('\t')
				# if arr[3] == "null":
				# 	continue
				new_facts_encoded.add((self.entities[arr[0]], self.entities[arr[2]], self.relations[arr[1]]))  # e1, e2, r ids
			inp.close()

		print("Inferred set size |U S_T| : ", str(len(new_facts_encoded)))

		new_facts = new_facts_encoded.difference(curr_train_encoded)

		print("Newly Inferred facts |G_T = U S_T \ G_0| : ", str(len(new_facts)))


		r_to_so = self.get_r_to_so(new_facts)
		# threshs = openke_con.get_thresh()
		res, threshs = tester.run_triple_classification()
		sampled_facts = []
		for rel_key, sos in r_to_so.items():
			sampled_facts += self.self_guided_sampling(tester, sos, rel_key, threshs)


		#sampled_new_facts = set(random.sample(new_facts_encoded, int(len(new_facts_encoded)*0.5)))
		sampled_new_facts = set(sampled_facts)

		print("Sampled new facts size |xi(G_T)| : ", str(len(sampled_new_facts)))

		new_train_encoded = curr_train_encoded.union(sampled_new_facts)

		with open(self.workspace.embedding_train_fpath, 'w') as out:
			out.write(str(len(new_train_encoded))+"\n")
			for inst in new_train_encoded:
				out.write("%d\t%d\t%d\n" % (inst[0], inst[1], inst[2]))
			out.close()

		print("Augmented %d new inferred triples" % (len(new_train_encoded)-len(curr_train_encoded)))
