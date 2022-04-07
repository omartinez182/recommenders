import sys
import os
import logging
import papermill as pm
import scrapbook as sb
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')
# Recommenders
sys.path.insert(1, '')
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (prepare_hparams)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing
from recommenders.datasets.download_utils import maybe_download
# Models
from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel1
from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel2
from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel3
from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel4
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator

# Config files
yaml_file1 = 'recommenders/models/deeprec/config/sli_rec.yaml'  
yaml_file2 = 'recommenders/models/deeprec/config/asvd.yaml'  
yaml_file3 = 'recommenders/models/deeprec/config/gru4rec.yaml'  
yaml_file4 = 'recommenders/models/deeprec/config/sum.yaml'  

# Parameters
EPOCHS = 10
BATCH_SIZE = 400
RANDOM_SEED = SEED  # Set None for non-deterministic result

data_path = os.path.join("..", "..", "tests", "resources", "deeprec", "slirec")

# Data
train_file = os.path.join(data_path, r'train_data')
valid_file = os.path.join(data_path, r'valid_data')
test_file = os.path.join(data_path, r'test_data')
user_vocab = os.path.join(data_path, r'user_vocab.pkl')
item_vocab = os.path.join(data_path, r'item_vocab.pkl')
cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
output_file = os.path.join(data_path, r'output.txt')

reviews_name = 'reviews_Amazon_Fashion.json'
meta_name = 'meta_Amazon_Fashion.json'
reviews_file = os.path.join(data_path, reviews_name)
meta_file = os.path.join(data_path, meta_name)
train_num_ngs = 4 # number of negative instances with a positive instance for training
valid_num_ngs = 4 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing
sample_rate = 0.01 # sample a small item set for training and testing here for fast example

input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]

if not os.path.exists(train_file):
    download_and_extract(reviews_name, reviews_file)
    download_and_extract(meta_name, meta_file)
    data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs)

# Hyperparameters
hparams1 = prepare_hparams(yaml_file11, 
                          embed_l2=0., 
                          layer_l2=0., 
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=20,
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs # provides the number of negative instances for each positive instance for loss computation.
            )


hparams2 = prepare_hparams(yaml_file2, 
                          embed_l2=0., 
                          layer_l2=0., 
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=20,
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs # provides the number of negative instances for each positive instance for loss computation.
            )

hparams3 = prepare_hparams(yaml_file3, 
                          embed_l2=0., 
                          layer_l2=0., 
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=20,
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs # provides the number of negative instances for each positive instance for loss computation.
            )

hparams4 = prepare_hparams(yaml_file4, 
                          embed_l2=0., 
                          layer_l2=0., 
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=20,
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs # provides the number of negative instances for each positive instance for loss computation.
            )

# Data loaders
input_creator1 = SequentialIterator
input_creator2 = SequentialIterator
input_creator3 = SequentialIterator
input_creator4 = SequentialIterator

# Instantiate models
model1 = SeqModel1(hparams1, input_creator1, seed=RANDOM_SEED)
model2 = SeqModel2(hparams2, input_creator2, seed=RANDOM_SEED)
model3 = SeqModel3(hparams3, input_creator3, seed=RANDOM_SEED)
model4 = SeqModel4(hparams4, input_creator4, seed=RANDOM_SEED)
model_list = [model1, model2, model3, model4]
# Get model names
model_names = [str(x).replace("<recommenders.models.deeprec.models.sequential.","").
               split()[0].split(".")[1].replace("Model","") for x in model_list]

# Training
times = []
for model in model_list:
  with Timer() as train_time:
      model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs)
  total_t = train_time.end - train_time.init 
  times.append(total_t)

# Evaluation
trained_results = []
for model in model_list:
  res_syn = model.run_eval(test_file, num_ngs=test_num_ngs)
  trained_results.append(res_syn)

trained_res_df = pd.DataFrame(trained_results)
trained_res_df['times_(s)'] = times
trained_res_df['model'] = model_names
cols = trained_res_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
trained_res_df = trained_res_df[cols]
trained_res_df.drop(['group_auc', 'mean_mrr'], axis=1, inplace=True)
trained_res_df.to_csv('Experiment_Results.csv')