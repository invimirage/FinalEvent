#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: main.py
@time: 2021/5/12 14:45
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""

import config
import json
import logging
from utils import *
from TextAnalysis.text_scorer import main as text_scorer
from VideoAnalysis.video_scorer import main as video_scorer

# 日常要做的实验， 5月12日：测试加入extra data， 测试使用CLS代替Mean Pooling， 测试开放bert参数
test_mode = False
# without cls
with open(config.parameter_file) as f:
    params = json.load(f)
test_params = {
    "hidden_length": [128, 256, 512],
    "layer_number": [1, 2, 3],
    "linear_hidden_length": [64, 128],
    "drop_out_rate": [0.5, 0.4],
    "batch_size": [400, 600, 800],
    "learning_rate": 1e-4,
    "training_size": 0.8,
    "number_epochs": 100,
}
video_params = {
    "frame_rate": 1,
    "input_length": 512,
    "pic_linear_hidden_length": 1024,
    "pic_grad_layer_name": "aaa_blocks.31",
    "pic_drop_out_rate": [0.5, 0.3, 0.6],
    "pic_channels": 3,
    "layer_number": 3,
    "hidden_length": 512,
    "linear_hidden_length": 128,
    "drop_out_rate": 0.5,
    "batch_size": 400,
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "training_size": 0.8,
    "number_epochs": 100,
    "random": 1,
}
# logger = init_logger(
#     log_level=logging.INFO,
#     name="Separated_LSTM_with_upload_time",
#     write_to_file=False,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=1,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=True,
#     use_cls=False,
#     test_mode=test_mode
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=20,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     test_mode=test_mode
# )

# with cls
test_params = {
    "hidden_length": [128, 256, 512],
    "layer_number": [1, 2, 3],
    "linear_hidden_length": [64, 128],
    "drop_out_rate": [0.5, 0.4],
    "batch_size": [400, 600, 800],
    "learning_rate": 1e-4,
    "training_size": 0.8,
    "number_epochs": 100,
}
# logger = init_logger(
#     log_level=logging.INFO,
#     name="Attention_LSTM_with_cls_extra",
#     write_to_file=True,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=1,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=True,
#     use_cls=True,
#     test_mode=test_mode
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=20,
#     model_name="BiLSTMWithAttention",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=True,
#     test_mode=test_mode,
#     tag_col="tag",
#     embed_file=os.path.join(config.data_folder, "vector_embed_Separated_LSTM_with_upload_time.npy")
# )

# with more texts
# test_params = {
#     "hidden_length": [128, 256, 512],
#     "layer_number": [1, 2, 3],
#     "linear_hidden_length": [64, 128],
#     "drop_out_rate": [0.5, 0.4],
#     "batch_size": [400, 600, 800],
#     "learning_rate": 1e-4,
#     "training_size": 0.8,
#     "number_epochs": 100,
# }


# logger = init_logger(
#     log_level=logging.INFO,
#     name="Separated_LSTM_with_extra_and_neighbors_no_center",
#     write_to_file=True,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=1,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=True,
#     use_cls=False,
#     neighbor=1,
#     only_center=False,
#     max_len=100,
#     test_mode=test_mode
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=20,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     neighbor=1,
#     only_center=True,
#     max_len=100,
#     test_mode=test_mode
# )

# Force 覆盖之前的extraction
# logger = init_logger(
#         logging.INFO, name="VideoScorerWithEmbedding", write_to_file=False, clean_up=True
#     )
# adjust_hyperparams(
#     logger, params, 10, "VideoNetEmbed", video_scorer, extract_frames=True, force=False
# )

# cnn_params = {
#                 "hidden_length": [32, 16, 8],
#                 "linear_hidden_length": [32, 64, 128],
#                 "grad_layer_name": ["none"],
#                 "drop_out_rate": 0.5,
#                 "channels": [64, 128],
#                 "batch_size": [20],
#                 "learning_rate": 1e-4,
#                 "training_size": 0.8,
#                 "number_epochs": 100,
#             }
# logger = init_logger(
#     log_level=logging.INFO,
#     name="BertCNN_more_layers",
#     write_to_file=True,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=20,
#     model_name="BertWithCNN",
#     run_model=text_scorer,
#     test_params=cnn_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     test_mode=test_mode
# )

logger = init_logger(
    log_level=logging.INFO,
    name="Separated_LSTM_with_upload_time&Video_Embed_with_cost_no_mean",
    write_to_file=True,
    clean_up=True,
)
adjust_hyperparams(
    logger,
    params,
    sample_number=20,
    model_name="JointNet",
    run_model=video_scorer,
    embed_type="local",
    use_cls=False,
    test_mode=test_mode,
    tag_col="tag",
    text_embed_file=os.path.join(config.data_folder, "vector_embed_Separated_LSTM_with_upload_time.npy"),
    extract_frames=True
)