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
    "hidden_length": [512],
    "layer_number": [2, 3],
    "linear_hidden_length": [256, 128],
    "drop_out_rate": [0.5],
    "batch_size": [400, 600, 800],
    "learning_rate": 1e-4,
    "training_size": 0.8,
    "number_epochs": 100,
}


"Test tags, without advid"
tag_names = ["cost", "like", "bclk", "negative", "clk", "play3s", "like_clk", "bclk_clk", "cost_clk", "negative_clk",
                 "like_play3s", "bclk_play3s", "cost_play3s", "negative_play3s", "play3s_clk"]
# tag_names = ["cost_clk"]
# np.random.shuffle(tag_names)
# for tag in tag_names:
#     tag_col = "tag_" + tag
#     mean_tag_col = "mean_tag_" + tag
    # logger = init_logger(
    #     log_level=logging.INFO,
    #     name=f"{tag_col}_test_Separated_LSTM_without_advid",
    #     write_to_file=True,
    #     clean_up=False,
    # )
    # adjust_hyperparams(
    #     logger,
    #     params,
    #     sample_number=3,
    #     model_name="SeparatedLSTM",
    #     run_model=text_scorer,
    #     test_params=test_params,
    #     embed_type="local",
    #     force_embed=False,
    #     use_cls=False,
    #     test_mode=test_mode,
    #     tag_col=tag_col,
    #     embed_file=os.path.join(config.data_folder, "vector_embed_Attention_LSTM_with_cls_extra.npy")
    # )
    # logger = init_logger(
    #     log_level=logging.INFO,
    #     name=f"{mean_tag_col}_test_Separated_LSTM_without_advid",
    #     write_to_file=True,
    #     clean_up=False,
    # )
    # adjust_hyperparams(
    #     logger,
    #     params,
    #     sample_number=3,
    #     model_name="SeparatedLSTM",
    #     run_model=text_scorer,
    #     test_params=test_params,
    #     embed_type="local",
    #     force_embed=False,
    #     use_cls=False,
    #     test_mode=test_mode,
    #     tag_col=mean_tag_col,
    #     embed_file=os.path.join(config.data_folder, "vector_embed_Attention_LSTM_with_cls_extra.npy")
    # )

tag_name = "mean_tag_like"
logger = init_logger(
    log_level=logging.INFO,
    name="Separated_LSTM_mean_like",
    write_to_file=True,
    clean_up=False,
)
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=5,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     tag_col=tag_name,
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

# double_net
# test_params = {
#     "hidden_length": [128, 256, 512],
#     "drop_out_rate": [0.5, 0.4, 0.6],
#     "batch_size": [400, 800, 1200],
#     "learning_rate": 1e-4,
#     "training_size": 0.8,
#     "number_epochs": 100,
# }
# logger = init_logger(
#     log_level=logging.INFO,
#     name="DoubleNet_mean_like_cls",
#     write_to_file=True,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=10,
#     model_name="DoubleNet",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=True,
#     tag_col=tag_name,
#     embed_file=os.path.join(config.data_folder, "vector_embed_Separated_LSTM_mean_like_cls.npy"),
#     test_mode=test_mode
# )

# with cls
# test_params = {
#     "hidden_length": [512],
#     "layer_number": [3],
#     "linear_hidden_length": [512],
#     "drop_out_rate": [0.5],
#     "batch_size": [20],
#     "learning_rate": 1e-4,
#     "training_size": 0.8,
#     "number_epochs": 100,
# }
# logger = init_logger(
#     log_level=logging.INFO,
#     name="Separated_LSTM_mean_like_cls",
#     write_to_file=True,
#     clean_up=False,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=5,
#     model_name="SeparatedLSTM",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=True,
#     tag_col=tag_name,
#     test_mode=test_mode
# )

test_params = {
    "hidden_length": [256, 512],
    "layer_number": [2, 3],
    "linear_hidden_length": [256, 128],
    "drop_out_rate": [0.5],
    "batch_size": [100, 400, 600, 800],
    "learning_rate": 1e-4,
    "training_size": 0.8,
    "number_epochs": 100,
}
logger = init_logger(
    log_level=logging.INFO,
    name="Attention_LSTM_mean_like_cls",
    write_to_file=True,
    clean_up=False,
)
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=5,
#     model_name="BiLSTMWithAttention",
#     run_model=text_scorer,
#     test_params=test_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=True,
#     test_mode=test_mode,
#     tag_col=tag_name,
#     embed_file=os.path.join(config.data_folder, "vector_embed_Separated_LSTM_mean_like_cls.npy")
# )

def run_seprated_lstm_with_neighbors():
    # with more texts
    test_params = {
        "hidden_length": [256, 512],
        "layer_number": [2, 3],
        "linear_hidden_length": [256, 128],
        "drop_out_rate": [0.5],
        "batch_size": [400, 600, 800],
        "learning_rate": 1e-4,
        "training_size": 0.8,
        "number_epochs": 100,
    }


    logger = init_logger(
        log_level=logging.INFO,
        name="Separated_LSTM_mean_like_neighbors_center",
        write_to_file=True,
        clean_up=False,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=1,
        model_name="SeparatedLSTM",
        run_model=text_scorer,
        test_params=test_params,
        embed_type="local",
        force_embed=False,
        use_cls=False,
        neighbor=1,
        only_center=True,
        max_len=100,
        test_mode=test_mode,
        tag_col=tag_name,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=5,
        model_name="SeparatedLSTM",
        run_model=text_scorer,
        test_params=test_params,
        embed_type="local",
        force_embed=False,
        use_cls=False,
        neighbor=1,
        only_center=True,
        max_len=100,
        test_mode=test_mode,
        tag_col=tag_name,
    )


# bertcnn/mlp
cnn_params = {
                "hidden_length": [32, 16, 8],
                "linear_hidden_length": [32, 64, 128],
                "grad_layer_name": ["none"],
                "drop_out_rate": 0.5,
                "channels": [64, 128],
                "batch_size": [100],
                "learning_rate": 1e-5,
                "training_size": 0.8,
                "number_epochs": 100,
            }
mlp_params = {
                "linear_hidden_length": [256, 512],
                "grad_layer_name": ["encoder.layer.23.attention.self.query.weight"],
                "drop_out_rate": 0.5,
                "batch_size": [200],
                "learning_rate": [1e-5],
                "training_size": 0.8,
                "number_epochs": 100,
            }
# logger = init_logger(
#     log_level=logging.INFO,
#     name="BertCNN_mean_like_without_tune",
#     write_to_file=True,
#     clean_up=True,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=5,
#     model_name="BertWithCNN",
#     run_model=text_scorer,
#     test_params=cnn_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     tag_col=tag_name,
#     test_mode=test_mode
# )

# logger = init_logger(
#     log_level=logging.INFO,
#     name="BertMLP_mean_like",
#     write_to_file=True,
#     clean_up=False,
# )
# adjust_hyperparams(
#     logger,
#     params,
#     sample_number=5,
#     model_name="BertWithMLP",
#     run_model=text_scorer,
#     test_params=mlp_params,
#     embed_type="local",
#     force_embed=False,
#     use_cls=False,
#     tag_col=tag_name,
#     test_mode=test_mode
# )

# videonet
def run_video_net_with_embed():
    config.embed_data_folder = r"E:\frames_embedding"
    video_params = {
        "frame_rate": 1,
        # 用embed设为1792
        "input_length": [768],
        "pic_linear_hidden_length": 1024,
        "pic_grad_layer_name": "aaa_blocks.31",
        "pic_drop_out_rate": [0.5, 0.3, 0.6],
        "pic_channels": 3,
        "layer_number": 3,
        "hidden_length": 512,
        "linear_hidden_length": 128,
        "drop_out_rate": 0.5,
        "batch_size": [200, 400, 600],
        "learning_rate": [1e-4, 1e-5],
        "training_size": 0.8,
        "number_epochs": 100,
        "random": 1,
    }
    logger = init_logger(
            logging.INFO, name="Video_LSTM_with_vit_embed_mean_like", write_to_file=True, clean_up=False
        )

    adjust_hyperparams(
        logger, params, 5, "VideoNetEmbed", video_scorer, test_params=video_params, extract_frames=True, tag_col=tag_name
    )

# Joint Net
def run_joint_net():
    # vit_embedding
    config.embed_data_folder = r"E:\frames_embedding"
    test_params = {
                    "text_hidden_length": [512, 256],
                    "text_layer_number": [2, 3],
                    "text_linear_hidden_length": [128, 256],
                    "text_drop_out_rate": [0.6],
                    # "text_batch_size": [400, 600, 800],
                    # "text_learning_rate": 1e-4,
                    # "text_training_size": 0.8,
                    # "text_number_epochs": 100,
                    "video_frame_rate": 1,
                    "video_input_length": [768],
                    "video_pic_linear_hidden_length": 1024,
                    "video_pic_grad_layer_name": "aaa_blocks.31",
                    # "video_pic_drop_out_rate": [0.5, 0.3, 0.6],
                    "video_pic_channels": 3,
                    "video_layer_number": [1, 2, 3],
                    "video_hidden_length": [512],
                    "video_linear_hidden_length": [128, 256],
                    "video_drop_out_rate": [0.6],
                    "batch_size": [200, 400, 600],
                    "learning_rate": [1e-4, 1e-5],
                    "training_size": 0.8,
                    "number_epochs": 100,
                    "random": [0, 1],
                }
    logger = init_logger(
        log_level=logging.INFO,
        name="Separated_LSTM_with_cls&Video_with_vit_embed_mean_like",
        write_to_file=True,
        clean_up=False,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=10,
        test_params=test_params,
        model_name="JointNet",
        run_model=video_scorer,
        embed_type="local",
        use_cls=False,
        test_mode=test_mode,
        tag_col=tag_name,
        text_embed_file=os.path.join(config.data_folder, "vector_embed_Separated_LSTM_mean_like_cls.npy")
    )
def run_video_attention():
    test_params = {
                "linear_length": 512,
                "linear_hidden_length": 128,
                "frames_per_clip": 16,
                "grad_layer_name": "aaa_blocks.31",
                "img_size": 180,
                "drop_out_rate": 0.5,
                "batch_size": [80],
                "learning_rate": 1e-5,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            }
    logger = init_logger(
        log_level=logging.INFO,
        name="STAM",
        write_to_file=True,
        clean_up=False,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=10,
        test_params=test_params,
        model_name="VideoAttention",
        run_model=video_scorer,
        embed_type="local",
        test_mode=test_mode,
        tag_col=tag_name,
        extract_frames=True
    )

def run_video_attention_with_embed():
    test_params = {
                "linear_length": 512,
                "linear_hidden_length": [128, 256],
                "frames_per_clip": 16,
                "grad_layer_name": "aaa_blocks.31",
                "img_size": 180,
                "drop_out_rate": 0.5,
                "batch_size": [200, 400],
                "learning_rate": 1e-5,
                "training_size": 0.8,
                "number_epochs": 100,
                "random": 1,
            }
    logger = init_logger(
        log_level=logging.INFO,
        name="STAM_mean_like_embed_full_layers_efficientembed",
        write_to_file=True,
        clean_up=False,
    )
    adjust_hyperparams(
        logger,
        params,
        sample_number=10,
        test_params=test_params,
        model_name="VideoAttentionEmbed",
        run_model=video_scorer,
        embed_type="local",
        test_mode=test_mode,
        tag_col=tag_name,
        extract_frames=True
    )

if __name__ == "__main__":
    # run_video_attention()
    # run_video_attention_with_embed()
    # run_video_net_with_embed()
    run_joint_net()