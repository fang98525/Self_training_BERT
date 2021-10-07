# -*- coding: utf-8 -*-
# @Time    : 2021/1/12
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm


class Config(object):
    def __init__(self):
        # -----------ARGS---------------------
        # 原始数据fold路径
        self.source_data_path = '/data1/home/fzq/projects/self_pretrained/dataset/'
        # 数据预处理后的预训练数据路径
        self.pretrain_train_path = "/data1/home/fzq/projects/self_pretrained/dataset/256_corpus.txt"
        # 模型保存路径
        self.output_dir = self.source_data_path + "outputs_model/"
        # MLM任务验证集数据，大多数情况选择不验证（predict需要时间,知道验证集只是表现当前MLM任务效果）
        self.pretrain_dev_path = ""

        # 预训练模型所在路径（文件夹）为''时从零训练，不为''时继续训练。huggingface roberta
        # 下载链接为：https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
        #'/data1/home/fzq/projects/nlpclassification/data/model/RoBERTa_zh_L12_PyTorch/'  # 使用nezha或者Roberta 的预训练模型
        self.pretrain_model_path = "/data1/home/fzq/projects/nlpclassification/data/model/RoBERTa_zh_L12_PyTorch/"
        self.bert_config_json = self.pretrain_model_path + "config.json"  # 为''时从零训练  "config.json"
        self.vocab_file = self.pretrain_model_path + "vocab.txt" # "vocab.txt"
        self.init_model = self.pretrain_model_path

        self.max_seq_length = 256  # 文本长度
        self.do_train = True
        self.do_eval = True
        self.do_lower_case = False  # 数据是否全变成小写（是否区分大小写）

        self.train_batch_size =4 # 根据GPU卡而定
        self.eval_batch_size = 4  #原始32
        # 继续预训练lr：5e-5，重新预训练：1e-4
        self.learning_rate = 5e-5
        self.num_train_epochs = 16  # 预训练轮次
        self.save_epochs = 2  # e % save_epochs == 0 保存
        # 前warmup_proportion的步伐 慢热学习  一开始的lr较小，每次增加0，1倍
        self.warmup_proportion = 0.1
        self.dupe_factor = 1  # 动态掩盖倍数
        self.no_cuda = False  # 是否使用gpu  false 表示使用
        self.local_rank = -1  # 分布式训练  不为-1 则
        self.seed = 42  # 随机种子

        # 梯度累积（相同显存下能跑更大的batch_size）1不使用   多个样本的梯度累计后再进行梯度更新
        self.gradient_accumulation_steps = 1
        self.fp16 = False  # 混合精度训练
        self.loss_scale = 0.  # 0时为动态   fp16的optimizer一般使用动态效果更好
        # bert Transormer的参数设置
        self.masked_lm_prob = 0.15  # 掩盖率
        # 最大掩盖字符数目
        self.max_predictions_per_seq = 20
        # 冻结word_embedding参数  加快训练 提高训练效率
        self.frozen = True

        # bert参数解释
        """
        {
          # 乘法attention时，softmax后dropout概率
          "attention_probs_dropout_prob": 0.1,  
          "directionality": "bidi", 
          "hidden_act": "gelu", # 激活函数
          "hidden_dropout_prob": 0.1, #隐藏层dropout概率
          "hidden_size": 768, # 最后输出词向量的维度
          "initializer_range": 0.02, # 初始化范围
          "intermediate_size": 3072, # 升维维度
          "max_position_embeddings": 512, # 最大的
          "num_attention_heads": 12, # 总的头数
          # 隐藏层数 ，也就是transformer的encode运行的次数
          "num_hidden_layers": 12, 
          "pooler_fc_size": 768, 
          "pooler_num_attention_heads": 12, 
          "pooler_num_fc_layers": 3, 
          "pooler_size_per_head": 128, 
          "pooler_type": "first_token_transform", 
          "type_vocab_size": 2, # segment_ids类别 [0,1]
          "vocab_size": 21128  # 词典中词数  
        }
        """
