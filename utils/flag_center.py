# -*- coding: utf-8 -*-
import tensorflow as tf

tf.flags.DEFINE_bool(name='do_train', default=True, help='开启训练标记')
tf.flags.DEFINE_bool(name='do_eval', default=False, help='开启评估标记')
tf.flags.DEFINE_integer(name="train_batch_size", default=256, help="训练batch size")
tf.flags.DEFINE_integer(name="eval_batch_size", default=8, help="评估batch size")
tf.flags.DEFINE_float(name="learning_rate", default=3e-4, help="Adam 初始学习率.")
tf.flags.DEFINE_float(name="warmup_ratio", default=0.1, help="Adam 的warmup 比例")
tf.flags.DEFINE_float(name="num_train_epochs", default=10.0, help="训练epoch 数量")
tf.flags.DEFINE_string(name="init_checkpoint",  default='', help="Initial checkpoint")
tf.flags.DEFINE_integer(name="save_checkpoints_steps", default=5000, help="保存checkpoint的周期")
tf.flags.DEFINE_bool(name="is_standalone", default=True, help="是不是使用standalone模式")

#sjmk_s: 单机单卡同步_
#mjmk_a: 多机多卡异步
tf.flags.DEFINE_string(name="distribution_mode", default='sjmk_s', help="指定多卡训练模式 ")
tf.flags.DEFINE_integer(name='num_train_sample', default=56000, help='训练样本数量')
# 最大warmup steps
tf.flags.DEFINE_integer(name='num_warmup_steps', default=4000, help='启动部署')
# 训练步数：如果设置为default值(0)，将通过num_train_sample和num_train_epochs来计算num_train_steps
tf.flags.DEFINE_integer(name='num_train_steps', default=0, help='训练步数')
tf.flags.DEFINE_integer(name='pretrain_step', default=0, help='预训练step')
tf.flags.DEFINE_integer(name='begin_step', default=0, help='lr开始步数')
tf.flags.DEFINE_string(name="train_sheet_file_list",  default="./transformer-phone/out/tf_list.txt", help="训练数据位置")
tf.flags.DEFINE_string(name="eval_sheet_file_list",  default="./transformer-phone/out/tf_list.txt", help="训练数据位置")
tf.flags.DEFINE_string(name="embedding_file", default="./dat/phone.ctc.embedding", help="embedding file")
tf.flags.DEFINE_string(name="user_name",  default="", help="user_name:your column in oss")
tf.flags.DEFINE_string(name="model_dir",  default="", help="model_dir:your model checkpoint")


tf.flags.DEFINE_string("ckpt_dir", "./out/ckpt", "standalone 模式 checkpoint 的产出位置")

#用于多卡训练框架专用参数，请不要改动和使用
tf.app.flags.DEFINE_integer("task_index", 0, "Worker task index")
tf.app.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.app.flags.DEFINE_string("job_name", 'worker', "job name: worker or ps")
tf.app.flags.DEFINE_string("buckets", "", "distribute 模式 checkpoint 的产出位置，注意使用前清空这个目录下的非文件夹")


#模型参数
tf.flags.DEFINE_integer('d_model', default=512, help="hidden dimension of encoder/decoder")
tf.flags.DEFINE_integer('d_ff', default=2048, help="hidden dimension of feedforward layer")
tf.flags.DEFINE_integer('num_blocks', default=6, help="number of encoder/decoder blocks")
tf.flags.DEFINE_integer('num_heads', default=8, help="number of attention heads")
tf.flags.DEFINE_integer('maxlen1', default=104, help="maximum length of a source sequence")
tf.flags.DEFINE_integer('maxlen2', default=39, help="maximum length of a target sequence")
tf.flags.DEFINE_float('dropout_rate', default=0.1, help="dropout rate")
tf.flags.DEFINE_float('smoothing', default=0.1, help="label smoothing rate")
tf.flags.DEFINE_integer('vocab_size', default=124, help="phone 词表的大小")
tf.flags.DEFINE_integer('encoder_masked_size', default=10, help="encoder mlm的mask大小")
tf.flags.DEFINE_integer('decoder_masked_size', default=10, help="encoder mlm的mask大小")


tf.flags.DEFINE_string("config_file", default="./transformer-phone/config/p2s_384_l4_seplm.json", help="transformer-bridge 配置")

tf.flags.DEFINE_string("fixed_lm", default="./transformer-phone/fixedlm/transformer_loss_2.139403-22792", help="fixed lm")


tf.flags.DEFINE_string("music_ckpt", default="oss://coin-ailab-id/shiyi.zxh/multi_task/v1_20191201_play_success/model.ckpt-651470", help="fixed lm")

tf.flags.DEFINE_string("encoder_lm", default="./transformer-phone/fixedlm/transformer_loss_2.139403-22792", help="fixed lm")

tf.flags.DEFINE_float('beta', default=0.0001, help="scaling for order loss")

tf.flags.DEFINE_bool('is_fix_encoder', default=False, help='whether to fix encoder param')



FLAGS = tf.flags.FLAGS

