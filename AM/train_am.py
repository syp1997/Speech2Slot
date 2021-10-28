import os
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint
from cnn_ctc import Am, am_hparams

flags = tf.flags
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string("am_model_name", "model_AM_0915.h5", help='model file')

# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = 'data/'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = True
data_args.batch_size = 16
data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)

print("=======am vocab length:"+str(len(train_data.am_vocab)))

# 1.声学模型训练-----------------------------------
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 2
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

if not os.path.exists('models'):
    os.mkdir('models')
    
save_model_name = FLAGS.am_model_name

if os.path.exists('models/'+save_model_name):
    print('load acoustic model...')
    am.ctc_model.load_weights('models/'+save_model_name)

epochs = 50
batch_num = len(train_data.wav_lst) // train_data.batch_size

# checkpoint
ckpt = "model_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join('am_checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)

batch = train_data.get_am_batch()
print("start training AM model")
if am_args.gpu_nums > 1:
    am.ctc_model_mg.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint], workers=1, use_multiprocessing=False)
else:
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint], workers=1, use_multiprocessing=False)
am.ctc_model.save_weights('models/'+save_model_name)

print("end training AM model")
