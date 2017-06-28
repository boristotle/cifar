
import pickle
meta_file = open('cifar-10-batches-py/batches.meta', "rb")
dict = pickle.load(meta_file)

label_names = dict["label_names"]
print('dict', dict["label_names"])



img_data_file = open('cifar-10-batches-py/data_batch_1', "rb")
img_dict = pickle.load(img_data_file, encoding="latin1")
print('img_data', img_dict['data'])


print('img_labels', img_dict['labels'])