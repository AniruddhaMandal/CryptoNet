import gzip
import numpy as np

def get_data(data_file, data_labels_file):

	with gzip.open(data_file) as f:
		magic_number = int.from_bytes(f.read(4),"big")
		image_count = int.from_bytes(f.read(4),"big")
		row_count = int.from_bytes(f.read(4),"big")
		column_count = int.from_bytes(f.read(4),"big")


		data=np.frombuffer(f.read(),dtype=np.uint8).reshape((image_count, row_count*column_count))


	with gzip.open(data_labels_file) as f:
		magic_number = int.from_bytes(f.read(4),"big")
		image_count = int.from_bytes(f.read(4),"big")

		data_labels = np.frombuffer(f.read(), dtype=np.uint8)

	return data, vectorize(data_labels)

def vectorize(array):
    output = np.zeros((len(array), 10))
    for i in range(len(array)):
        a = np.zeros(10)
        a[array[i]] = 1
        output[i] = a
    return output

