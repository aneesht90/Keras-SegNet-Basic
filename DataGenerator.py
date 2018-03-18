import numpy as np

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, labels, data):
      'Generates batches of samples'
      # Infinite loop

      while 1:
          # Generate order of exploration of dataset
          #indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax  = int(len(data)/self.batch_size)
          #data /=255
          #print("print data shape",np.array(data).shape)
          #print("print data type",data.dtype)
          for step in range(0,imax-1):
              yield data[step*self.batch_size : (step+1)*self.batch_size], labels[step*self.batch_size : (step+1)*self.batch_size]

#   def __get_exploration_order(self, list_IDs):
#       'Generates order of exploration'
#       # Find exploration order
#       indexes = np.arange(len(list_IDs))
#       if self.shuffle == True:
#           np.random.shuffle(indexes)
#
#       return indexes
#
#   def __data_generation(self, labels, list_IDs_temp):
#       'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
#       # Initialization
#       X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
#       y = np.empty((self.batch_size), dtype = int)
#
#       # Generate data
#       for i, ID in enumerate(list_IDs_temp):
#           # Store volume
#           X[i, :, :, :, 0] = np.load(ID + '.npy')
#
#           # Store class
#           y[i] = labels[ID]
#
#       return X, sparsify(y)
#
# def sparsify(y):
#   'Returns labels in binary NumPy array'
#   n_classes = 12# Enter number of classes
#   return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
#                    for i in range(y.shape[0])])
