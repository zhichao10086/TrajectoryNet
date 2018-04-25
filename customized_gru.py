import tensorflow as tf
import linear
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class CustomizedGRU(RNNCell):

    def __init__(self,num_units,maxOut_numUnits,activation = tanh):
        self._num_units = num_units
        self._activation = activation
        self._maxOut_numUnits = maxOut_numUnits


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with vs.variable_scope(scope or "gru_cell"):
            with vs.variable_scope("gates"):
                gate = linear._linear([inputs,state],2*self._num_units,True,1.0,scope = scope)

                r,u = array_ops.split(gate,num_or_size_splits=2,axis=1)

                r,u = sigmoid(r),sigmoid(u)

            with vs.variable_scope("candidate"):

                c = self.maxout(inputs,r*state,self._maxOut_numUnits,0,self._num_units,scope= scope)

            new_h = u*state +(1-u)*c

        return new_h,new_h



    def maxout(self, input1, input2, num_units, ini_value, output_size, scope=None):
        shape = input1.get_shape().as_list()
        dim = shape[-1]
        outputs = None
        for i in range(num_units):
            with tf.variable_scope(str(i)):
                y = self._activation(linear._linear([input1, input2],output_size, True, ini_value,scope=scope))
                if outputs is None:
                    outputs = y
                else:
                    outputs = tf.maximum(outputs, y)
        c = outputs
        return c
