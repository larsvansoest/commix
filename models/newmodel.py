import tensorflow as tf

from models import AbstractModel
from utils.ops import uv_affine_transform


class NewModel(AbstractModel):
    """
    Contains the architecture of the TransWeight composition model.
    Takes as input of two vectors (or batches of vectors). Its superclass is the AbstractModel,
    all properties are defined and described there.
    """

    def __init__(self, embedding_size, nonlinearity, dropout_rate, transforms):
        super(NewModel, self).__init__(embedding_size)
        self._nonlinearity = nonlinearity
        self._dropout_rate = dropout_rate
        self._transforms = transforms # the number of transformations to use

    def create_architecture(self):

        self._transformations_tensor = tf.get_variable("transformations_tensor",
                        shape=[self.transforms, 2*self.embedding_size, self.embedding_size])
        self._transformations_bias = tf.get_variable("transformations_bias",
                        shape=[self.transforms, self.embedding_size])

        # rank 3 combination tensor - combines the transformed representations in the previous step
        self._W = tf.get_variable("W", shape=[self.transforms, self.embedding_size, self.embedding_size])

        # bias vector for the combination tensor
        self._Wb = tf.get_variable("Wb", shape=[self.embedding_size, self.embedding_size])

        self._V = tf.get_variable("V", shape=[self.embedding_size, self.embedding_size])

        # bias vector for the combination tensor
        self._Vb = tf.get_variable("Vb", shape=[self.embedding_size])

        self._architecture = self.compose(
            u=self.embeddings_u,
            v=self.embeddings_v,
            transformations_tensor=self.transformations_tensor, 
            transformations_bias=self.transformations_bias,
            W=self.W, 
            Wb=self.Wb,
            V=self.V,
            Vb=self.Vb)

        self._architecture_normalized = super(
            NewModel,self).l2_normalization_layer(self._architecture,1)

    def transform(self, u, v, transformations_tensor, transformations_bias):
        # batch_size x 2embedding_size
        uv = tf.concat(values=[u, v], axis=1)

        # create all the transformations of the input vectors u and v
        # batch_size x 2embedding_size * transformations x 2embedding_size x embedding_size -> 
        # batch_size x transformations x embedding_size
        trans_uv = tf.tensordot(uv, transformations_tensor, axes=[[1], [1]])

        # add biases
        # batch_size x transformations x embedding_size + transformations x embedding_size (auto broadcast)
        trans_uv_bias_sum = tf.add(trans_uv, transformations_bias)

        return trans_uv_bias_sum        

    def weight(self, reg_uv, W, V, Wb, Vb, _batchsize):
        # transformations are weighted using W into a final composed representation
        
        # option2: W(t x n x n) . H('b' x t x n) -> I('b' x n x n)
        reg_uv.set_shape([_batchsize, self.transforms, self.embedding_size]) # Combats tf.einsum unknown shape rank error.
        weighted_W_uv_opt2 = tf.einsum('btn,tmn->bmn', reg_uv, W)
        weighted_W_uv_opt2_bias = tf.add(weighted_W_uv_opt2, Wb)

        # option2.2: V(n x n) . I('b' x n x n) -> J('b' x n)
        #weighted_V_W_uv_opt2 = tf.tensordot(weighted_W_uv_opt2_bias, V, axes=[[1], [0]])
        #weighted_V_W_uv_opt2_bias = tf.add(weighted_V_W_uv_opt2, Vb)

        with tf.control_dependencies([weighted_W_uv_opt2_bias]):
            return tf.add(tf.ones([100,200]), Vb)

    def compose(self, u, v, transformations_tensor, transformations_bias, W, V, Wb, Vb):
        """
        composition of the form:
        p = W[T1[u;v]+b_1;T2[u;v]+b_2;T3[u;v]+b_3; ...; T_t[u;v]+b_t] + b
        """

        # perform all transformations
        transformed_uv = self.transform(u, v, transformations_tensor, transformations_bias)

        # apply dropout and nonlinearity
        reg_uv = self.nonlinearity(
            tf.layers.dropout(transformed_uv, rate=self.dropout_rate, training=self.is_training))

        # weight the transformations into the final composed representation
        weighted_transformations = self.weight(
            reg_uv, 
            W, 
            V, 
            Wb, 
            Vb,
            reg_uv.get_shape()[0])

        return weighted_transformations

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def transformations_tensor(self):
        return self._transformations_tensor
        
    @property
    def transformations_bias(self):
        return self._transformations_bias
        
    @property
    def W(self):
        return self._W
    
    @property
    def V(self):
        return self._V

    @property
    def Vb(self):
        return self._Vb

    @property
    def Wb(self):
        return self._Wb

    @property
    def nonlinearity(self):
        return self._nonlinearity

    @property
    def transforms(self):
        return self._transforms
