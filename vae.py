import tensorflow as tf

def xavier(fan_in, fan_out, constant=1): 
    ''' 
    Xavier initialization of network weights 
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    '''
    low = -constant*tf.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*tf.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VAE:
    ''' 
    Variation Autoencoder (VAE) implementation using TensorFlow
    based on https://jmetzen.github.io/2015-11-27/vae.html by Jan Hendrik Metzen
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
   
    Parameters
    ----------
    
    Attributes
    ----------
    zmean - the mean of the Gaussian distribution of the latent space

    
    References
    ----------
    
    See 'Auto-Encoding Variational Bayes' by Kingma and Welling for more details.

    '''
    def __init__(self, structure, transfer_fn=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        
        self.X = tf.placeholder(tf.float32, [None, structure[0]])
        
        
        # 1. Create autoencoder structure -------------------------------------
        weights = []
        bias    = []
        
        # recognizer structure
        _layer  = self.X
        
        for i in range(len(structure)-1):
            weights.append(tf.Variable(xavier(structure[i], structure[i+1])))
            bias.append(tf.Variable(tf.zeros([structure[i+1]], dtype=tf.float32)))
            
            if(i == len(structure)-2):
                # additional weight, bias for the zlogvar
                weights.append(tf.Variable(xavier(structure[i], structure[i+1])))
                bias.append(tf.Variable(tf.zeros([structure[i+1]], dtype=tf.float32)))
                
                self.zmean = tf.add(tf.matmul(_layer, weights[i]), bias[i])
                self.zlogvar = tf.add(tf.matmul(_layer, weights[i+1]), bias[i+1])
                
            else:
                _layer = tf.nn.softplus(tf.add(tf.matmul(_layer, weights[i]), bias[i]))  
        
        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((batch_size, structure[-1]), 0, 1, dtype=tf.float32)
        
        # z = mu + zlogvar*epsilon
        self.Z = tf.add(self.zmean, tf.mul(tf.sqrt(tf.exp(self.zlogvar)), eps))
        
        # (TODO): extract labels
        self.labels = tf.nn.softmax(self.Z)
        
        # generator structure
        _layer = self.Z
        
        for i in range(len(structure)-1):
            weights.append(tf.Variable(xavier(structure[len(structure)-i-1], structure[len(structure)-i-2])))
            bias.append(tf.Variable(tf.zeros([structure[len(structure)-i-2]], dtype=tf.float32)))
            
            if(i == len(structure)-2):
                self.rmean = tf.nn.sigmoid(tf.add(tf.matmul(_layer, weights[i+len(structure)]), bias[i+len(structure)]))
            else:
                _layer = tf.nn.softplus(tf.add(tf.matmul(_layer, weights[i+len(structure)]), bias[i+len(structure)]))
        
        # 2. Calculate the cost --------------------------------------------
        
        # The first component of the cost to be minimized is the _reconstruction loss_.
        reconstruction_loss = -tf.reduce_sum(self.X * tf.log(1e-10 + self.rmean) + (1-self.X) * tf.log(1e-10 + 1 - self.rmean), 1)
        
        # The second component of the cost is the _latent loss_
        latent_loss = -0.5 * tf.reduce_sum(1 + self.zlogvar - tf.square(self.zmean) - tf.exp(self.zlogvar), 1)

        # calculate the cost
        self.cost = tf.reduce_mean(reconstruction_loss + latent_loss) 
        
        # setup the optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        # 3. Initialize the variables and launch the session 
        tf.set_random_seed(42)
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)  
        
    def fit(self, X):
        '''
        Train model based on mini-batch of input data.
        
        Parameters
        ----------
        
        Return
        ------
        Return cost of mini-batch
        A class label
        '''
        opt, cost, labels = self.sess.run((self.optimizer, self.cost, self.labels), feed_dict={self.X: X})
        return cost, labels
    
    def transform(self, X):
        '''
        Transform data by mapping it into the latent space.
        Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        
        Parameters
        ----------
        
        Return
        ------
        
        '''
        return self.sess.run(self.zmean, feed_dict={self.X: X})
    
    def generate(self, zmu=None):
        ''' 
        Generate data by sampling from latent space.
        If zmu is not None, data for this point in latent space is
        generated. Otherwise, zmu is drawn from prior in latent 
        space.        
        
        Note: This maps to mean of distribution, we could alternatively 
        sample from Gaussian distribution
        
        Parameters
        ----------
        
        Return
        ------

        '''
        if zmu is None:
            zmu = tf.random.normal(size=self.structure[-1])
        return self.sess.run(self.rmean, feed_dict={self.Z: zmu})
    
    def reconstruct(self, X):
        ''' 
        Use VAE to reconstruct given data. 
        
        Parameters
        ----------
        
        Return
        ------
        
        '''
        reconstruction, labels = self.sess.run((self.rmean, self.labels), feed_dict={self.X: X})
        return reconstruction, labels
