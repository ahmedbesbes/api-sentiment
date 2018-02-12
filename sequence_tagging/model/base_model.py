import os
import tensorflow as tf
import pickle


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config, FLAGS):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config=config
        self.FLAGS=FLAGS
        #        self.logger=config.logger
        self.sess=None
        self.saver=None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables=tf.contrib.framework.get_variables(scope_name)
        init=tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m=lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer=tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer=tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer=tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer=tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs=zip(*optimizer.compute_gradients(loss))
                grads, gnorm=tf.clip_by_global_norm(grads, clip)
                self.train_op=optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op=optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.config.logger.info("Initializing tf session")
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        # self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def close_session(self):
        """close session

        Args:
            sess: tf.Session()

        """
        self.sess.close()

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.FLAGS.dir_model):
            os.makedirs(self.FLAGS.dir_model)
        self.saver.save(self.sess, self.FLAGS.dir_model)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged=tf.summary.merge_all()
        self.file_writer=tf.summary.FileWriter(self.FLAGS.dir_output, self.sess.graph)

    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score=0
        nepoch_no_imprv=0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.FLAGS.nepochs):
            self.config.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.FLAGS.nepochs))

            score=self.run_epoch(train, dev, epoch)
            self.FLAGS.lr*=self.FLAGS.lr_decay  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv=0
                self.save_session()
                best_score=score
                self.config.logger.info("- new best score!")
            else:
                nepoch_no_imprv+=1
                if nepoch_no_imprv >= self.FLAGS.nepoch_no_imprv:
                    self.config.logger.info("- early stopping {} epochs without " \
                                            "improvement".format(nepoch_no_imprv))
                    break

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.config.logger.info("Testing model over test set")
        metrics=self.run_evaluate(test)
        msg=" - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.config.logger.info(msg)

    def evaluate_file(self, fileinput, fileoutput, fr_nlp):
        """Evaluate model on file samples

        Args:
            res: list of chunks and their tags

        """
        self.config.logger.info("Testing model over file samples")
        results=[]
        with open(fileinput, "r") as f:
            for line in f:
                # print(line)
                line=line.strip()
                sentence_nlp=fr_nlp(line)
                words_raw=[]
                words_raw.extend([sp.text for sp in sentence_nlp])
                # print(words_raw)
                _, res=self.predict(words_raw)
                # print(res)
                results.append(res)

        with open(fileoutput, "wb") as fp:
            pickle.dump(results, fp)

        return results
