"""Define the model."""

import tensorflow as tf


def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = inputs['sentence']

    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)

        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        # Output is sequence of outputs for each cell, state is the final state
        output, state  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # output, state  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)

        # Take mean of all cell outputs
        avg_output = tf.reduce_mean(output, axis = 1) 
        # State is a tuple hidden state output and activated output
        c, h = state
        # Compute logits from the output of the LSTM
        # logits = tf.layers.dense(avg_output, params.number_of_tags)

        # Compute logits from the last cell output of the LSTM
        # Try this and see if it works..?
        last_output = tf.gather(output, indices = tf.shape(output)[1] - 1, axis = 1)
        logits = tf.layers.dense(last_output, params.number_of_tags)

        # print('logits:',logits)


    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))


    # Return logits, but also return average activations
    return logits, avg_output, output


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.reshape(labels, [-1])
    sentence_lengths = inputs['sentence_lengths']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits, avg_output, outputs = build_model(mode, inputs, params)
        predictions = tf.argmax(logits, -1)

        # Add something to record the activations here..?
        # I should obtain it from build model!  ^^ above... return multiple args



    # Find weights so that we can regularize 
    w = [w for w in tf.trainable_variables() if 'lstm_cell/kernel' in w.name]
    reg_loss = tf.reduce_sum(tf.nn.l2_loss(w))

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # TODO: Confirm that a mask is not necessary for this problem
#    mask = tf.sequence_mask(sentence_lengths)
#    print('mask:',mask)
#    exit(0)
#    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses) + params.reg_strength*reg_loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['avg_output'] = avg_output
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['outputs'] = outputs

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
