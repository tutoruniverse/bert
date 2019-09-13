from nishant_run_classifier import *




class Model:
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.FATAL)

        self.processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mrpc": MrpcProcessor,
            "xnli": XnliProcessor,
            "raw" : RawQPInputProcessor
        }

        tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                      FLAGS.init_checkpoint)

        if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.raw_predict:
          raise ValueError(
              "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        if FLAGS.max_seq_length > self.bert_config.max_position_embeddings:
          raise ValueError(
              "Cannot use sequence length %d because the BERT model "
              "was only trained up to sequence length %d" %
              (FLAGS.max_seq_length, self.bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(FLAGS.output_dir)

        self.task_name = FLAGS.task_name.lower()

        if self.task_name not in self.processors:
          raise ValueError("Task not found: %s" % (task_name))


        self.processor = self.processors[self.task_name]()

        self.label_list = self.processor.get_labels()

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        self.tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
          self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=self.tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=self.is_per_host))

        self.train_examples = None
        self.num_train_steps = None
        self.num_warmup_steps = None
        if FLAGS.do_train:
          self.train_examples = processor.get_train_examples(FLAGS.data_dir)
          self.num_train_steps = int(
              len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
          self.num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        self.model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)




    def _predict(self):
      if FLAGS.do_predict or FLAGS.raw_predict:
        predict_examples = self.processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
          # TPU requires a fixed batch size for all batches, therefore the number
          # of examples must be a multiple of the batch size, or else examples
          # will get dropped. So we pad with fake examples which are ignored
          # later on.
          while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())


        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, self.label_list,
                                                FLAGS.max_seq_length, self.tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)


        result = self.estimator.predict(input_fn=predict_input_fn)
        scores = []
        # if FLAGS.do_predict:
        #     output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        #     with tf.gfile.GFile(output_predict_file, "w") as writer:
        #       num_written_lines = 0
        #       tf.logging.info("***** Predict results *****")
        #       for (i, prediction) in enumerate(result):
        #         probabilities = prediction["probabilities"]
        #         if i >= num_actual_predict_examples:
        #           break
        #         output_line = "\t".join(
        #             str(class_probability)
        #             for class_probability in probabilities) + "\n"
        #         writer.write(output_line)
        #         num_written_lines += 1

        num_written_lines = 0
        for (i, prediction) in enumerate(result):
            probabilities = prediction['probabilities']
            if i >= num_actual_predict_examples:
                break
            scores.append(probabilities[1])
            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        return scores





    def predict(self, lst):
        FLAGS.sentences = lst
        return self._predict()



if __name__ == '__main__':
    model = Model()
    print(model.predict([("The building was destroyed by the monster.",
                "The monster destroyed the building."), ("Hello.", "How are you.")]))
