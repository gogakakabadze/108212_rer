
# Configure an ML pipeline, which consists of three stages: tokenizer,
hashingTF, and lr.
tokenizer = Tokenize(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(),
 outputCol="features")
lr = LogisticRegression(maxIter=model_config["maxIter"],
 regParam=model_config["regParam"])
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])