.. _usage:

###########
Basic Usage
###########

Welcome to the core usage guide for OmniGenBench! This page provides a hands-on introduction to the main functionalities of the library. You'll learn how to perform end-to-end benchmarking, train your own models, run inference, and manage datasets.
All examples assume you have already installed OmniGenBench. If you haven't, please see the :doc:`installation` guide first.

The library is designed around a few key components: `AutoBench` for evaluation, `AutoTrain` for training, and model classes for inference. Let's see them in action.

************************
End-to-End Benchmarking
************************

The quickest way to evaluate a model on a benchmark task is to use the ``AutoBench`` class. It automates the entire process: downloading the dataset, loading the model, running evaluation, and reporting metrics.

Here's how to evaluate the `yangheng/OmniGenome-186M <https://huggingface.co/yangheng/OmniGenome-186M>`_ model on the "RGB" benchmark dataset:

.. code-block:: python

   from omnigenbench import AutoBench

   # Initialize the benchmarker with the benchmark name and a model identifier
   bench = AutoBench(benchmark_name="RGB", model_name_or_path="yangheng/OmniGenome-186M")

   # Run the benchmark. This will handle everything automatically.
   bench.run()

The results, including scores and logs, will be saved to a local directory for analysis.

**********************
Training a New Model
**********************

OmniGenBench simplifies the training process with the ``AutoTrain`` class. You provide a dataset and a base model, and it handles the rest.

In this example, we'll fine-tune the `yangheng/OmniGenome-186M` model on a custom dataset named "MyCustomDataset".

.. code-block:: python

   from omnigenbench import AutoTrain

   # Initialize the trainer with your dataset and a base model
   trainer = AutoTrain(dataset_name="MyCustomDataset", model_name_or_path="yangheng/OmniGenome-186M")

   # Start the training process
   trainer.run()

.. tip::
   Your dataset should be prepared in a compatible format. Refer to the "Preparing Datasets" section for more details on data formatting.

*********************
Inference with a Model
*********************

Once you have a trained model, running inference is straightforward. You can load any compatible model directly using its specific class, like ``OmniModelForSequenceClassification``.

This example shows how to load a model from the Hugging Face Hub and get predictions for a given sequence.

.. code-block:: python

   from omnigenbench import OmniModelForSequenceClassification

   # Load a pre-trained sequence classification model
   model = OmniModelForSequenceClassification.from_pretrained("yangheng/OmniGenome-186M")

   # Define your input sequence
   rna_sequence = "ACGUAGGUAUCGUAGA"

   # Get the prediction
   result = model.predict(rna_sequence)

   print(result)


.. ************************************
.. Downloading Benchmarks (Datasets)
.. ************************************

.. .. code-block:: python

..    from omnigenbench.utility.hub_utils import download_model, download_benchmark
..    download_model("OmniGenome-186M-SSP")
..    download_benchmark("RGB")



************************************
Managing Datasets and Models Manually
************************************

While the ``AutoBench`` and ``AutoTrain`` pipelines handle asset downloads automatically, you might need to download models or benchmark datasets manually in certain scenarios, such as:

*   Pre-loading assets in an environment with limited internet access.
*   Inspecting the contents of a benchmark dataset.
*   Scripting custom workflows.

The ``omnigenbench.utils.hub_utils`` module provides simple functions for this purpose. These functions download files from the Hugging Face Hub and store them in a local cache for future use, avoiding redundant downloads.

.. tip::
   The first time you download an asset, it might take a while depending on its size and your connection speed. Subsequent calls for the same asset will be nearly instant as it will be loaded directly from your local cache.


To download a specific benchmark dataset, use the ``download_benchmark`` function. Provide the benchmark's name as an argument.

.. code-block:: python

   from omnigenbench.utils.hub_utils import download_benchmark

   # Define the name of the benchmark to download
   benchmark_name = "RGB"

   # Download the dataset from the Hugging Face Hub
   local_path = download_benchmark(benchmark_name)

   print(f"Benchmark '{benchmark_name}' downloaded successfully to: {local_path}")


Similarly, the ``download_model`` function allows you to fetch a pre-trained model. Use the model's identifier from the Hub.

.. code-block:: python

   from omnigenbench.utils.hub_utils import download_model

   # Define the model identifier from the Hugging Face Hub
   model_id = "OmniGenome-186M-SSP"

   # Download the model files
   local_path = download_model(model_id)

   print(f"Model '{model_id}' downloaded successfully to: {local_path}")


*******************************
Analyzing Dataset Similarity
*******************************

When preparing experimental splits it is often useful to quantify how many items in the validation and test sets are highly similar to the training data. OmniGenBench ships a lightweight helper around the `cd-hit-est-2d` executable so you can automate this check across multiple identity thresholds.

.. note::
   Make sure `CD-HIT <https://github.com/weizhongli/cdhit>`_ is installed and the ``cd-hit-est-2d`` binary is available on your ``PATH`` before running the command below.

To generate similarity reports at 80%, 85%, 90%, and 95% identity:

.. code-block:: bash

   python -m omnigenbench.src.utility.cdhit_similarity \
       --train path/to/train.fasta \
       --validation path/to/val.fasta \
       --test path/to/test.fasta \
       --output-dir results/cdhit_similarity \
       --summary-json results/cdhit_similarity/summary.json

The utility produces, for each threshold, the raw ``cd-hit`` clustering outputs plus helper files listing which validation/test identifiers overlap with the training set and which remain unique. The optional ``--summary-json`` flag writes an aggregated report that you can load in downstream analytics pipelines.

***************
What's Next?
***************

You've now seen the basic workflows in OmniGenBench! To dive deeper, check out these resources:

*   **Command Line Interface**: See how to run benchmarking and training directly from your terminal in the :doc:`cli` guide.
*   **API Reference**: Explore all classes and functions in detail in the :doc:`api_reference`.