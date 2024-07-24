Welcome to InternVL's tutorials!
====================================

.. figure:: ./_static/image/internvl-logo.svg
  :width: 50%
  :align: center
  :alt: InternVL
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LMDeploy is a toolkit for compressing, deploying, and serving LLM.
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/OpenGVLab/InternVL" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/OpenGVLab/InternVL/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/OpenGVLab/InternVL/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

LMDeploy has the following core features:

* **Efficient Inference**: LMDeploy delivers up to 1.8x higher request throughput than vLLM, by introducing key features like persistent batch(a.k.a. continuous batching), blocked KV cache, dynamic split&fuse, tensor parallelism, high-performance CUDA kernels and so on.

* **Effective Quantization**: LMDeploy supports weight-only and k/v quantization, and the 4-bit inference performance is 2.4x higher than FP16. The quantization quality has been confirmed via OpenCompass evaluation.

* **Effortless Distribution Server**: Leveraging the request distribution service, LMDeploy facilitates an easy and efficient deployment of multi-model services across multiple machines and cards.

* **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, the engine remembers dialogue history, thus avoiding repetitive processing of historical sessions.

* **Excellent Compatibility**: LMDeploy supports `KV Cache Quant <https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html>`_, `AWQ <https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html>`_ and `Automatic Prefix Caching <https://lmdeploy.readthedocs.io/en/latest/inference/turbomind_config.html>`_ to be used simultaneously.

Documentation
-------------

.. _get_started:
.. toctree::
   :maxdepth: 2
   :caption: Get Started

   get_started.md

.. _internvl_chat:
.. toctree::
   :maxdepth: 1
   :caption: InternVL-Chat

   build.md

.. _internvl_chat_llava:
.. toctree::
   :maxdepth: 1
   :caption: InternVL-Chat-LLaVA

   build.md

.. _classification:
.. toctree::
   :maxdepth: 1
   :caption: Classification

   build.md

.. _clip_benchmark:
.. toctree::
   :maxdepth: 1
   :caption: CLIP Benchmark

   build.md

.. _internvl_g:
.. toctree::
   :maxdepth: 1
   :caption: InternVL-G

   build.md

.. _segmentation:
.. toctree::
   :maxdepth: 2
   :caption: InternVL 1.0
   :titlesonly:

   classification <internvl1.0/classification.md>
   clip_benchmark <internvl1.0/clip_benchmark.md>
   internvl_chat_llava <internvl1.0/internvl_chat_llava.md>
   internvl_g <internvl1.0/internvl_g.md>
   segmentation <internvl1.0/segmentation.md>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
