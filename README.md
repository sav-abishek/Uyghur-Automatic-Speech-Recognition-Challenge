# Uyghur Automatic Speech Recognition (ASR) System – NPPE-2 Challenge

## Overview

This repository contains the implementation of an **end-to-end Automatic Speech Recognition (ASR) pipeline** developed for the **NPPE-2 Uyghur ASR Challenge**. The objective is to build a high-performance speech-to-text system capable of accurately transcribing Uyghur speech, a **low-resource and underrepresented language**, using modern deep learning based acoustic and language modeling techniques.

The system is designed to convert raw audio waveforms into textual transcriptions by learning robust acoustic representations, aligning phonetic patterns to character-level outputs, and optimizing sequence decoding under the Character Error Rate (CER) objective.

The challenge dataset consists of **23+ hours of labeled Uyghur speech**, and the competition explicitly allows the use of:

* Large pretrained models (e.g., Whisper, Wav2Vec2, HuBERT)
* Self-supervised learning approaches
* External corpora and transfer learning
* Any compute or architectural innovation

This project treats ASR as a **sequence-to-sequence learning problem** under low-resource constraints and focuses on maximizing transcription fidelity while minimizing character-level edit distance.

---

## Problem Formulation

The task is framed as a **supervised sequence transduction problem**:

[
X = {x_1, x_2, \dots, x_T} \rightarrow Y = {y_1, y_2, \dots, y_N}
]

Where:

* ( X ) represents the acoustic feature sequence extracted from the audio waveform
* ( Y ) represents the target Uyghur character sequence
* ( T ) is the number of acoustic frames
* ( N ) is the number of output characters

The model learns the conditional distribution:

[
P(Y \mid X)
]

and is optimized to minimize transcription errors at the **character level**, measured by **Character Error Rate (CER)**.

---

## Evaluation Metric: Character Error Rate (CER)

The system is evaluated using **Character Error Rate (CER)**, defined as:

[
CER = \frac{S + D + I}{N}
]

Where:

* **S** = number of substitutions
* **D** = number of deletions
* **I** = number of insertions
* **N** = total number of characters in the ground truth

CER is computed using **Levenshtein distance** and directly reflects the model’s ability to produce accurate character-level transcriptions. Lower CER indicates better performance, with **0.0 representing a perfect transcription**.

This metric is particularly important for Uyghur due to:

* Agglutinative morphology
* Rich character composition
* High sensitivity to small transcription errors

---

## System Architecture

The ASR pipeline is structured into the following logical stages:

---

### 1. Audio Ingestion and Preprocessing

Raw audio waveforms are:

* Loaded from disk
* Resampled to a standardized sampling rate (typically 16 kHz)
* Normalized for amplitude stability
* Optionally trimmed or padded for batch consistency

This stage ensures that all inputs conform to a consistent acoustic distribution before feature extraction.

---

### 2. Acoustic Feature Extraction

Depending on the chosen backbone, the system extracts representations using one of:

#### a) Log-Mel Spectrograms (Classical Pipeline)

* Short-Time Fourier Transform (STFT)
* Mel filterbanks
* Log compression
* Time–frequency normalization

#### b) Learned Representations (Self-Supervised Models)

* Wav2Vec2 / HuBERT style convolutional feature encoders
* Contextual transformer blocks
* Quantized latent representations

These representations act as high-dimensional embeddings capturing phonetic, prosodic, and temporal structure in speech.

---

### 3. Encoder Network (Acoustic Model)

The encoder transforms acoustic features into higher-level latent representations.

Depending on the approach, this may include:

* **Convolutional layers** for local temporal feature extraction
* **Transformer encoders** for long-range dependency modeling
* **Self-attention mechanisms** to capture contextual phoneme relationships

For pretrained models, this stage leverages large-scale self-supervised pretraining on multilingual speech corpora, enabling effective transfer learning to Uyghur despite limited labeled data.

---

### 4. Decoder / Prediction Head

The decoder maps latent acoustic representations to character sequences using one of:

* **CTC (Connectionist Temporal Classification) heads**
* **Attention-based sequence decoders**
* **Transformer decoders (seq2seq)**

This stage learns the alignment between acoustic frames and output characters, handling:

* Variable-length sequences
* Temporal compression
* Non-monotonic phoneme-to-character mapping

---

### 5. Language Modeling and Decoding

During inference, decoding may use:

* **Greedy decoding**
* **Beam search**
* **Language model fusion (optional)**

The goal is to maximize:

[
\arg\max_Y P(Y \mid X)
]

under decoding constraints while minimizing insertions, deletions, and substitutions.

---

## Low-Resource Learning Strategy

Uyghur is a low-resource language, which introduces challenges such as:

* Limited labeled speech data
* High phonetic and morphological variability
* Scarce external corpora

To mitigate this, the system is designed around:

### Transfer Learning

* Fine-tuning large pretrained speech models (e.g., Whisper, Wav2Vec2, HuBERT)
* Leveraging multilingual representations learned from high-resource languages

### Self-Supervised Pretraining

* Using contrastive or masked prediction objectives
* Learning speech representations without reliance on labeled data

### Regularization and Augmentation

* SpecAugment (time masking, frequency masking)
* Noise injection
* Speed perturbation

These techniques improve generalization and prevent overfitting on small datasets.

---

## Training Objective

The training process minimizes a sequence loss function, typically:

* **CTC Loss** for alignment-free training
* **Cross-Entropy Loss** for attention-based decoders

The optimization objective is to minimize transcription error under the CER metric, indirectly reducing Levenshtein distance between predictions and ground truth.

---

## Submission Pipeline

The final system outputs predictions in the required CSV format:

```
ID,transcription
<uuid_1>,<predicted_text_1>
<uuid_2>,<predicted_text_2>
...
```

Where:

* **ID** corresponds to the audio sample identifier from `test.csv`
* **transcription** is the model-generated Uyghur text output

The pipeline includes:

* Batch inference over test audio files
* Post-processing and normalization of predicted text
* CSV serialization for submission

---

## Design Philosophy

This project is structured as a **research-grade ASR pipeline**, emphasizing:

* Modular architecture (preprocessing, encoder, decoder, decoding)
* Replaceable acoustic backbones
* Transfer learning and self-supervised learning
* Low-resource robustness
* Metric-driven optimization (CER-focused)

The system is intentionally designed to be extensible, allowing:

* Swapping encoders (CNN, Transformer, Conformer)
* Integrating language models
* Experimenting with different decoding strategies

---

## Why This Matters

Building ASR systems for low-resource languages like Uyghur is significantly more challenging than for English or Mandarin due to:

* Limited labeled data
* Linguistic complexity
* Lack of pretrained resources

This project demonstrates the application of **modern deep learning, representation learning, and transfer learning techniques** to push performance boundaries in a constrained data regime, aligning with real-world ASR research problems.

---

## Summary

This repository implements a **deep learning based Automatic Speech Recognition system** for Uyghur that:

* Treats ASR as a sequence-to-sequence learning problem
* Leverages self-supervised and pretrained speech models
* Optimizes directly for Character Error Rate (CER)
* Is designed for low-resource language robustness
* Supports end-to-end training, inference, and submission workflows

The project reflects a **production-grade ASR engineering mindset** combined with **research-level modeling strategies**, making it suitable for both academic and industry-facing portfolios.

---

Kaggle Competition Link: https://www.kaggle.com/competitions/the-uyghur-voice-cup/overview
