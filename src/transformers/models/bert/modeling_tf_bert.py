from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import dill

from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from transformers.modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from tensorflow import keras
from transformers.tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.configuration_bert import BertConfig

from transformers.models.bert.modeling_tf_bert import (
    TFBertPreTrainingLoss,
    TFBertSelfOutput,
    TFBertOutput,
    TFBertSelfAttention,
    TFBertAttention,
    TFBertIntermediate,
    TFBertLayer,
    TFBertPooler,
    TFBertPreTrainedModel,
    TFBertForQuestionAnswering,
    TFBertForPreTraining,
    TFBertForMaskedLM,
    TFBertLMHeadModel,
    TFBertForNextSentencePrediction,
    TFBertForSequenceClassification,
    TFBertForMultipleChoice,
    TFBertForTokenClassification,
    TFBertNSPHead,
    TFBertMLMHead,
)

from transformers.models.roformer.modeling_tf_roformer import (
    TFRoFormerSinusoidalPositionalEmbedding,
    TFRoFormerAttention
)

import keras_nlp


import inspect
import warnings
from typing import Dict, Optional, Union

from packaging.version import parse

from transformers.tf_utils import (
    expand_1d,
    shape_list,
)
from transformers.utils import (
    ModelOutput,
    find_labels,
    logging,
)


# try:
#     import tf_keras as keras
#     from tf_keras import backend as K
# except (ModuleNotFoundError, ImportError):
#     import keras
#     from keras import backend as K

#     if parse(keras.__version__).major > 2:
#         raise ValueError(
#             "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
#             "Transformers. Please install the backwards-compatible tf-keras package with "
#             "`pip install tf-keras`."
#         )


logger = logging.get_logger(__name__)
tf_logger = tf.get_logger()


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "ydshieh/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ydshieh/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-cased",
    "google-bert/bert-base-multilingual-uncased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-chinese",
    "google-bert/bert-base-german-cased",
    "google-bert/bert-large-uncased-whole-word-masking",
    "google-bert/bert-large-cased-whole-word-masking",
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
    "google-bert/bert-base-cased-finetuned-mrpc",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=eval(f"tf.{keras.mixed_precision.global_policy()._name.split('_')[-1]}"))


class PFFBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=self.config.hidden_dropout_prob)

        if kwargs.get('factorized_size'):
            self.factorized_size = kwargs.get('factorized_size')
            self.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.EmbedProject = keras.layers.Dense(self.config.hidden_size, name='EmbedProject')
        else:
            self.factorized_size=None

        self.position_embeddings = positional_encoding(length=4096, depth=self.config.hidden_size)

        # keras_nlp.layers.SinePositionEncoding(max_wavelength=4096, name='embeddings')


    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=(
                    [self.config.vocab_size, self.hidden_size]
                    # if not self.factorized_size 
                    # else [self.config.vocab_size, self.factorized_size]
                ),
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # with tf.name_scope("position_embeddings"):
            # # self.position_embeddings = keras_nlp.layers.SinePositionEncoding(max_wavelength=4096, name='embeddings')
            # # self.position_embeddings.trainable = False  # Freeze the layer
            
            # self.position_embeddings = self.add_weight(
            #     name="embeddings",
            #     shape=[self.max_position_embeddings, self.hidden_size],
            #     initializer=get_initializer(self.initializer_range),
            # )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        if getattr(self, "factorized_size", None) is not None:
            with tf.name_scope(self.FactorNorm.name):
                self.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.EmbedProject.name):
                self.EmbedProject.build([None, None, self.hidden_size])

    def extend_position_embeddings(self, new_size=4096):

        delta = new_size - self.config.max_position_embeddings
        self.config.max_position_embeddings = new_size

        initializer = get_initializer()
        
        current_pos = self.position_embeddings
        larger_pos = tf.concat([
            current_pos,
            initializer([delta, self.hidden_size])
        ], axis=0)

        delattr(self, 'position_embeddings')

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        self.position_embeddings = larger_pos

        return None

    def shrink(self, weights=None):
        import cupy, cuml

        to_shrink = self.weight.numpy() # if weights is None else weights
        
        condense_array = cupy.array(to_shrink, copy=False)
        dim_r = cuml.UMAP(n_neighbors=100, n_components=self.factorized_size)
        factorized_embeddings = dim_r.fit_transform(condense_array)
        
        prepared_embeddings = tf.Variable(factorized_embeddings.get())

        return prepared_embeddings

    def set_weights_and_shrink(self, weights=None, dim=128):

        if not hasattr(self, "factorized_size") or not getattr(self, "factorized_size"):

            self.factorized_size = dim
            print(f'factorized size set to {self.factorized_size}')
            
            self.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.EmbedProject = keras.layers.Dense(self.hidden_size, name='EmbedProject')
    
            with tf.name_scope(self.FactorNorm.name):
                self.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.EmbedProject.name):
                self.EmbedProject.build([None, None, self.hidden_size])

            if not weights: 
                weights = self.weight

        factorized_weights = self.shrink(weights)
        self.weight = factorized_weights

        return None

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

            if self.factorized_size:
                inputs_embeds = self.EmbedProject(self.FactorNorm(inputs_embeds))

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # if position_ids is None:
        #     position_ids = tf.expand_dims(
        #         tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
        #     )

        # sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]

        # position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        position_embeds = self.position_embeddings[tf.newaxis, :input_shape[1], :] 
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + tf.cast(position_embeds, inputs_embeds.dtype) + tf.cast(token_type_embeds, inputs_embeds.dtype)
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings      


class PFFRoFormerEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=self.config.hidden_dropout_prob)

        if kwargs.get('factorized_size'):
            self.factorized_size = kwargs.get('factorized_size')
            self.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.EmbedProject = keras.layers.Dense(self.config.hidden_size, name='EmbedProject')
        else:
            self.factorized_size=None


    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=(
                    [self.config.vocab_size, self.hidden_size]
                    if not self.factorized_size 
                    else [self.config.vocab_size, self.factorized_size]
                ),
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                 shape=(
                    [self.config.vocab_size, self.hidden_size]
                ),
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        if getattr(self, "factorized_size", None) is not None:
            with tf.name_scope(self.FactorNorm.name):
                self.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.EmbedProject.name):
                self.EmbedProject.build([None, None, self.factorized_size])

    def extend_position_embeddings(self, new_size=4096):

        delta = new_size - self.config.max_position_embeddings
        self.config.max_position_embeddings = new_size

        initializer = get_initializer()
        
        current_pos = self.position_embeddings
        larger_pos = tf.concat([
            current_pos,
            initializer([delta, self.hidden_size])
        ], axis=0)

        delattr(self, 'position_embeddings')

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        self.position_embeddings = larger_pos

        return None

    def shrink(self, weights=None):
        import cupy, cuml

        to_shrink = self.weight.numpy() if weights is None else weights
        
        condense_array = cupy.array(to_shrink, copy=False)

        print(condense_array)
        dim_r = cuml.UMAP(n_neighbors=100, n_components=getattr(self, 'factorized_size', 128))
        factorized_embeddings = dim_r.fit_transform(condense_array)
        
        prepared_embeddings = tf.Variable(factorized_embeddings.get())

        return prepared_embeddings

    def set_weights_and_shrink(self, weights=None, dim=128):

        if not weights: 
            weights = self.weight

        if not hasattr(self, "factorized_size") or not getattr(self, "factorized_size"):

            self.factorized_size = dim
            print(f'factorized size set to {self.factorized_size}')
            
            self.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.EmbedProject = keras.layers.Dense(self.hidden_size, name='EmbedProject')
    
            with tf.name_scope(self.FactorNorm.name):
                self.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.EmbedProject.name):
                self.EmbedProject.build([None, None, self.factorized_size])

        factorized_weights = self.shrink(weights)
        self.weight = factorized_weights

        return None

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

            if self.factorized_size:
                inputs_embeds = self.EmbedProject(self.FactorNorm(inputs_embeds))

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # if position_ids is None:
        #     position_ids = tf.expand_dims(
        #         tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
        #     )

        # sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]

        # position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # position_embeds = self.position_embeddings[tf.newaxis, :input_shape[1], :] 
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + tf.cast(token_type_embeds, inputs_embeds.dtype)
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings      


class PFFBertLayer(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFBertAttention(config, name="attention")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            raise NotImplementedError("Cross attention not implemented")
            # if not self.is_decoder:
            #     raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # self.crossattention = TFBertAttention(config, name="crossattention")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = (attention_output, self_attention_outputs[1:])  # add self attentions if we output attention weights

        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
        #             " by setting `config.add_cross_attention=True`"
        #         )

        #     # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        #     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        #     cross_attention_outputs = self.crossattention(
        #         input_tensor=attention_output,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         past_key_value=cross_attn_past_key_value,
        #         output_attentions=output_attentions,
        #         training=training,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        #     # add cross-attn cache to positions 3,4 of present_key_value tuple
        #     cross_attn_present_key_value = cross_attention_outputs[-1]
        #     present_key_value = present_key_value + cross_attn_present_key_value

        ########################################################################################
        # intermediate_output = self.intermediate(hidden_states=attention_output)
        # layer_output = self.bert_output(
        #     hidden_states=intermediate_output, input_tensor=attention_output, training=training
        # )
        # outputs = (layer_output,) + outputs  # add attentions if we output them
        ########################################################################################

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)


class PFFRoFormerEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFRoFormerAttention(config, name="attention")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            raise NotImplementedError("Cross attention not implemented")
            # if not self.is_decoder:
            #     raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # self.crossattention = TFBertAttention(config, name="crossattention")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
     ) -> Tuple[tf.Tensor]:
         
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )

        # attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = attention_outputs # (attention_output, self_attention_outputs[1:])  # add self attentions if we output attention weights

        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
        #             " by setting `config.add_cross_attention=True`"
        #         )

        #     # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        #     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        #     cross_attention_outputs = self.crossattention(
        #         input_tensor=attention_output,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         past_key_value=cross_attn_past_key_value,
        #         output_attentions=output_attentions,
        #         training=training,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        #     # add cross-attn cache to positions 3,4 of present_key_value tuple
        #     cross_attn_present_key_value = cross_attention_outputs[-1]
        #     present_key_value = present_key_value + cross_attn_present_key_value

        ########################################################################################
        # intermediate_output = self.intermediate(hidden_states=attention_output)
        # layer_output = self.bert_output(
        #     hidden_states=intermediate_output, input_tensor=attention_output, training=training
        # )
        # outputs = (layer_output,) + outputs  # add attentions if we output them
        ########################################################################################

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)


@tf.function
def split_with_remainder(x, max_size):
    seq = tf.shape(x)[1]
    chunked = tf.repeat(tf.constant(max_size), repeats=seq//max_size)
    l = seq % max_size
    splits = tf.concat((chunked, tf.expand_dims(l, axis=0)), axis=0)    
    return splits


class PFFBertIntermediate(TFBertIntermediate):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(config, **kwargs)

    def dense_zeros(self, hs_chunk):
        # would be interesting to avoid dense over 0s
        return None

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:

        # shape = tf.shape(hidden_states)
        splits = split_with_remainder(hidden_states, 32)
        
        hidden_states = tf.concat([
            self.dense(inputs=hs_chunk) 
            for hs_chunk in tf.split(hidden_states, splits, axis=1)
        ], axis=1)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class PFFBertOutput(TFBertOutput):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        
        # shape = tf.shape(hidden_states)
        # splits = tf.reduce_min(10, tf.reduce_max(shape[1] // 10, 1))
        splits = split_with_remainder(hidden_states, 32)

        hidden_states = tf.concat([
            self.dense(inputs=hs_chunk) 
            for hs_chunk in tf.split(hidden_states, splits, axis=1)
        ], axis=1)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class PFFBertEncoder(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # attention layers
        # self.layer = [TFBertLayer(config, name=f"layer_._0")] + [PFFBertLayer(config, name=f"layer_._{i}") for i in range(1, config.num_hidden_layers)]
        # self.layer_0 = PFFBertLayer(config, name=f"layer_._0")
        self.layer = [PFFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

        # ff, norm+add
        self.intermediate = PFFBertIntermediate(config, name="intermediate")
        self.bert_output = PFFBertOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        # Where we call each block in the blocks
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]
            
            # This is where we need to pff
            intermediate_output = self.intermediate(hidden_states=hidden_states) 
            hidden_states = self.bert_output(
                hidden_states=intermediate_output, input_tensor=hidden_states, training=training
            )

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
              self.intermediate.build(None)
              # self.intermediate.set_weights(self.layer[0].intermediate.weights)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
              self.bert_output.build(None)
              # self.bert_output.set_weights(self.layer[0].bert_output.weights)
        # if getattr(self, "layer_0", None) is not None:
        #     with tf.name_scope(self.layer_0.name):
        #       self.layer_0.build(None)
        #       self.layer_0.attention.set_weights(self.layer[0].attention.weights)
        #       del_layer = self.layer.pop(0)
        #       self.layer.insert(0, self.layer_0)
        #       delattr(self, "layer_0")
        #       del del_layer

    def expand_ff(self, desired_size=2048):
        # https://arxiv.org/pdf/2308.06103.pdf

        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        
        p_delta = desired_size - intermediate_size
        self.config.intermediate_size = desired_size

        initializer = get_initializer() # tf.keras.initializers.GlorotNormal(seed=desired_size)

        new_weights = [
            tf.concat([
                self.intermediate.weights[0],
                initializer([hidden_size, p_delta])
            ], axis=1),
            tf.concat([
                self.intermediate.weights[1],
                initializer([p_delta])
            ], axis=0)
        ]

        delattr(self, 'intermediate')
        self.intermediate = TFBertIntermediate(self.config, name="intermediate")
        with tf.name_scope(self.intermediate.name):
              self.intermediate.build(None)
            
        self.intermediate.set_weights(new_weights)

        new_weights = [
            tf.concat([
                self.bert_output.weights[0],
                initializer([p_delta, hidden_size])
            ], axis=0),
            # tf.concat([
            #     self.bert_output.weights[1],
            #     tf.zeros(p_delta)
            # ], axis=0),
            *self.bert_output.weights[1:]
        ]

        delattr(self, 'bert_output')
        self.bert_output = TFBertOutput(self.config, name="output")
        with tf.name_scope(self.bert_output.name):
              self.bert_output.build(None)
        
        self.bert_output.set_weights(new_weights)

        # what about layernorm?

        return None
    
    
    
    # A bit cleaner but eh...
    # def build(self, input_shape=None):
    #     if self.built:
    #         return
    #     self.built = True
    #     if getattr(self, "layer", None) is not None:
    #         for layer in self.layer:
    #             with tf.name_scope(layer.name):
    #                 layer.build(None)
    #     if getattr(self, "intermediate", None) is not None:
    #         # with tf.name_scope(self.intermediate.name):
    #         self.intermediate.build(None)
    #         self.intermediate.set_weights(self.layer[0].intermediate.weights)
    #     if getattr(self, "layer_0", None) is not None:
    #         with tf.name_scope(self.layer_0.name):
    #           self.layer_0.build(None)
    #           # self.layer_0.attention.set_weights(self.layer[0].attention)
    #           self.layer_0.attention.set_weights(self.layer[0].attention.weights)
    #           self.layer[0] =ncoder self.layer_0
    #           delattr(self, "layer_0")


class PFFRoFormerEncoder(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.embed_positions = TFRoFormerSinusoidalPositionalEmbedding(
            4096, # config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            name="embed_positions",
        )

        # attention layers
        # self.layer = [TFBertLayer(config, name=f"layer_._0")] + [PFFRoFormerEncoderLayer(config, name=f"layer_._{i}") for i in range(1, config.num_hidden_layers-1)]
        # self.layer_0 = PFFRoFormerEncoderLayer(config, name=f"layer_._0")
        self.layer = [PFFRoFormerEncoderLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers-1)]

        # ff, norm+add
        self.intermediate = PFFBertIntermediate(config, name="intermediate")
        self.bert_output = PFFBertOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        # Where we call each block in the blocks

        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                sinusoidal_pos=sinusoidal_pos,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]
            
            # This is where we need to pff
            intermediate_output = self.intermediate(hidden_states=hidden_states) 
            hidden_states = self.bert_output(
                hidden_states=intermediate_output, input_tensor=hidden_states, training=training
            )

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
              self.intermediate.build(None)
              # self.intermediate.set_weights(self.layer[0].intermediate.weights)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
              self.bert_output.build(None)
              # self.bert_output.set_weights(self.layer[0].bert_output.weights)
        # if getattr(self, "layer_0", None) is not None:
        #     with tf.name_scope(self.layer_0.name):
        #       self.layer_0.build(None)
        #       self.layer_0.attention.set_weights(self.layer[0].attention.weights)
        #       del_layer = self.layer.pop(0)
        #       self.layer.insert(0, self.layer_0)
        #       delattr(self, "layer_0")
        #       del del_layer

    # def rebuild(self, input_shape=None):
    #     self.built = False
    #     if getattr(self, "layer", None) is not None:
    #         for layer in self.layer:
    #             with tf.name_scope(layer.name):
    #                 layer.build(None)
    #     if getattr(self, "intermediate", None) is not None:
    #         with tf.name_scope(self.intermediate.name):
    #           self.intermediate.build(None)
    #     if getattr(self, "bert_output", None) is not None:
    #         with tf.name_scope(self.bert_output.name):
    #           self.bert_output.build(None)

    def expand_ff(self, desired_size=2048):
        # https://arxiv.org/pdf/2308.06103.pdf

        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        
        p_delta = desired_size - intermediate_size
        self.config.intermediate_size = desired_size

        initializer = get_initializer() # tf.keras.initializers.GlorotNormal(seed=desired_size)

        new_weights = [
            tf.concat([
                self.intermediate.weights[0],
                initializer([hidden_size, p_delta])
            ], axis=1),
            tf.concat([
                self.intermediate.weights[1],
                initializer([p_delta])
            ], axis=0)
        ]

        delattr(self, 'intermediate')
        self.intermediate = TFBertIntermediate(self.config, name="intermediate")
        with tf.name_scope(self.intermediate.name):
              self.intermediate.build(None)
            
        self.intermediate.set_weights(new_weights)

        new_weights = [
            tf.concat([
                self.bert_output.weights[0],
                initializer([p_delta, hidden_size])
            ], axis=0),
            # tf.concat([
            #     self.bert_output.weights[1],
            #     tf.zeros(p_delta)
            # ], axis=0),
            *self.bert_output.weights[1:]
        ]

        delattr(self, 'bert_output')
        self.bert_output = TFBertOutput(self.config, name="output")
        with tf.name_scope(self.bert_output.name):
              self.bert_output.build(None)
        
        self.bert_output.set_weights(new_weights)

        # what about layernorm?

        return None
    
    
    # A bit cleaner but eh...
    # def build(self, input_shape=None):
    #     if self.built:
    #         return
    #     self.built = True
    #     if getattr(self, "layer", None) is not None:
    #         for layer in self.layer:
    #             with tf.name_scope(layer.name):
    #                 layer.build(None)
    #     if getattr(self, "intermediate", None) is not None:
    #         # with tf.name_scope(self.intermediate.name):
    #         self.intermediate.build(None)
    #         self.intermediate.set_weights(self.layer[0].intermediate.weights)
    #     if getattr(self, "layer_0", None) is not None:
    #         with tf.name_scope(self.layer_0.name):
    #           self.layer_0.build(None)
    #           # self.layer_0.attention.set_weights(self.layer[0].attention)
    #           self.layer_0.attention.set_weights(self.layer[0].attention.weights)
    #           self.layer[0] = self.layer_0
    #           delattr(self, "layer_0")


@keras_serializable
class PFFBertMainLayer(keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        self.embeddings = PFFBertEmbeddings(config, name="embeddings")
        self.encoder = PFFBertEncoder(config, name="encoder")
        # Eh, gpu size savings
        self.pooler = tf.keras.layers.Identity() # TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if value is not None:
            self.embeddings.weight = value
            self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]

        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            training=training,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # Copied from `modeling_tf_t5.py`
        # Provided a padding mask of dimensions [batch_size, mask_seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
            if past_key_values[0] is not None:
                # attention_mask needs to be sliced to the shape `[batch_size, 1, from_seq_length - cached_seq_length, to_seq_length]
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        # Commenting or none-ing a lot just because of GPU size...
        # pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=None, # pooled_output,
            past_key_values=None, # encoder_outputs.past_key_values,
            hidden_states=None, # encoder_outputs.hidden_states,
            attentions=None, # encoder_outputs.attentions,
            cross_attentions=None, # encoder_outputs.cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
                if self.config.__dict__.get('factorized_size'):
                    factorized_weights = self.embeddings.shrink()
                    self.set_input_embeddings(factorized_weights)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


@keras_serializable
class PFFRoFormerMainLayer(keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        self.embeddings = PFFRoFormerEmbeddings(config, name="embeddings")
        self.encoder = PFFRoFormerEncoder(config, name="encoder")
        # Eh, gpu size savings
        self.pooler = tf.keras.layers.Identity() # TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if value is not None:
            self.embeddings.weight = value
            self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]

        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            training=training,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # Copied from `modeling_tf_t5.py`
        # Provided a padding mask of dimensions [batch_size, mask_seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
            if past_key_values[0] is not None:
                # attention_mask needs to be sliced to the shape `[batch_size, 1, from_seq_length - cached_seq_length, to_seq_length]
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        # Commenting or none-ing a lot just because of GPU size...
        # pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=None, # pooled_output,
            past_key_values=None, # encoder_outputs.past_key_values,
            hidden_states=None, # encoder_outputs.hidden_states,
            attentions=None, # encoder_outputs.attentions,
            cross_attentions=None, # encoder_outputs.cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
                if self.config.__dict__.get('factorized_size'):
                    # NEEDED FOR PRETRAINED
                    # Wanted for scratch?
                    # factorized_weights = self.embeddings.shrink(self.embeddings.weight)
                    # self.set_input_embeddings(factorized_weights)
                    pass
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class PFFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"


class PFFBertForPreTraining(TFBertForPreTraining): 
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForMaskedLM(TFBertForMaskedLM):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertLMHeadModel(TFBertLMHeadModel): 
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForNextSentencePrediction(TFBertForNextSentencePrediction): 
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForSequenceClassification(TFBertForSequenceClassification):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForMultipleChoice(TFBertForMultipleChoice):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForTokenClassification(TFBertForTokenClassification):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


class PFFBertForQuestionAnswering(TFBertForQuestionAnswering):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")


@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFBertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None
    attentions: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class PFFBertModel(PFFBertPreTrainedModel):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFBertMainLayer(config, name="bert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class PFFRoFormerModel(PFFBertPreTrainedModel):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = PFFRoFormerMainLayer(config, name="bert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        

class PFFBertModelForHeadless(PFFBertPreTrainedModel):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # super(tf.keras.models.Model, self).__init__()

        from transformers import AutoTokenizer

        # self.bert = PFFBertModel.from_pretrained(config, name="bert").bert
        self.bert = PFFBertMainLayer(config, name="bert")
        self.emb_predictor = tf.keras.layers.Identity()
        
        self.mask_token_id = -100
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path) # getattr(config, '_name_or_path', config.vocab_size))
        except:
            print('using bge tokenizer')
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        
        self.masker = tf.function()(keras_nlp.layers.MaskedLMMaskGenerator(
            vocabulary_size=len(self.tokenizer.vocab),
            mask_selection_rate=0.1, # 0.1 ?
            mask_token_id=self.mask_token_id,
            mask_selection_length=None # 1?
        ))


    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        mask,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Dict[int, tf.Tensor]: # Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
                
        last_hidden_state = tf.gather_nd(outputs['last_hidden_state'], mask)
        emb_prediction = self.emb_predictor(last_hidden_state)

        del outputs

        return {0: emb_prediction}

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)

    def set_weights_and_shrink(self, weights=None, dim=128):

        self.bert.set_input_embeddings(weights) if weights else None

        if not hasattr(self, "factorized_size"):

            setattr(self.bert.embeddings, "factorized_size", dim)
            print(f'factorized size set to {self.factorized_size}')
            
            self.bert.embeddings.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.bert.embeddings.EmbedProject = keras.layers.Dense(self.config.hidden_size, name='EmbedProject')
    
            with tf.name_scope(self.bert.embeddings.FactorNorm.name):
                self.bert.embeddings.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.bert.embeddings.EmbedProject.name):
                self.bert.embeddings.EmbedProject.build([None, None, self.hidden_size])

            if not weights:
                weights = self.bert.embeddings.weight

        factorized_weights = self.bert.embeddings.shrink(weights)
        # self.bert.embeddings.weight = factorized_weights
        self.bert.set_input_embeddings(factorized_weights)

        return None

    def train_step(self, inputs):
        x, y = self.make_xy(inputs)
        return super().train_step((x, y))

    # def compile(self, *args, **kwargs):
    #     super().compile(*args, **kwargs)

    def make_xy(self, inputs):
        tokens = inputs['input_ids']
        mask_dict = self.masker(tokens)
        mask = tf.where(mask_dict['token_ids'] == self.mask_token_id)

        embeddings = self.bert.get_input_embeddings()
        target_input_embeddings = tf.gather_nd(embeddings(tokens), mask)

        inputs['mask'] = mask

        input_copy = {k:tf.identity(v) for k,v in inputs.items()}

        x = {'mask': mask, **input_copy}

        return x, target_input_embeddings


class PFFRoFormerModelForHeadless(PFFBertPreTrainedModel): # , tf.keras.models.Model, self):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # super(tf.keras.models.Model, self).__init__()

        from transformers import AutoTokenizer

        # self.bert = PFFBertModel.from_pretrained(config, name="bert").bert
        setattr(config, 'rotary_value', False)
        self.bert = PFFRoFormerMainLayer(config, name="bert")
        self.emb_predictor = tf.keras.layers.Identity()
        
        self.mask_token_id = -100

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path) # getattr(config, '_name_or_path', config.vocab_size))
        except:
            print('using bge tokenizer')
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        
        self.masker = tf.function()(keras_nlp.layers.MaskedLMMaskGenerator(
            vocabulary_size=len(self.tokenizer.vocab),
            mask_selection_rate=0.1, # 0.1 ?
            mask_token_id=self.mask_token_id,
            mask_selection_length=None # 1?
        ))

        # setattr(config, 'accumulate_steps', 2048)


    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        mask,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Dict[int, tf.Tensor]: # Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
                
        last_hidden_state = tf.gather_nd(outputs['last_hidden_state'], mask)
        emb_prediction = self.emb_predictor(last_hidden_state)

        del outputs

        return {0: emb_prediction}

    def encode(self, sentences_train, batch_size=2, *args, **kwargs):
        from cytoolz.curried import pipe, partition_all, map, compose_left, valmap
        from cytoolz import concat 
        import numpy as np
        
        return pipe(
            sentences_train,
            partition_all(batch_size),
            map(compose_left(
                self.tokenizer,
                valmap(lambda v: tf.ragged.stack(v)),
                valmap(lambda v: v.to_tensor()),
                valmap(lambda v: v[:,:4096]),
                lambda bat: self.bert(**bat), 
                lambda opts: tf.reduce_mean(opts['last_hidden_state'], axis=1).numpy()
            )),
            iter,
            concat,
            partition_all(1),
            list,
            np.stack,
            np.squeeze,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)

    def set_weights_and_shrink(self, weights=None, dim=128):

        if not weights:
            weights = self.bert.embeddings.weight

        self.bert.set_input_embeddings(weights)

        if not hasattr(self, "factorized_size"):

            setattr(self.bert.embeddings, "factorized_size", dim)
            print(f'factorized size set to {self.factorized_size}')
            
            self.bert.embeddings.FactorNorm = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps, name="FactorNorm")
            self.bert.embeddings.EmbedProject = keras.layers.Dense(self.config.hidden_size, name='EmbedProject')
    
            with tf.name_scope(self.bert.embeddings.FactorNorm.name):
                self.bert.embeddings.FactorNorm.build([None, None, self.factorized_size])
            with tf.name_scope(self.bert.embeddings.EmbedProject.name):
                self.bert.embeddings.EmbedProject.build([None, None, self.factorized_size])

        factorized_weights = self.bert.embeddings.shrink(weights)
        self.bert.embeddings.weight = factorized_weights

        return None


    def test_step(self, data):
        """
        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models
        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the
        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure
        that they are available to the model during the forward pass.
        """

        x, y = self.make_xy(data)
        data = (x, y)
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer versions leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(inspect.signature(self.call).parameters)
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        if self._using_dummy_loss and "return_loss" in arg_names:
            y_pred = self(x, return_loss=True, training=False)
        else:
            y_pred = self(x, training=False)
        if self._using_dummy_loss:
            loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            loss = None

        # This next block matches outputs to label keys. Tensorflow's standard method for doing this
        # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
        if isinstance(y, dict) and len(y) == 1:
            if list(y.keys())[0] in y_pred.keys():
                y_pred = y_pred[list(y.keys())[0]]
            elif list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            _, y = y.popitem()
        elif isinstance(y, dict):
            # If the labels are a dict, match keys from the output by name
            y_pred = {key: val for key, val in y_pred.items() if key in y}
        elif isinstance(y, tuple) or isinstance(y, list):
            # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred.to_tuple()[1:]
            else:
                y_pred = y_pred.to_tuple()
            y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
        else:
            # If the labels are a single tensor, match them to the first non-loss tensor in the output
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]

        if loss is None:
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            if tf.keras.mixed_precision.global_policy()._name.startswith('mixed_'):
                loss = self.optimizer.get_scaled_loss(loss)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def train_step(self, data):

        x, y = self.make_xy(data)
        data = x,y
    
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer TF train steps leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()
    
        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")
    
        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}
    
        # Run forward pass.
        with tf.GradientTape() as tape:
            if self._using_dummy_loss and "return_loss" in arg_names:
                y_pred = self(x, training=True, return_loss=True)
            else:
                y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = None
    
            # This next block matches outputs to label keys. Tensorflow's standard method for doing this
            # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
            if isinstance(y, dict) and len(y) == 1:
                if list(y.keys())[0] in y_pred.keys():
                    y_pred = y_pred[list(y.keys())[0]]
                elif list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                _, y = y.popitem()
            elif isinstance(y, dict):
                # If the labels are a dict, match keys from the output by name
                y_pred = {key: val for key, val in y_pred.items() if key in y}
            elif isinstance(y, tuple) or isinstance(y, list):
                # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred.to_tuple()[1:]
                else:
                    y_pred = y_pred.to_tuple()
                y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
            else:
                # If the labels are a single tensor, match them to the first non-loss tensor in the output
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
    
            if loss is None:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
                if tf.keras.mixed_precision.global_policy()._name.startswith('mixed_'):
                    loss = self.optimizer.get_scaled_loss(loss)

                if not tf.is_symbolic_tensor(loss) and tf.math.is_nan(loss):
                    with open('./naughty_batch_x.dill', 'wb') as f:
                        dill.dumps(x, f)
                        
        del x

        if not tf.is_symbolic_tensor(loss) and tf.math.is_nan(loss):
            if self.gradient_accumulation:
                self.gradient_accumulation[i].assign_add(tf.math.divide(self.gradient_accumulation[i], tf.constant(self.config.accumulate_steps, dtype=tf.float32)))
            else:
                # really bad, shouldn't drop these batches but...
                pass

        else:
            grads = tape.gradient(loss, self.trainable_variables)
            if tf.keras.mixed_precision.global_policy()._name.startswith('mixed_'):
                grads = self.optimizer.get_unscaled_gradients(grads)        
        
            # # Run backwards pass.
            # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
            if not hasattr(self.config, 'accumulate_steps') or self.config.accumulate_steps < 2:
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
            else:
                self.n_acum_step.assign_add(1.)
                
                # should divide here fwiw -- maybe even by seq len sometimes
                if self.gradient_accumulation:
                    for i in range(len(self.gradient_accumulation)):
                        self.gradient_accumulation[i].assign_add(tf.math.divide(grads[i], tf.constant(self.config.accumulate_steps, dtype=tf.float32)))
                        # self.gradient_accumulation[i].assign_add(grads[i])
                        # Technically I think this should be...
                        # self.gradient_accumulation[i].assign_add(tf.math.divide(grads[i], tf.constant(self.n_ccum_step, dtype=tf.float32)))
    
                else:
                    self.gradient_accumulation = [tf.math.divide(grad, tf.constant(self.config.accumulate_steps, dtype=tf.float32)) for grad in grads] 
    
            
                # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
                # might need tf.constant(self.config.accumulate_steps)
                tf.cond(tf.equal(self.n_acum_step, tf.constant(self.config.accumulate_steps, dtype=tf.float32)), self.apply_accu_gradients, lambda: None)
    
    
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0.)
        self.gradient_accumulation = []
        # for i in range(len(self.gradient_accumulation)):
        #     self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i])) # , dtype=tf.float32))

    # def compile(self, *args, **kwargs):
    #     super().compile(*args, **kwargs)

    def make_xy(self, inputs):
        tokens = inputs['input_ids']
        mask_dict = self.masker(tokens)
        mask = tf.where(mask_dict['token_ids'] == self.mask_token_id)

        embeddings = self.bert.get_input_embeddings()
        target_input_embeddings = tf.gather_nd(embeddings(tokens), mask)

        inputs['mask'] = mask

        input_copy = {k:tf.identity(v) for k,v in inputs.items()}

        x = {'mask': mask, **input_copy}

        return x, target_input_embeddings



# @add_start_docstrings(
#     """
# Bert Model with two heads on top as done during the pretraining:
#     a `masked language modeling` head and a `next sentence prediction (classification)` head.
#     """,
#     BERT_START_DOCSTRING,
# )
# class PFFBertForPreTraining(PFFBertPreTrainedModel, TFBertPreTrainingLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [
#         r"position_ids",
#         r"cls.predictions.decoder.weight",
#         r"cls.predictions.decoder.bias",
#     ]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.bert = PFFBertMainLayer(config, name="bert")
#         self.nsp = TFBertNSPHead(config, name="nsp___cls")
#         self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

#     def get_lm_head(self) -> keras.layers.Layer:
#         return self.mlm.predictions

#     def get_prefix_bias_name(self) -> str:
#         warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
#         return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         next_sentence_label: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFBertForPreTrainingOutput, Tuple[tf.Tensor]]:
#         r"""
#         labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         next_sentence_label (`tf.Tensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
#             (see `input_ids` docstring) Indices should be in `[0, 1]`:

#             - 0 indicates sequence B is a continuation of sequence A,
#             - 1 indicates sequence B is a random sequence.
#         kwargs (`Dict[str, any]`, optional, defaults to *{}*):
#             Used to hide legacy arguments that have been deprecated.

#         Return:

#         Examples:

#         ```python
#         >>> import tensorflow as tf
#         >>> from transformers import AutoTokenizer, TFBertForPreTraining

#         >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#         >>> model = TFBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")
#         >>> input_ids = tokenizer("Hello, my dog is cute", add_special_tokens=True, return_tensors="tf")
#         >>> # Batch size 1

#         >>> outputs = model(input_ids)
#         >>> prediction_logits, seq_relationship_logits = outputs[:2]
#         ```"""
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
#         seq_relationship_score = self.nsp(pooled_output=pooled_output)
#         total_loss = None

#         if labels is not None and next_sentence_label is not None:
#             d_labels = {"labels": labels}
#             d_labels["next_sentence_label"] = next_sentence_label
#             total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))

#         if not return_dict:
#             output = (prediction_scores, seq_relationship_score) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return TFBertForPreTrainingOutput(
#             loss=total_loss,
#             prediction_logits=prediction_scores,
#             seq_relationship_logits=seq_relationship_score,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "nsp", None) is not None:
#             with tf.name_scope(self.nsp.name):
#                 self.nsp.build(None)
#         if getattr(self, "mlm", None) is not None:
#             with tf.name_scope(self.mlm.name):
#                 self.mlm.build(None)


# @add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
# class PFFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [
#         r"pooler",
#         r"cls.seq_relationship",
#         r"cls.predictions.decoder.weight",
#         r"nsp___cls",
#     ]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         if config.is_decoder:
#             logger.warning(
#                 "If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for "
#                 "bi-directional self-attention."
#             )

#         self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")
#         self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

#     def get_lm_head(self) -> keras.layers.Layer:
#         return self.mlm.predictions

#     def get_prefix_bias_name(self) -> str:
#         warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
#         return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=TFMaskedLMOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output="'paris'",
#         expected_loss=0.88,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
#         r"""
#         labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         sequence_output = outputs[0]
#         prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
#         loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFMaskedLMOutput(
#             loss=loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "mlm", None) is not None:
#             with tf.name_scope(self.mlm.name):
#                 self.mlm.build(None)


# class PFFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [
#         r"pooler",
#         r"cls.seq_relationship",
#         r"cls.predictions.decoder.weight",
#         r"nsp___cls",
#     ]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         if not config.is_decoder:
#             logger.warning("If you want to use `TFBertLMHeadModel` as a standalone, add `is_decoder=True.`")

#         self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")
#         self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

#     def get_lm_head(self) -> keras.layers.Layer:
#         return self.mlm.predictions

#     def get_prefix_bias_name(self) -> str:
#         warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
#         return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
#         input_shape = input_ids.shape
#         # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
#         if attention_mask is None:
#             attention_mask = tf.ones(input_shape)

#         # cut decoder_input_ids if past is used
#         if past_key_values is not None:
#             input_ids = input_ids[:, -1:]

#         return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

#     @unpack_inputs
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=TFCausalLMOutputWithCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
#         encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
#         past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#         **kwargs,
#     ) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
#         r"""
#         encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#         past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
#             contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`). Set to `False` during training, `True` during generation
#         labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
#             config.vocab_size - 1]`.
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         sequence_output = outputs[0]
#         logits = self.mlm(sequence_output=sequence_output, training=training)
#         loss = None

#         if labels is not None:
#             # shift labels to the left and cut last logit token
#             shifted_logits = logits[:, :-1]
#             labels = labels[:, 1:]
#             loss = self.hf_compute_loss(labels=labels, logits=shifted_logits)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFCausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "mlm", None) is not None:
#             with tf.name_scope(self.mlm.name):
#                 self.mlm.build(None)


# @add_start_docstrings(
#     """Bert Model with a `next sentence prediction (classification)` head on top.""",
#     BERT_START_DOCSTRING,
# )
# class PFFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"cls.predictions"]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.bert = PFFBertMainLayer(config, name="bert")
#         self.nsp = TFBertNSPHead(config, name="nsp___cls")

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         next_sentence_label: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
#         r"""
#         Return:

#         Examples:

#         ```python
#         >>> import tensorflow as tf
#         >>> from transformers import AutoTokenizer, TFBertForNextSentencePrediction

#         >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#         >>> model = TFBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

#         >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
#         >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
#         >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

#         >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
#         >>> assert logits[0][0] < logits[0][1]  # the next sentence was random
#         ```"""
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         pooled_output = outputs[1]
#         seq_relationship_scores = self.nsp(pooled_output=pooled_output)
#         next_sentence_loss = (
#             None
#             if next_sentence_label is None
#             else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
#         )

#         if not return_dict:
#             output = (seq_relationship_scores,) + outputs[2:]
#             return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

#         return TFNextSentencePredictorOutput(
#             loss=next_sentence_loss,
#             logits=seq_relationship_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "nsp", None) is not None:
#             with tf.name_scope(self.nsp.name):
#                 self.nsp.build(None)


# @add_start_docstrings(
#     """
#     Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
#     output) e.g. for GLUE tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class PFFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
#     _keys_to_ignore_on_load_missing = [r"dropout"]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.num_labels = config.num_labels

#         self.bert = PFFBertMainLayer(config, name="bert")
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = keras.layers.Dropout(rate=classifier_dropout)
#         self.classifier = keras.layers.Dense(
#             units=config.num_labels,
#             kernel_initializer=get_initializer(config.initializer_range),
#             name="classifier",
#         )
#         self.config = config

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
#         output_type=TFSequenceClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
#         expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
#         r"""
#         labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(inputs=pooled_output, training=training)
#         logits = self.classifier(inputs=pooled_output)
#         loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFSequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)print(
#         if getattr(self, "classifier", None) is not None:
#             with tf.name_scope(self.classifier.name):
#                 self.classifier.build([None, None, self.config.hidden_size])


# @add_start_docstrings(
#     """
#     Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
#     softmax) e.g. for RocStories/SWAG tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class PFFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
#     _keys_to_ignore_on_load_missing = [r"dropout"]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.bert = PFFBertMainLayer(config, name="bert")
#         self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
#         self.classifier = keras.layers.Dense(
#             units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
#         )
#         self.config = config

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=TFMultipleChoiceModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
#         r"""
#         labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
#             Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
#             where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
#         """
#         if input_ids is not None:
#             num_choices = shape_list(input_ids)[1]
#             seq_length = shape_list(input_ids)[2]
#         else:
#             num_choices = shape_list(inputs_embeds)[1]
#             seq_length = shape_list(inputs_embeds)[2]

#         flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
#         flat_attention_mask = (
#             tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
#         )
#         flat_token_type_ids = (
#             tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
#         )
#         flat_position_ids = (
#             tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
#         )
#         flat_inputs_embeds = (
#             tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
#             if inputs_embeds is not None
#             else None
#         )
#         outputs = self.bert(
#             input_ids=flat_input_ids,
#             attention_mask=flat_attention_mask,
#             token_type_ids=flat_token_type_ids,
#             position_ids=flat_position_ids,
#             head_mask=head_mask,
#             inputs_embeds=flat_inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(inputs=pooled_output, training=training)
#         logits = self.classifier(inputs=pooled_output)
#         reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
#         loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

#         if not return_dict:
#             output = (reshaped_logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFMultipleChoiceModelOutput(
#             loss=loss,
#             logits=reshaped_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "classifier", None) is not None:
#             with tf.name_scope(self.classifier.name):
#                 self.classifier.build([None, None, self.config.hidden_size])


# @add_start_docstrings(
#     """
#     Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
#     Named-Entity-Recognition (NER) tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class PFFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [
#         r"pooler",
#         r"mlm___cls",
#         r"nsp___cls",
#         r"cls.predictions",
#         r"cls.seq_relationship",
#     ]
#     _keys_to_ignore_on_load_missing = [r"dropout"]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.num_labels = config.num_labels

#         self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = keras.layers.Dropout(rate=classifier_dropout)
#         self.classifier = keras.layers.Dense(
#             units=config.num_labels,
#             kernel_initializer=get_initializer(config.initializer_range),
#             name="classifier",
#         )
#         self.config = config

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
#         output_type=TFTokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
#         expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
#         r"""
#         labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(inputs=sequence_output, training=training)
#         logits = self.classifier(inputs=sequence_output)
#         loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFTokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "classifier", None) is not None:
#             with tf.name_scope(self.classifier.name):
#                 self.classifier.build([None, None, self.config.hidden_size])


# @add_start_docstrings(
#     """
#     Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     BERT_START_DOCSTRING,
# )
# class PFFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
#     # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
#     _keys_to_ignore_on_load_unexpected = [
#         r"pooler",
#         r"mlm___cls",
#         r"nsp___cls",
#         r"cls.predictions",
#         r"cls.seq_relationship",
#     ]

#     def __init__(self, config: BertConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)

#         self.num_labels = config.num_labels

#         self.bert = PFFBertMainLayer(config, add_pooling_layer=False, name="bert")
#         self.qa_outputs = keras.layers.Dense(
#             units=config.num_labels,
#             kernel_initializer=get_initializer(config.initializer_range),
#             name="qa_outputs",
#         )
#         self.config = config

#     @unpack_inputs
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_QA,
#         output_type=TFQuestionAnsweringModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#         qa_target_start_index=_QA_TARGET_START_INDEX,
#         qa_target_end_index=_QA_TARGET_END_INDEX,
#         expected_output=_QA_EXPECTED_OUTPUT,
#         expected_loss=_QA_EXPECTED_LOSS,
#     )
#     def call(
#         self,
#         input_ids: TFModelInputType | None = None,
#         attention_mask: np.ndarray | tf.Tensor | None = None,
#         token_type_ids: np.ndarray | tf.Tensor | None = None,
#         position_ids: np.ndarray | tf.Tensor | None = None,
#         head_mask: np.ndarray | tf.Tensor | None = None,
#         inputs_embeds: np.ndarray | tf.Tensor | None = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         start_positions: np.ndarray | tf.Tensor | None = None,
#         end_positions: np.ndarray | tf.Tensor | None = None,
#         training: Optional[bool] = False,
#     ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
#         r"""
#         start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             training=training,
#         )
#         sequence_output = outputs[0]
#         logits = self.qa_outputs(inputs=sequence_output)
#         start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
#         start_logits = tf.squeeze(input=start_logits, axis=-1)
#         end_logits = tf.squeeze(input=end_logits, axis=-1)
#         loss = None

#         if start_positions is not None and end_positions is not None:
#             labels = {"start_position": start_positions}
#             labels["end_position"] = end_positions
#             loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TFQuestionAnsweringModelOutput(
#             loss=loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "bert", None) is not None:
#             with tf.name_scope(self.bert.name):
#                 self.bert.build(None)
#         if getattr(self, "qa_outputs", None) is not None:
#             with tf.name_scope(self.qa_outputs.name):
#                 self.qa_outputs.build([None, None, self.config.hidden_size])


# class TFBertPredictionHeadTransform(keras.layers.Layer):
#     def __init__(self, config: BertConfig, **kwargs):
#         super().__init__(**kwargs)

#         self.dense = keras.layers.Dense(
#             units=config.hidden_size,
#             kernel_initializer=get_initializer(config.initializer_range),
#             name="dense",
#         )

#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = get_tf_activation(config.hidden_act)
#         else:
#             self.transform_act_fn = config.hidden_act

#         self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
#         self.config = config

#     def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
#         hidden_states = self.dense(inputs=hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(inputs=hidden_states)

#         return hidden_states

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "dense", None) is not None:
#             with tf.name_scope(self.dense.name):
#                 self.dense.build([None, None, self.config.hidden_size])
#         if getattr(self, "LayerNorm", None) is not None:
#             with tf.name_scope(self.LayerNorm.name):
#                 self.LayerNorm.build([None, None, self.config.hidden_size])


# class TFBertLMPredictionHead(keras.layers.Layer):
#     def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs):
#         super().__init__(**kwargs)

#         self.config = config
#         self.hidden_size = config.hidden_size

#         self.transform = TFBertPredictionHeadTransform(config, name="transform")

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.input_embeddings = input_embeddings

#     def build(self, input_shape=None):
#         self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "transform", None) is not None:
#             with tf.name_scope(self.transform.name):
#                 self.transform.build(None)

#     def get_output_embeddings(self) -> keras.layers.Layer:
#         return self.input_embeddings

#     def set_output_embeddings(self, value: tf.Variable):
#         self.input_embeddings.weight = value
#         self.input_embeddings.vocab_size = shape_list(value)[0]

#     def get_bias(self) -> Dict[str, tf.Variable]:
#         return {"bias": self.bias}

#     def set_bias(self, value: tf.Variable):
#         self.bias = value["bias"]
#         self.config.vocab_size = shape_list(value["bias"])[0]

#     def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
#         hidden_states = self.transform(hidden_states=hidden_states)
#         seq_length = shape_list(hidden_states)[1]
#         hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
#         hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
#         hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
#         hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

#         return hidden_states


# class TFBertMLMHead(keras.layers.Layer):
#     def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs):
#         super().__init__(**kwargs)

#         self.predictions = TFBertLMPredictionHead(config, input_embeddings, name="predictions")

#     def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
#         prediction_scores = self.predictions(hidden_states=sequence_output)

#         return prediction_scores

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "predictions", None) is not None:
#             with tf.name_scope(self.predictions.name):
#                 self.predictions.build(None)


# class TFBertNSPHead(keras.layers.Layer):
#     def __init__(self, config: BertConfig, **kwargs):
#         super().__init__(**kwargs)

#         self.seq_relationship = keras.layers.Dense(
#             units=2,
#             kernel_initializer=get_initializer(config.initializer_range),
#             name="seq_relationship",
#         )
#         self.config = config

#     def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
#         seq_relationship_score = self.seq_relationship(inputs=pooled_output)

#         return seq_relationship_score

#     def build(self, input_shape=None):
#         if self.built:
#             return
#         self.built = True
#         if getattr(self, "seq_relationship", None) is not None:
#             with tf.name_scope(self.seq_relationship.name):
#                 self.seq_relationship.build([None, None, self.config.hidden_size])



@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: tf.constant,
        value_states: tf.constant,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[tf.constant, tf.constant]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`tf.constant`):
                The new key states to cache.
            value_states (`tf.constant`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = 1_000_000 # self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length



class TFGemmaAttention(keras.layers.Layer):
    def __init__(
        self, 
        config: GemmaConfig, 
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = keras.layers.Dense(self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        self.embed_positions = TFRoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            name="embed_positions",
        )

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def forward(
        self,
        hidden_states: tf.constant,
        attention_mask: Optional[tf.constant] = None,
        position_ids: Optional[tf.constant] = None,
        past_key_value: Optional[Cache] = None,
        output_fs: bool = False,
        use_cache: bool = False,
        cache_position: Optional[tf.constant] = None,
        output_attentions: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tuple[tf.constant, Optional[tf.constant], Optional[Tuple[tf.constant]]]:
        input_shape = shape_list(hidden_states)
        bsz, q_len, _ = input_shape[0], input_shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.transpose_for_scores(self.query(inputs=hidden_states), bsz)
        key_states = self.transpose_for_scores(self.key(inputs=hidden_states), bsz)
        value_states = self.transpose_for_scores(self.value(inputs=hidden_states), bsz)

        # CHECK ALL OF THESE
        ###################################################################################################
        # Copied from transformers.models.llama.modeling_llama.repeat_kv
        def repeat_kv(hidden_states: tf.constant, n_rep: int) -> tf.constant:
            """
            This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
            num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
            """
            batch, num_key_value_heads, slen, head_dim = shape_list(hidden_states)
            if n_rep == 1:
                return hidden_states
            hidden_states = tf.broadcast_to(hidden_states[:, :, None, :, :], shape=(batch, num_key_value_heads, n_rep, slen, head_dim))
            return tf.reshape(hidden_states, shape=(batch, num_key_value_heads * n_rep, slen, head_dim))
        
        past_key_value = getattr(self, "past_key_value", past_key_value)

        sin, cos = sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, sinusoidal_pos, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #####################################################################################################

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        
        ########### check ########################
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            if cache_position is not None:
                casual_mask = attention_mask[:, :, cache_position, shape_list(key_states)[-2]]
            else:
                causal_mask = attention_mask
            attention_scores = tf.add(attention_scores + causal_mask)
        ###########################################

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        
        # # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_states)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(bsz, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        # if self.is_decoder:
        #     outputs = outputs + (past_key_value,)
        # return outputs

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.view(bsz, q_len, -1)
        # attn_output = self.o_proj(attn_output)

        return outputs, past_key_value
    
    @staticmethod
    def apply_rotary_position_embeddings(query_layer, key_layer, sinusoidal_pos, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = tf.split(sinusoidal_pos, num_or_size_splits=2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1]-> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        # cos [θ0,θ1,θ2......θd/2-1]-> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = tf.repeat(sin, 2, axis=-1)
        cos_pos = tf.repeat(cos, 2, axis=-1)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
        rotate_half_query_layer = tf.reshape(rotate_half_query_layer, shape_list(query_layer))
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = tf.stack([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
        rotate_half_key_layer = tf.reshape(rotate_half_key_layer, shape_list(key_layer))
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = tf.stack([-value_layer[..., 1::2], value_layer[..., ::2]], axis=-1)
            rotate_half_value_layer = tf.reshape(rotate_half_value_layer, shape_list(value_layer))
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer