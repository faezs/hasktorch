{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased where

import Data.Singletons (SingI (sing))
import Data.Singletons.TypeLits (SNat)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Transformer.BERT.Common (BERTModel, BERTModelSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | BERT-Base-Uncased number of layers.
-- 'num_hidden_layers = 12'
type BERTBaseUncasedNumLayers = 12

-- | BERT-Base-Uncased number of layers singleton.
bertBaseUncasedNumLayers :: SNat BERTBaseUncasedNumLayers
bertBaseUncasedNumLayers = sing

-- | BERT-Base-Uncased number of attention heads.
-- 'num_attention_heads = 12'
type BERTBaseUncasedHeadDim = 'Dim ('Name "*") ('Size 12)

-- | BERT-Base-Uncased number of attention heads singleton.
bertBaseUncasedHeadDim :: SDim BERTBaseUncasedHeadDim
bertBaseUncasedHeadDim = sing

-- | BERT-Base-Uncased head embedding dimension.
-- 'd_kv = 64'
type BERTBaseUncasedHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BERT-Base-Uncased head embedding dimension singleton.
bertBaseUncasedHeadEmbedDim :: SDim BERTBaseUncasedHeadEmbedDim
bertBaseUncasedHeadEmbedDim = sing

-- | BERT-Base-Uncased embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type BERTBaseUncasedEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased embedding dimension singleton.
bertBaseUncasedEmbedDim :: SDim BERTBaseUncasedEmbedDim
bertBaseUncasedEmbedDim = sing

-- | BERT-Base-Uncased model dimension.
-- 'hidden_size = 768'
type BERTBaseUncasedInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased model dimension singleton.
bertBaseUncasedInputEmbedDim :: SDim BERTBaseUncasedInputEmbedDim
bertBaseUncasedInputEmbedDim = sing

-- | BERT-Base-Uncased feed-forward network dimension.
-- 'intermediate_size = 3072'
type BERTBaseUncasedFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | BERT-Base-Uncased feed-forward network dimension singleton.
bertBaseUncasedFFNDim :: SDim BERTBaseUncasedFFNDim
bertBaseUncasedFFNDim = sing

-- | BERT-Base-Uncased vocabulary dimension.
-- 'vocab_size = 30522'
type BERTBaseUncasedVocabDim = 'Dim ('Name "*") ('Size 30522)

-- | BERT-Base-Uncased vocabulary dimension singleton.
bertBaseUncasedVocabDim :: SDim BERTBaseUncasedVocabDim
bertBaseUncasedVocabDim = sing

-- | BERT-Base-Uncased type vocabulary dimension.
-- 'type_vocab_size = 2'
type BERTBaseUncasedTypeVocabDim = 'Dim ('Name "*") ('Size 2)

-- | BERT-Base-Uncased type vocabulary dimension singleton.
bertBaseUncasedTypeVocabDim :: SDim BERTBaseUncasedTypeVocabDim
bertBaseUncasedTypeVocabDim = sing

-- | BERT-Base-Uncased model.
type BERTBaseUncased
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat)) =
  BERTModel transformerHead BERTBaseUncasedNumLayers gradient device BERTBaseUncasedHeadDim BERTBaseUncasedHeadEmbedDim BERTBaseUncasedEmbedDim BERTBaseUncasedInputEmbedDim BERTBaseUncasedFFNDim BERTBaseUncasedVocabDim BERTBaseUncasedTypeVocabDim

bertBaseUnchasedSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  BERTModelSpec transformerHead BERTBaseUncasedNumLayers gradient device BERTBaseUncasedHeadDim BERTBaseUncasedHeadEmbedDim BERTBaseUncasedEmbedDim BERTBaseUncasedInputEmbedDim BERTBaseUncasedFFNDim BERTBaseUncasedVocabDim BERTBaseUncasedTypeVocabDim
bertBaseUnchasedSpec transformerHead = BERTModelSpec transformerHead bertBaseUncasedNumLayers
