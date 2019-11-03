{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module Main where

import           Prelude                 hiding ( (.), id, tanh )
import           Control.Arrow
import           Control.Category
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import           Torch.Typed
import           Torch.Typed.Native     hiding ( linear )
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D

{---

I want a spacetime graph

Graph Construction:
Use algebraic-graphs to assemble a test graph
1. Node Features
2. Edge Features

Have functions that give you:
1) outgoing :: Node -> [Node]
2) incoming :: Node -> [Node]

---}


data NodeType = Storage { batteryVoltage :: Double }
              | Generation { produced :: Double }
              | Consumption { consumed :: Double, demand :: Double }
              deriving (Eq, Ord)

data EdgeType = Send | Recieve deriving (Eq, Ord)


-- SHouldn't this be an alga instance implementation for Graph.Labelled 
class DistControl a where
  type Obs a
  type Act a
  toObservationVector :: a -> Obs a
  toControlVector :: Act a -> a

{--
Nervenet modules:

Policy Network :: Observation@T -> Policy@T
1. input model :: Embedding MLP (x_u -> h0_u)
2. propagation model :: [h0_u] -> [ht_u]
3. output model :: [ht_u] -> [(mu_u, sigma)]

Value Network :: Observation@T -> Value@T
1. input model :: Embedding MLP (x_u -> h0_u)
2. propagation model :: [h0_u] -> [ht_u]
3. output model :: [ht_u] -> (V :: R)

Learning Algorithm:
1. Advantage Estimate
2. PPO Loss Construction
3. Policy Gradient

--}

data Activation (dtype :: D.DType)
  where
    Activation :: forall dtype . { unActivation :: forall shape. Tensor dtype shape -> Tensor dtype shape }
               -> Activation dtype

instance Show (Activation dtype) where
  show _ = mempty

instance A.Parameterized (Activation dtype) where
  flattenParameters _ = []
  replaceOwnParameters = return


data MLPSpec (dtype :: D.DType)
             (inputFeatures :: Nat) (outputFeatures :: Nat)
             (hiddenFeatures0 :: Nat) (hiddenFeatures1 :: Nat)
  where
    MLPSpec
      :: forall dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 . { mlpDropoutProbSpec :: Double }
      -> MLPSpec dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1
  deriving (Eq, Show)


data MLP (dtype :: D.DType)
     (inputFeatures :: Nat) (outputFeatures :: Nat)
     (hiddenFeatures0 :: Nat) (hiddenFeatures1 :: Nat)
     where
       MLP
         :: forall dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1
         . { mlpLayer0 :: Linear dtype inputFeatures hiddenFeatures0
           , mlpLayer1 :: Linear dtype hiddenFeatures0 hiddenFeatures1
           , mlpLayer2 :: Linear dtype hiddenFeatures1 outputFeatures
           , mlpDropout :: Dropout
           }
         -> MLP dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1
  deriving (Show, Generic)


mlp ::
  MLP dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1
  -> Activation dtype
  -> Bool
  -> Tensor dtype '[batchSize, inputFeatures]
  -> IO (Tensor dtype '[batchSize, outputFeatures])
mlp MLP {..} act input train = do
  return
    linear mlpLayer2
    =<< layer mlpLayer1 act
    =<< layer mlpLayer0 act 
    =<< pure input
    where
      layer :: Linear dtype b c -> Activation dtype -> cat0 a0 c0
      layer lparams act =
        (Torch.Typed.NN.dropout mlpDropout train)
        . act
        . linear lparams

instance A.Parameterized (MLP dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1)


instance ( KnownDType dtype
         , KnownNat inputFeatures
         , KnownNat outputFeatures
         , KnownNat hiddenFeatures0
         , KnownNat hiddenFeatures1
         )
         => A.Randomizable (MLPSpec dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1)
                           (MLP dtype inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1)
         where
           sample MLPSpec {..} =
             MLP
             <$> A.sample LinearSpec
             <*> A.sample LinearSpec
             <*> A.sample LinearSpec
             <*> A.sample (DropoutSpec mlpDropoutProbSpec)


type NodeObsDim = 5
type StateDim = 32
type InputEmbedderLDim = 128

{--embedInput :: 
embedInput obs = do
  let dropoutProb = 0.5
  init <- A.sample (MLPSpec @D.Float @NodeObsDim @StateDim @InputEmbedderLDim @InputEmbedderLDim dropoutProb)
  init' <- A.replaceParameters init <$> traverse
    (A.makeIndependent . toBackend backend . A.toDependent)
    (A.flattenParameters init)
  

data InputEmbedderSpec :: (dtype :: D.DType) (obsDim :: Nat) (stateDim :: Nat) (layers :: Nat) (lDim :: Nat)
  where
    InputEmbedderSpec
      :: forall dtype obsDim stateDim layers lDim . { mlpDropoutProbSpec :: Double }
      -> InputEmbedderSpec dtype obsDim stateDim layers lDim deriving (Eq, Show)
--}
{--

IMPORTANT:
All the networks are mapped over each node.
Let's start with MLPs

inputEmbedder :: MLP :: obsDim -> hDim

propagation models:
1. message Fn :: [MLP_edge_type ht_u -> mt_uv for edge_type in EdgeType]
The message fn must incorporate the distance from node to node within the message... perhaps a linear discounting wrt to distance?
How would that constraint be formulated? Maybe just pass the distance as an input to the messageFn?

2. aggregation fn :: Agg ([mvu_t | v in N_in(u)]) :: [mvu_t] -> mt_u
3. status update fn :: [GRU or LSTM :: (ht_u, mt_u) for node_type in node_type]

output model:
1. MLP :: hu_t -> muu_t
2. sigmau_t :: R^u
3. output_distribution :: Gaussian (mu_u, sigma_u)
4. action selected :: output_distribution -> [au_t | u in O]
4. policyTheta = reduce (*) [(1/(2*pi*sigma_u**2)**(1/2))*e**((au_t - muu_t)**2/(2*sigma_u**2)) for (au_t, muu_t, sigma_u in O)]
--}

{--
data MDP = MDP
  (forall . (s a theta)) =>
  { stateSpace :: s
  ,  actionSpace :: a
  ,  policyNetwork :: theta
  , valueNetwork :: theta
  }
--}





{--
data NerveNetSpec (dtype :: D.DType)
                  (obsDim :: Nat) (stateDim :: Nat) (controlDim :: Nat)
  where
--}

main :: IO ()
main = putStrLn "NerveNet Started!"
