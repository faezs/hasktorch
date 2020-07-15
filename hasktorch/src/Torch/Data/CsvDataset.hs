{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Torch.Data.CsvDataset ( CsvDataset(..)
                             , csvDataset
                             , NamedColumns
                             , FromField(..)
                             , FromRecord(..)
                             ) where

import           Torch.Typed
import qualified Torch.DType as D
import           Data.Reflection (Reifies(reflect))
import           Data.Proxy (Proxy(Proxy))
import           GHC.Exts (IsList(fromList))
import           Control.Monad 
import qualified Data.Vector as V
import qualified Torch.Tensor as D
import           GHC.TypeLits (KnownNat)
import           Torch.Data.StreamedPipeline
import           Pipes.Safe

import qualified Control.Foldl as L
import           Control.Foldl.Text (Text)
import           Control.Monad.Base (MonadBase)
import           Data.ByteString (hGetLine, hGetContents)
import           Data.Set.Ordered as OSet hiding (fromList)
import           Lens.Family (view)
import           Pipes (liftIO, ListT(Select), yield, (>->), await)
import qualified Pipes.ByteString as B
import           Pipes.Csv
import           Pipes.Group (takes, folds, chunksOf)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import           System.IO (IOMode(ReadMode))
import Pipes.Concurrent (unbounded)
import Control.Monad.Trans.Control (MonadBaseControl)


-- instance FromField
instance FromField a => FromField [a] where
  -- simply wrap a single 'a' into a list
  parseField = fmap pure . parseField 

instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Int]
         )
    => FromRecord (Tensor device 'Int64 '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

  -- these two instances actually don't make sense right now
  -- since fields only work between each delimiter
instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Float]
          )
    => FromRecord (Tensor device 'D.Float '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

data NamedColumns = Unnamed | Named
data CsvDataset batches = CsvDataset { filePath :: FilePath
                                     , decDelimiter :: !B.Word8
                                     , byName :: NamedColumns
                                     , hasHeader :: HasHeader
                                     , batchSize :: Int
                                     , filter :: Maybe (batches -> Bool)
                                     , numBatches :: Maybe Int
                                     }

csvDataset :: forall batches . FilePath -> CsvDataset batches
csvDataset filePath  = CsvDataset { filePath = filePath
                                  , decDelimiter = 44 -- comma
                                  , byName = Unnamed
                                  , hasHeader = NoHeader
                                  , batchSize = 1
                                  , filter = Nothing
                                  , numBatches = Nothing
                                  }

instance ( MonadPlus m
         , MonadBase IO m
         , MonadBaseControl IO m
         , Safe.MonadSafe m
         , FromRecord batch -- these constraints make CsvDatasets only able to parse records, might not be the best idea
         , FromNamedRecord batch
         -- , Monoid batch
         ) => Datastream m () (CsvDataset batch) [batch] where
  streamBatch CsvDataset{..} _ = Select $ Safe.withFile filePath ReadMode $ \fh ->
      -- this quietly discards errors right now, probably would like to log this
    -- TODO : optionally take a fixed number of batches
     -- L.purely folds L.mconcat $ view (chunksOf batchSize) $ decodeRecords fh >-> P.concat
     L.purely folds L.list $ view (chunksOf batchSize) $ decodeRecords fh >-> P.concat

  -- TODO: we could concurrently stream in records, and batch records in another thread
        where
          decodeRecords fh = case byName of
                               Unnamed -> decode hasHeader (produceLine fh)
                               Named   -> decodeByName (produceLine fh)
          -- what's a good default chunk size? 
          produceLine fh = B.hGetSome 1000 fh

  -- TODO: add shuffles with a fixed buffer size
