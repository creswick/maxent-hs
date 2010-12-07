-- |
-- Module      : Numeric.MaxEnt.Train.Enumerator
-- Copyright   : (c) 2010 Daniël de Kok
-- License     : Apache 2
--
-- Maintainer  : Daniël de Kok <me@danieldk.eu>
-- Stability   : experimental

module Numeric.MaxEnt.Train.Enumerator (trainCorpus) where

import Data.Enumerator hiding (isEOF, length, map)
import qualified Data.Map as M
import Numeric.MaxEnt (Context(..))

import Numeric.MaxEnt.Train.Internal

trainCorpus :: (Monad m, Ord a) =>
            Iteratee (Context a) m (TrainCorpus a)
trainCorpus = liftI $ step emptyMapping where
    step acc chunk = case chunk of
                       Chunks [] -> Continue $ returnI . step acc
                       Chunks xs -> Continue $ returnI .
                                    (step $ ctxsToNum acc xs)
                       EOF -> Yield (TrainCorpus mapping normCtxs) EOF
                              where (mapping, ctxs) = acc
                                    normCtxs = normalizeTrainCorpus ctxs
