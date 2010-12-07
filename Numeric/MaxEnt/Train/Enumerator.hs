-- |
-- Module      : Numeric.MaxEnt.Train.Enumerator
-- Copyright   : (c) 2010 Daniël de Kok
-- License     : Apache 2
--
-- Maintainer  : Daniël de Kok <me@danieldk.eu>
-- Stability   : experimental

{-# LANGUAGE BangPatterns #-}

module Numeric.MaxEnt.Train.Enumerator (toTrainCorpus) where

import Data.Enumerator hiding (isEOF, length, map)
import qualified Data.Map as M
import Numeric.MaxEnt (Context(..))

import Numeric.MaxEnt.Train.Internal

toTrainCorpus :: (Monad m, Ord a) =>
            Iteratee (Context a) m (TrainCorpus a)
toTrainCorpus = liftI $ step emptyMapping where
    step acc@(FeatureMapping !i !f n, !corpus) chunk = case chunk of
                       Chunks [] -> Continue $ returnI . step acc
                       Chunks xs -> Continue $ returnI .
                                    (step $ ctxsToNum acc xs)
                       EOF -> Yield (TrainCorpus mapping normCtxs) EOF
                              where (mapping, ctxs) = acc
                                    normCtxs = normalizeTrainCorpus ctxs
