-- |
-- Module      : Numeric.MaxEnt.Train.Internal
-- Copyright   : (c) 2010 Daniël de Kok
-- License     : Apache 2
--
-- Maintainer  : Daniël de Kok <me@danieldk.eu>
-- Stability   : experimental
--
-- Non-public data structures for training maximum entropy models.

module Numeric.MaxEnt.Train.Internal (FeatureMapping(..),
                                      TrainContext(..),
                                      TrainCorpus(..),
                                      TrainEvent(..),
                                      ctxsToNum,
                                      emptyMapping,
                                      normalizeTrainCorpus,
                                     ) where

import qualified Data.Map as M

import Numeric.MaxEnt

data TrainCorpus a = TrainCorpus (FeatureMapping a) [TrainContext]

data FeatureMapping a = FeatureMapping {
      featureInt   :: M.Map a Int,
      intFeature   :: M.Map Int a,
      featureCount :: Int
}

data TrainContext = TrainContext {
      trainCtxP      :: Double,
      trainCtxEvents :: [TrainEvent]
} deriving (Show)

data TrainEvent = TrainEvent {
      trainEvtP             :: Double,
      trainEvtFeatureValues :: [(Int, Double)]
} deriving (Show)

emptyMapping :: (FeatureMapping a, [TrainContext])
emptyMapping = (FeatureMapping M.empty M.empty (0), [])

ctxsToNum :: Ord a => (FeatureMapping a, [TrainContext]) -> [Context a] ->
             (FeatureMapping a, [TrainContext])
ctxsToNum acc = foldl ctxToNum acc

ctxToNum :: Ord a => (FeatureMapping a, [TrainContext]) -> Context a ->
            (FeatureMapping a, [TrainContext])
ctxToNum (m, ctxs) (Context evts) = (newM, TrainContext scoreSum trainEvts:ctxs)
    where scoreSum = sumScores evts
          (newM, trainEvts) = foldl evtToNum (m, []) evts


evtToNum :: Ord a => (FeatureMapping a, [TrainEvent]) -> Event a ->
            (FeatureMapping a, [TrainEvent])
evtToNum (m, evts) (Event score fVals) = (newM, evt:evts)
    where (newM, newFVals) = fsToNum m fVals
          evt = TrainEvent score newFVals

fsToNum :: Ord a => FeatureMapping a -> [(a, Double)] ->
           (FeatureMapping a, [(Int, Double)])
fsToNum m = foldl fToNum (m, [])

fToNum :: Ord a => (FeatureMapping a, [(Int, Double)]) -> (a, Double) ->
          (FeatureMapping a, [(Int, Double)])
fToNum (m@(FeatureMapping fInt intF count), acc) (f, val) =
    case M.lookup f fInt of
      Just i -> (m, (i, val):acc)
      Nothing -> (newMapping, (count, val):acc)
          where newCount = count + 1
                newMapping = FeatureMapping (M.insert f count fInt)
                                            (M.insert count f intF)
                                            newCount

sumScores :: [Event a] -> Double 
sumScores = foldl (\acc (Event score _) -> acc + score) 0.0

-- Normalize the scores of a training corpus, giving a real probability
-- distribution for contexts and events.
normalizeTrainCorpus :: [TrainContext] -> [TrainContext]
normalizeTrainCorpus ctxs = map (normalizeCtx (scoreSum ctxs)) ctxs
    where scoreSum = foldl (\acc (TrainContext score _) -> acc + score) 0.0
          normalizeCtx sum (TrainContext score evts) =
              TrainContext (score / sum) $ map (normalizeEvt sum) evts
          normalizeEvt sum (TrainEvent score fVals) =
              TrainEvent (score / sum) fVals
