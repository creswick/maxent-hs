-- |
-- Module      : Numeric.MaxEnt.Train
-- Copyright   : (c) 2010 Daniël de Kok
-- License     : Apache 2
--
-- Maintainer  : Daniël de Kok <me@danieldk.eu>
-- Stability   : experimental

module Numeric.MaxEnt.Train (EstimateData(..), FeatureValues, TrainCorpus,
                             estimate, estimateBy, progress_silent,
                             progress_verbose, toTrainCorpus) where

import Control.Monad (forM_, foldM, liftM, mapM, liftM)
import Control.Monad.ST (runST)
import Data.Array.Storable (StorableArray, readArray, writeArray)
import Data.Either
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Generic as G
import Data.Vector.Unboxed ((!))
import qualified Data.Vector.Generic.Mutable as GM
import Foreign.C.Types (CDouble, CInt)
import Numeric.LBFGS
import System.IO (stderr)
import Text.Printf

import Numeric.MaxEnt
import Numeric.MaxEnt.Train.Internal

type FeatureValues = V.Vector Double

-- | Data that is required during optimization.
data EstimateData = EstimateData [TrainContext] FeatureValues
    deriving Show

-- | 
-- Estimate maximum entropy model parameters. Uses the `progress_verbose`
-- function to report on progress.
estimate :: Ord a => TrainCorpus a -> Int ->
            IO (Either LBFGSResult (M.Map a Double))
estimate = estimateBy progress_verbose

-- |
-- Estimate maximum entropy model parameters.
estimateBy :: Ord a => ProgressFun EstimateData -> TrainCorpus a -> Int ->
              IO (Either LBFGSResult (M.Map a Double))
estimateBy progress (TrainCorpus featureMapping corpus) nFeatures = do
  (r, weights) <- lbfgs params maxent_evaluate
                 progress lbfgsData $ take nFeatures $ repeat 0.0
  return $ case r of
             Success          -> Right $ n2f weights
             Stop             -> Right $ n2f weights
             AlreadyMinimized -> Right $ n2f weights
             _                -> Left  r
    where fVals = featureValues corpus nFeatures
          params = LBFGSParameters (Just 1) 1e-7 DefaultLineSearch Nothing
          lbfgsData = EstimateData corpus fVals
          n2f = numbersToFeatures featureMapping


numbersToFeatures :: Ord a => FeatureMapping a -> [Double] ->
                     M.Map a Double
numbersToFeatures (FeatureMapping _ intToFeatureMapping _) weights =
    foldl insertWeight M.empty indexedWeights
    where indexedWeights = zip [0..] weights
          insertWeight acc (idx, weight) = M.insert f weight acc
              where f = case M.lookup idx intToFeatureMapping of
                          Just val -> val
                          Nothing  -> error "Feature does not occur in mapping."

toTrainCorpus :: Ord a => [Context a] -> TrainCorpus a
toTrainCorpus ctxs = TrainCorpus mapping (normalizeTrainCorpus corpus)
    where (mapping, corpus) = ctxsToNum emptyMapping ctxs

-- Calculate the empirical value of features.
featureValues :: [TrainContext] -> Int -> V.Vector Double
featureValues ctxs n = runST $ do
  v <- GM.replicate n 0.0
  forM_ ctxs (fvCtx v)
  G.unsafeFreeze v
    where fvCtx v (TrainContext _ evts) = forM_ evts (fvEvt v)
          fvEvt v (TrainEvent p fVals) = forM_ fVals (fvMap v p) 
          fvMap v p (feature, value) = do
                          cur <- GM.unsafeRead v feature
                          GM.unsafeWrite v feature (cur + p * value)

pyxs :: StorableArray Int CDouble -> TrainContext -> IO [Double]
pyxs v (TrainContext _ evts) = do
  s <- sums
  let z = sum s
  return $ map (/ z) s
    where sums = mapM calcSum evts
          calcSum (TrainEvent _ fVals) = liftM exp $ foldM calcSum_ 0.0 fVals
          calcSum_ acc (feature, val) = do
               w <- readArray v feature
               return $ acc + ((realToFrac w) * val)

maxent_evaluate :: EstimateData -> StorableArray Int CDouble ->
                   StorableArray Int CDouble -> CInt -> CDouble -> IO CDouble
maxent_evaluate instData x g n step = do
  let (EstimateData corpus fVals) = instData
  let nVars = fromIntegral n
  initialGradients fVals g nVars
  ll <- foldM (ctxLL x g) 0.0 corpus
  return $ - (realToFrac ll)

-- Initial variable gradients are -p~(f).
initialGradients :: V.Vector Double -> StorableArray Int CDouble ->
                    Int -> IO ()
initialGradients fVals g nVars =
    mapM_ (\idx -> writeArray g idx $ realToFrac $ -(fVals ! idx)) [0..nVars - 1]

-- Calculate the log-likelihood contribution of a context.
ctxLL :: StorableArray Int CDouble -> StorableArray Int CDouble -> Double ->
         TrainContext -> IO (Double)
ctxLL x g acc ctx@(TrainContext pCtx evts)
    | pCtx == 0 = return acc
    | otherwise = do
  p <- pyxs x ctx
  ll <- foldM (evtLL x g pCtx) acc $ zip evts p
  return ll

-- Calculate the log-likelyhood contribution of an event.
evtLL :: StorableArray Int CDouble -> StorableArray Int CDouble -> Double ->
         Double -> (TrainEvent, Double) -> IO (Double)
evtLL x g pCtx acc (TrainEvent evtP features, pYx) = do
  let ll = acc + evtP * (log pYx)
  mapM_ (updateGradient g pCtx pYx) features
  return ll

-- Update variable gradients.
updateGradient :: StorableArray Int CDouble -> Double -> Double ->
                  (Int, Double) -> IO ()
updateGradient g pCtx pYx (f, val) = do
  cur <- readArray g f
  let new = (realToFrac cur) + pCtx * pYx * val
  writeArray g f (realToFrac new)

-- | Silent progress function. Does not report on progress at all.
progress_silent :: a -> StorableArray Int CDouble ->
                    StorableArray Int CDouble -> CDouble -> CDouble ->
                    CDouble -> CDouble -> CInt -> CInt ->
                    CInt -> IO (CInt)
progress_silent _ _ _ _ _ _ _ _ _ _ = return 0

-- |
-- Verbose progress function, reports the value of the objective function
-- and the gradient norm.
progress_verbose :: a -> StorableArray Int CDouble ->
                    StorableArray Int CDouble -> CDouble -> CDouble ->
                    CDouble -> CDouble -> CInt -> CInt ->
                    CInt -> IO (CInt)
progress_verbose _ _ _ fx xnorm gnorm _ _ k _ = do
  hPrintf stderr "%d\t%.4e\t%.4e\t%.4e\n" (fromIntegral k :: Int)
              (realToFrac fx :: Double) (realToFrac xnorm :: Double)
              (realToFrac gnorm :: Double)
  return 0

-- corpus = [ Context [
--            Event 1.0 [("f1", 1)],
--            Event 0.5 [("f1", 0),("f2", 1)]
--           ],
--           Context [
--            Event 1.0 [("f1", 2)]
--           ]
--         ]

