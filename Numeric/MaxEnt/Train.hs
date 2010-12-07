module Numeric.MaxEnt.Train (EstimateData(..), FeatureValues, TrainCorpus(..),
                             TrainContext(..), TrainEvent(..), estimate,
                             estimateBy, progress_silent,
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

data TrainCorpus a = TrainCorpus (FeatureMapping a) [TrainContext]

data TrainContext = TrainContext {
      trainCtxP      :: Double,
      trainCtxEvents :: [TrainEvent]
} deriving (Show)

data TrainEvent = TrainEvent {
      trainEvtP             :: Double,
      trainEvtFeatureValues :: [(Int, Double)]
} deriving (Show)

type FeatureValues = V.Vector Double

data FeatureMapping a = FeatureMapping {
      featureInt   :: M.Map a Int,
      intFeature   :: M.Map Int a,
      featureCount :: Int
}

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

emptyMapping :: (FeatureMapping a, [TrainContext])
emptyMapping = (FeatureMapping M.empty M.empty (-1), [])

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
      Nothing -> (newMapping, (newCount, val):acc)
          where newCount = count + 1
                newMapping = FeatureMapping (M.insert f newCount fInt)
                                            (M.insert newCount f intF)
                                            newCount

sumScores :: [Event a] -> Double 
sumScores = foldl (\acc (Event score _) -> acc + score) 0.0

numbersToFeatures :: Ord a => FeatureMapping a -> [Double] ->
                     M.Map a Double
numbersToFeatures (FeatureMapping _ intToFeatureMapping _) weights =
    foldl insertWeight M.empty indexedWeights
    where indexedWeights = zip [0..] weights
          insertWeight acc (idx, weight) = M.insert f weight acc
              where f = case M.lookup idx intToFeatureMapping of
                          Just val -> val
                          Nothing  -> error "Feature does not occur in mapping."

-- Normalize the scores of a training corpus, giving a real probability
-- distribution for contexts and events.
normalizeTrainCorpus :: [TrainContext] -> [TrainContext]
normalizeTrainCorpus ctxs = map (normalizeCtx (scoreSum ctxs)) ctxs
    where scoreSum = foldl (\acc (TrainContext score _) -> acc + score) 0.0
          normalizeCtx sum (TrainContext score evts) =
              TrainContext (score / sum) $ map (normalizeEvt sum) evts
          normalizeEvt sum (TrainEvent score fVals) =
              TrainEvent (score / sum) fVals


toTrainCorpus :: Ord a => [Context a] -> TrainCorpus a
toTrainCorpus ctxs = TrainCorpus mapping corpus
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

