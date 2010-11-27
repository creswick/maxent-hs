module Numeric.MaxEnt (
                       Context(..),
                       Event(..),
                      ) where

data Context a = Context [Event a]
               deriving (Show)

data Event a = Event {
      evtFrequency     :: Double,
      evtFeatureValues :: [(a, Double)]
    } deriving (Show)

