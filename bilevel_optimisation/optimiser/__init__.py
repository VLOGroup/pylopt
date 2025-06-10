from bilevel_optimisation.optimiser.NAGOptimiser import NAGOptimiser, UnrollingNAGOptimiser
from bilevel_optimisation.optimiser.AlternatingNAGOptimiser import AlternatingNAGOptimiser
from bilevel_optimisation.optimiser.StoppingCriterion import EarlyStopping, FixedIterationsStopping

NAG_TYPE_OPTIMISER = [NAGOptimiser.__name__, AlternatingNAGOptimiser.__name__]      # List of optimisers which are of NAG type
                                                                                    # and need a specific closure function

UNROLLING_TYPE_OPTIMISER = [UnrollingNAGOptimiser.__name__]                         # List of optimisers which are suitable for
                                                                                    # bilevel unrolling scheme