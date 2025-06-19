from bilevel_optimisation.optimise.NAGOptimiser import NAGOptimiser
from bilevel_optimisation.optimise.UnrollingNAGOptimiser import UnrollingNAGOptimiser
from bilevel_optimisation.optimise.StoppingCriterion import EarlyStopping, FixedIterationsStopping

NAG_TYPE_OPTIMISER = [NAGOptimiser.__name__]      # List of optimisers which are of NAG type
                                                  # and need a specific closure function

UNROLLING_TYPE_OPTIMISER = [UnrollingNAGOptimiser.__name__]                         # List of optimisers which are suitable for
                                                                                    # bilevel unrolling scheme



# ### NEW

from bilevel_optimisation.optimise.optimise_nag import optimise_nag