# pylint: disable=abstract-method

""" ReLeASE """

import sys
sys.path.append('../autotvm/tuner')
from model_based_tuner import ModelBasedTuner, ModelOptimizer, CostModel
from xgboost_cost_model import XGBoostCostModel
from sa_model_optimizer import SimulatedAnnealingOptimizer
from search import ReinforcementLearningSearch

class ReLeASE(ModelBasedTuner):
    # rank --> reg because we need regression of the true value over mere relative rank
    def __init__(self, task, cost_model='xgb', rnn_params=None, plan_size=64,
                 feature_type='itervar', loss_type='reg', num_threads=None,
                 optimizer='rl', diversity_filter_ratio=None, log_interval=50):
        
        if cost_model == 'xgb':
            cost_model = XGBoostCostModel(task,
                                          feature_type=feature_type,
                                          loss_type=loss_type,
                                          num_threads=num_threads,
                                          log_interval=log_interval // 2)
        else:
            assert isinstance(optimizer, CostModel), "CostModel must be " \
                                                     "a supported name string" \
                                                     "or a CostModel object."

        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        elif optimizer == 'rl':
            optimizer = ReinforcementLearningOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(RLTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio)

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(RLTuner, self).tune(*args, **kwargs)

        ## manually close pool to avoid multiprocessing issues
        #self.cost_model._close_pool()
        #self.model_optimizer.agent._close_session()
