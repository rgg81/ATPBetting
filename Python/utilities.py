#!/usr/bin/env python3
# -*- coding: utf-8 -*-

    
    
############################### STORAGE ############################
## Some useful functions to store and load data

import pickle
def dump(obj,name):
	pickle.dump(obj,open(name+'.p',"wb")) 
def load(name):
	obj=pickle.load( open( name+".p", "rb" ) ) 
	return obj


def _fmt_metric(value, show_stdv=True):
	"""format metric string"""
	if len(value) == 2:
		return '%s:%g' % (value[0], value[1])
	if len(value) == 3:
		if show_stdv:
			return '%s:%g+%g' % (value[0], value[1], value[2])
		return '%s:%g' % (value[0], value[1])
	raise ValueError("wrong metric value")


def early_stop(stopping_rounds, maximize=False, verbose=True, min_iteration=0):
	from xgboost import rabit
	from xgboost.core import EarlyStopException
	"""Create a callback that activates early stoppping.

    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)

    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.

    maximize : bool
        Whether to maximize evaluation metric.

    verbose : optional, bool
        Whether to print message about early stopping information.

    Returns
    -------
    callback : function
        The requested callback function.
    """
	state = {}

	def init(env):
		"""internal function"""
		bst = env.model

		if not env.evaluation_result_list:
			raise ValueError('For early stopping you need at least one set in evals.')
		if len(env.evaluation_result_list) > 1 and verbose:
			msg = ("Multiple eval metrics have been passed: "
				   "'{0}' will be used for early stopping.\n\n")
			rabit.tracker_print(msg.format(env.evaluation_result_list[-1][0]))
		maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg')
		maximize_at_n_metrics = ('auc@', 'aucpr@', 'map@', 'ndcg@')
		maximize_score = maximize
		metric_label = env.evaluation_result_list[-1][0]
		metric = metric_label.split('-', 1)[-1]

		if any(metric.startswith(x) for x in maximize_at_n_metrics):
			maximize_score = True

		if any(metric.split(":")[0] == x for x in maximize_metrics):
			maximize_score = True

		if verbose and env.rank == 0:
			msg = "Will train until {} hasn't improved in {} rounds.\n"
			rabit.tracker_print(msg.format(metric_label, stopping_rounds))

		state['maximize_score'] = maximize_score
		state['best_iteration'] = 0
		if maximize_score:
			state['best_score'] = float('-inf')
		else:
			state['best_score'] = float('inf')

		if bst is not None:
			if bst.attr('best_score') is not None:
				state['best_score'] = float(bst.attr('best_score'))
				state['best_iteration'] = int(bst.attr('best_iteration'))
				state['best_msg'] = bst.attr('best_msg')
			else:
				bst.set_attr(best_iteration=str(state['best_iteration']))
				bst.set_attr(best_score=str(state['best_score']))
		else:
			assert env.cvfolds is not None

	def callback(env):
		"""internal function"""
		if env.iteration >= min_iteration:
			score = env.evaluation_result_list[-1][1]
			if not state:
				init(env)
			best_score = state['best_score']
			best_iteration = state['best_iteration']
			maximize_score = state['maximize_score']
			if (maximize_score and score > best_score) or \
					(not maximize_score and score < best_score):
				msg = '[%d]\t%s' % (
					env.iteration,
					'\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
				state['best_msg'] = msg
				state['best_score'] = score
				state['best_iteration'] = env.iteration
				# save the property to attributes, so they will occur in checkpoint.
				if env.model is not None:
					env.model.set_attr(best_score=str(state['best_score']),
									   best_iteration=str(state['best_iteration']),
									   best_msg=state['best_msg'])
			elif env.iteration - best_iteration >= stopping_rounds:
				best_msg = state['best_msg']
				if verbose and env.rank == 0:
					msg = "Stopping. Best iteration:\n{}\n\n"
					rabit.tracker_print(msg.format(best_msg))
				raise EarlyStopException(best_iteration)

	return callback
