import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from random import randint

# HEXDEV-700: GBM reproducibility issue.
def gbm_reproducibility():

  # run GBM twice with true_reproducibility = True
  seedv = randint(1,10000000000)
  auc1 = runGBM(seedv, True)
  auc2 = runGBM(seedv, True)
  pyunit_utils.equal_two_arrays(auc1, auc2, 1e-10, True)  # should be equal in this case

  auc3 = runGBM(seedv, False)  # threshold should be different this run.
  assert not(pyunit_utils.equal_two_arrays(auc2, auc3, 1e-10, False)), "parameter true_reproducibility is not working."


def runGBM(seedV, repo):
  cars = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/jira/reproducibility_issue.csv.zip"))
  gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=50, seed= seedV, max_depth = 4,
                                   min_rows = 7, true_reproducibility=repo)
  gbm.train(x=list(range(2,365)), y="response", training_frame=cars)
  auc = pyunit_utils.extract_from_twoDimTable(gbm._model_json['output']['training_metrics']._metric_json['thresholds_and_metric_scores'], 'threshold', takeFirst=False)
  h2o.remove(cars)
  h2o.remove(gbm)
  return auc

if __name__ == "__main__":
  pyunit_utils.standalone_test(gbm_reproducibility)
else:
  gbm_reproducibility()