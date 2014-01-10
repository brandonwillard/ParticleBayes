package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.gaussian.GaussianArHpHmmPlFilter;
import plm.hmm.gaussian.GaussianArHpTransitionState;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.util.ObservedValue;

public class GaussianArHpHmmPLAdapter {

  public static class GaussianArHpHMMPLResult {

    private final int[][] classIds;
    private final double[][][] stateMeans;
    private final double[][][] stateCovs;
    private final double[][] covDof;
    private final double[][][] covScale;
    private final double[][] logWeights;

    public GaussianArHpHMMPLResult(int[][] classIds, double[][] logWeights,
        double[][][] stateMeans, double[][][] stateCovs, double [][] covDof, 
        double[][][] covScale) {
      this.stateMeans = stateMeans;
      this.stateCovs = stateCovs;
      this.covDof = covDof;
      this.covScale = covScale;
      this.classIds = classIds; 
      this.logWeights = logWeights;
    }

    public double[][][] getStateMeans() {
      return this.stateMeans;
    }

    public double[][][] getStateCovs() {
      return this.stateCovs;
    }

    public double[][] getCovDof() {
      return this.covDof;
    }

    public double[][][] getCovScale() {
      return this.covScale;
    }

    public int[][] getClassIds() {
      return this.classIds;
    }

    public double[][] getLogWeights() {
      return this.logWeights;
    }
    
  }


  /**
   * A HMM with discrete observations/responses/emissions.
   * 
   * @param hmmInitialProb
   * @param hmmTransitionProb
   * @param emissionProbs rows span components, columns span observation support 
   * @param y
   * @param N
   * @param resampleOnly 
   * @param seed
   * @return
   */
  public static GaussianArHpHMMPLResult batchUpdate(
      double[] hmmInitialProb, double[][] hmmTransitionProb, 
      double[][] initialStateMeans, double[][][] initialStateCovs,
      double[][][] initialFs, double[][][] initialGs,
      double[][][] initialVs, double[][][] initialWs,
      double[][] y, int N, boolean resampleOnly, int seed) {
    
    final Random rng = new Random(seed);

    List<KalmanFilter> filters = Lists.newArrayList();
    for (int i = 0; i < hmmInitialProb.length; i++) {
      KalmanFilter kf = new KalmanFilter(model1, 
          modelCovariance1, measurementCovariance);
      kf.setCurrentInput(VectorFactory.getDefault().copyValues(
          psis.get(0).getElement(0)));
      filters.add(kf);
    }

    DlmHiddenMarkovModel hmm = new DlmHiddenMarkovModel(
        filters,
        VectorFactory.getDefault().copyArray(hmmInitialProb),
        MatrixFactory.getDefault().copyArray(hmmTransitionProb));

    final HmmPlFilter<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> filter =
        new GaussianArHpHmmPlFilter(hmm, sigmaPrior, priorPhis, rng, resampleOnly);
    filter.setNumParticles(N);

    DataDistribution<GaussianArHpTransitionState> currentDist = filter.createInitialLearnedObject();
    double [][] logWeights = new double[y.length][filter.getNumParticles()];
    int [][] classIds = new int[y.length][filter.getNumParticles()];
    
    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
    classIds[0] = 
        Ints.toArray(Collections2.transform(currentDist.asMap().keySet(), 
        new Function<HmmTransitionState<Integer, StandardHMM<Integer>>, Integer>() {
          @Override
          public Integer apply(HmmTransitionState<Integer, StandardHMM<Integer>> input) {
            return input.getClassId();
          }}));

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Void> obs = ObservedValue.<Vector>create(
          t, 
          VectorFactory.getDefault().copyArray(y[t]));
      filter.update(currentDist, obs);

      logWeights[t] = new double[filter.getNumParticles()];
      classIds[t] = new int[filter.getNumParticles()];
      int k = 0;
      for (Entry<HmmTransitionState<Integer, StandardHMM<Integer>>, ? extends Number> entry : currentDist.asMap().entrySet()) {
        final int count;
        if (entry.getValue() instanceof MutableDoubleCount) {
          count = ((MutableDoubleCount)entry.getValue()).count;
        } else {
          count = 1;
        }
        for (int r = 0; r < count; r++) {
          // FIXME might not be log scale
          final double normLogWeight = entry.getValue().doubleValue() - Math.log(count) - currentDist.getTotal();
          Preconditions.checkState(normLogWeight < 0d);
          logWeights[t][k] = normLogWeight;
          classIds[t][k] = entry.getKey().getClassId();
          k++;
        }
      }
    }

    return new GaussianArHpHMMPLResult(classIds, logWeights);
  }
}
